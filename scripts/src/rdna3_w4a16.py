# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 GPTQ kernel for AMD RDNA3 (gfx1100) — fp16 + bf16.

Drop-in replacement for ExllamaLinearKernel on RDNA3 that adds native bf16
support. The HIP kernel lives in ``csrc/rocm/q_gemm_rdna3.cu``
and is exposed via ``torch.ops._rocm_C.gptq_gemm_rdna3``.

Registered ahead of TritonW4A16LinearKernel for the ROCm-RDNA3 path; falls
through to the Triton kernel on non-RDNA3 ROCm devices (e.g. CDNA/MI300).
"""

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import direct_register_custom_op

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

# --- HyperQwen: a single-read W4A16 GEMV for up to 8 activation rows (rocm/w4a16_gemv.cu) ---------------------
# gptq_gemm_rdna3 reads the weights once per 4 rows, so a 5-row speculative verify reads them twice. The GEMV reads
# them once for up to 8 rows, on the same weight layout. It is optional: without the shared library, or with
# HQ_W4A16_GEMV=0, or for fp16 activations, the original kernel runs.


def _load_hq_gemv() -> bool:
    import os

    if os.environ.get("HQ_W4A16_GEMV", "1") == "0":
        return False
    path = os.environ.get("HQ_W4A16_GEMV_SO", "/opt/hq/w4a16_gemv.so")
    if not os.path.exists(path):
        return False
    try:
        torch.ops.load_library(path)
    except Exception:
        return False
    return hasattr(torch.ops, "hq") and hasattr(torch.ops.hq, "w4a16_gemv")


_HQ_GEMV = _load_hq_gemv()
# Large batches (prefill chunks): dequantize to bf16 and run hipBLASLt's GEMM, ~10% faster than gptq_gemm_rdna3 at
# 1024-2048 rows, no gain below. HQ_W4A16_DQ_MIN_M=0 turns it off.
_HQ_DQ_MIN_M = int(__import__("os").environ.get("HQ_W4A16_DQ_MIN_M", "1024"))
_HQ_DQ = _HQ_GEMV and _HQ_DQ_MIN_M > 0 and hasattr(torch.ops.hq, "w4a16_dequant")


def _rdna3_w4a16_gemm_impl(
    a: torch.Tensor, w_q: torch.Tensor, w_zp: torch.Tensor, w_s: torch.Tensor
) -> torch.Tensor:
    if (
        _HQ_GEMV
        and a.shape[0] <= 8
        and a.dtype == torch.bfloat16
        and a.is_contiguous()
        and a.data_ptr() % 16 == 0
        and w_s.dtype == torch.bfloat16
        and w_s.shape[0] * 128 == a.shape[1]
        and w_s.is_contiguous()
    ):
        return torch.ops.hq.w4a16_gemv(a, w_q, w_zp, w_s, False, 0)
    if (
        _HQ_DQ
        and a.shape[0] >= _HQ_DQ_MIN_M
        and a.dtype == torch.bfloat16
        and w_s.dtype == torch.bfloat16
        and w_s.shape[0] * 128 == a.shape[1]
        and w_s.is_contiguous()
    ):
        return a @ torch.ops.hq.w4a16_dequant(w_q, w_zp, w_s, False)
    return torch.ops._rocm_C.gptq_gemm_rdna3(a, w_q, w_zp, w_s, False)


def _rdna3_w4a16_gemm_fake(
    a: torch.Tensor, w_q: torch.Tensor, w_zp: torch.Tensor, w_s: torch.Tensor
) -> torch.Tensor:
    return torch.empty((a.size(0), w_q.size(1)), dtype=a.dtype, device=a.device)


direct_register_custom_op(
    op_name="hq_rdna3_w4a16_gemm",
    op_func=_rdna3_w4a16_gemm_impl,
    mutates_args=[],
    fake_impl=_rdna3_w4a16_gemm_fake,
)


class RDNA3W4A16LinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]

    @classmethod
    def get_min_capability(cls) -> int:
        # ROCm gates via on_gfx1100() in can_implement.
        return 60

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RDNA3 W4A16 kernel is ROCm-only"

        from vllm.platforms.rocm import on_gfx1100

        if not on_gfx1100():
            return False, "RDNA3 W4A16 kernel requires gfx1100"

        # The HIP op is registered by the C++ extension; if a user is running
        # against a vLLM build that doesn't include it (e.g. partial rebuild),
        # fall through gracefully to the next kernel in the registry.
        if not (
            hasattr(torch.ops, "_rocm_C")
            and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
        ):
            return (
                False,
                "torch.ops._rocm_C.gptq_gemm_rdna3 missing — rebuild C++ extension",
            )

        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "RDNA3 W4A16 kernel only supports fp16 and bf16"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                f"RDNA3 W4A16 kernel; supported: {cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                "RDNA3 W4A16 kernel does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide K "
                f"({c.full_weight_shape[0]})",
            )

        # Output features must be a multiple of the pack factor (8 nibbles per
        # int32) and of 8 so that qzeros (packed 4-bit per col) align cleanly
        # against the BLOCK_KN_SIZE*4 = 512 N-stride and per-thread 4 columns.
        if c.partition_weight_shape[1] % 8 != 0:
            return (
                False,
                "Output features must be a multiple of 8 for the RDNA3 "
                "W4A16 kernel (qzeros packing)",
            )

        return True, None

    # ----- Weight prep (identical layout/shuffle as ExllamaLinearKernel) -----

    def process_weights_after_loading(self, layer: torch.nn.Module):
        c = self.config
        device = getattr(layer, self.w_q_name).device

        # Synthesize zero points if the checkpoint doesn't carry them.
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # GPTQv1 quirk: the kernel adds 1 to the stored zero, so we
                # encode (bias - 1) here. See exllama.py for the link to the
                # documentation of this checkpoint-format wart.
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "RDNA3 W4A16 kernel: zero-bias 4-bit quant requires "
                    "explicit zero points (GPTQv1 +1 quirk)."
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            replace_parameter(layer, self.w_zp_name, zeros)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            # Same 4-bit shuffle as exllama. The RDNA3 kernel reads weights in
            # the same shuffled int32 layout and uses the (qa & 0x000F000F)
            # bit-trick on top.
            ops.gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            # Keep scales in the activation dtype (fp16 OR bf16) — the kernel
            # branches on dtype internally.
            return x.to(dtype=c.act_type)

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    # ----- Forward --------------------------------------------------------

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        w_q, w_s, w_zp = self._get_weight_params(layer)
        assert w_zp is not None, "Zero points are required by RDNA3 W4A16"
        output = torch.ops.vllm.hq_rdna3_w4a16_gemm(x_2d, w_q, w_zp, w_s)

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
