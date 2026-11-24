#!/bin/bash
# HyperQwen on an AMD Radeon RX 7900 XTX (RDNA3, gfx1100), through ROCm.
#
# Same pattern as start-turing.sh: set the environment and EXTRA_ARGS for this card, then hand over to
# single-user/start_qwen.sh. Everything NVIDIA-specific in the repo is off here -- Marlin W4A8 (INT8_ACT),
# FlashInfer, the FA2 fp8-KV plugin (fa2/), the pth-flash-attn kernel (pthfa/), LSA and the CUDA-UVA
# embedding / KV offload tier -- and the attention is vLLM's Triton backend on a bf16 KV cache.
#
# STATUS: EXPERIMENTAL. With the defaults below (SPEC=mtp with 3 drafts, 8k context) Qwen3.8-27B boots on a 7900 XTX
# through Docker Desktop and decodes 93-98 tok/s on the repo's real-prompt cohort at one stream (bench/run_benchmarks.sh,
# one machine, with the W4A16 GEMV of rocm/w4a16_gemv.cu); SPEC=off is ~44 tok/s. HQ_ROCM_KV=octave reaches a 150k
# window and decodes 88 tok/s on bench/labd_bench.py at a 74k prompt (the 3090's 84.7 at ~66k). SPEC=dflash2 runs but is slower (69.5 tok/s before the GEMV). Quality and a soak have not
# been run. docs/rocm.md lists what is verified and what is not.
#
# Everything below can be overridden from the environment, e.g.
#   SPEC=off bash start-rocm.sh                 # no speculation
#   HQ_ROCM_KV=octave MAX_LEN=150000 MAX_SEQS=1 bash start-rocm.sh   # 150k context, 3-bit KV cache
#   MAX_LEN=16384 GPU_UTIL=0.92 bash start-rocm.sh
#   KV_MEM=3000000000 bash start-rocm.sh        # pin the KV pool in bytes instead of sizing it from GPU_UTIL
#   HQ_ROCM_W4A16_KERNEL=exllama bash start-rocm.sh   # force vLLM's exllama GEMM instead of the RDNA3 kernel
#   EXTRA_ARGS="--enforce-eager" bash start-rocm.sh   # no CUDA/HIP graphs, for debugging
#
# VRAM is the constraint on Windows: the desktop (dwm) and any other GPU program take their share of the
# 24 GiB before vLLM starts, and vLLM refuses to boot when the free memory is below GPU_UTIL of the total.
# Close whatever uses the GPU first. The server refuses to start rather than overcommit, because under WSL2
# an overcommitted card silently spills into system RAM (docs/gotchas.md).

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- the GPU runtime --------------------------------------------------------------------------------
# On WSL2 (Docker Desktop included) the card is /dev/dxg, reached through librocdxg, not /dev/kfd.
if [ -e /dev/dxg ]; then
  export HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION:-1}   # needed before ROCm 7.13, harmless after
  export ROCPROFILER_REGISTER_ENABLED=${ROCPROFILER_REGISTER_ENABLED:-0}   # it looks for /dev/kfd at load
  case ":${LD_LIBRARY_PATH:-}:" in *":/usr/lib/wsl/lib:"*) ;; *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/lib/wsl/lib" ;; esac
  # The image's ROCr runtime rebuilt with the AsyncEventsLoop poll backoff (rocm/rocr/): without it every process that
  # touches the GPU spins two threads at 100% CPU (an idle server read ~420%, with it ~2%). HQ_ROCR_PATCH=0 uses the
  # stock runtime.
  if [ "${HQ_ROCR_PATCH:-1}" != 0 ] && [ -f /opt/hq/rocr/libhsa-runtime64.so.1 ]; then
    export LD_LIBRARY_PATH="/opt/hq/rocr:${LD_LIBRARY_PATH:-}"
  fi
fi
# PyTorch < 2.14 on RDNA3 picks a slow default GEMM backend for batch >= 8 decode (AMD release notes).
export TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT:-1}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
# Triton's AMD backend is the one in the vLLM wheel; the flag below is the AMD flash-attn Triton port's gate and
# costs nothing when that package is absent.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}

case "${HQ_ROCM_W4A16_KERNEL:-auto}" in
  auto) ;;
  # vLLM's own ROCm kernel order for compressed-tensors W4A16 is RDNA3 HIP kernel, then Conch (Triton), then exllama.
  # Disabling Conch makes it fall through to the exllama GEMM (gptq_gemm), the fallback if the RDNA3 kernel misbehaves.
  exllama) export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:+$VLLM_DISABLED_KERNELS,}ConchLinearKernel ;;
  *) echo "HQ_ROCM_W4A16_KERNEL=${HQ_ROCM_W4A16_KERNEL}: use auto or exllama" >&2; exit 1 ;;
esac

# --- what this card cannot run -----------------------------------------------------------------------
# Refuse an explicit request for an NVIDIA-only arm instead of booting something else under its name.
for k in INT8_ACT PTH_FA FA2_FP8KV LSA EMBED_UVA KV_OFFLOAD_GB PREFILL_ATTN; do
  v=${!k-}
  case "$v" in ""|0) ;; *)
    echo "start-rocm.sh: $k=$v is an NVIDIA-only setting (Marlin, FlashInfer, CUDA kernels or CUDA UVA); refusing." >&2
    echo "  See docs/rocm.md, \"What does not run on ROCm\"." >&2
    exit 1 ;;
  esac
done
unset INT8_ACT PREFILL_ATTN KV_OFFLOAD_GB
export PTH_FA=0 FA2_FP8KV=0 LSA=0 EMBED_UVA=0

# --- the arm ------------------------------------------------------------------------------------------
export CTX=${CTX:-fast}
case "$CTX" in
  fast) ;;
  *) echo "start-rocm.sh: CTX=$CTX is not available here yet (only CTX=fast; docs/rocm.md, \"Long context\")." >&2; exit 1 ;;
esac
export SPEC=${SPEC:-mtp}
# MTP drafts 3 tokens, not the launcher's 4. The stock RDNA3 W4A16 kernel (gptq_gemm_rdna3) reads the weights once for
# up to 4 rows and a second time for the 5th: with 4 drafts a step cost ~43 ms (65 tok/s on the repo's real-prompt
# cohort), with 3 ~31 ms (84 tok/s). With the single-read GEMV (rocm/w4a16_gemv.cu, in the image) 3 drafts give 93-98
# tok/s and 4 give 95, no better, so 3 stays.
export DRAFT_TOKENS=${DRAFT_TOKENS:-3}
export SPEC_ATTN=${SPEC_ATTN:-0}   # the split-KV verify kernel (patches/spec-decode-attn.patch) is untested on ROCm
export DRAFT_ATTN_BACKEND=${DRAFT_ATTN_BACKEND:-TRITON_ATTN}
# HQ_ROCM_KV=octave: 3-bit KV cache (octave_k3v3_compact) on vLLM's ROCM_OCTAVE backend, for the long contexts the
# bf16 cache cannot hold (a 150k window fits with a 505k-token pool). The drafter uses the same backend.
case "${HQ_ROCM_KV:-bf16}" in
  bf16) ;;
  octave)
    export DRAFT_ATTN_BACKEND=ROCM_OCTAVE
    # 192 KV splits, not vLLM's 32: the decode kernel needs them to fill the 48 CUs with one sequence, and its WMMA
    # tier caps the count at 192 (kSqWmmaSplits in octave_decode_wmma_rdna3.cu), so more is not used. One layer over
    # a 125k cache, 4 verify rows: 573 us at 32 splits, ~400 us at 192+; 1 draft row: 429 us -> 210 us. No loss at
    # short contexts (1k: 58 -> 71 us, 8k: 62 -> 53 us). docs/rocm.md, "Long context".
    EXTRA_ARGS="--attention-backend ROCM_OCTAVE --kv-cache-dtype octave_k3v3_compact --attention-config.tq_max_kv_splits_for_cuda_graph 192 ${EXTRA_ARGS:-}" ;;
  *) echo "HQ_ROCM_KV=${HQ_ROCM_KV}: use bf16 or octave" >&2; exit 1 ;;
esac
export PREFIX_CACHE=${PREFIX_CACHE:-1}

# Conservative, unmeasured: the 27B W4A16 weights are ~16 GiB, and on Windows ~4 GiB of the 24 GiB card go to the
# desktop and the WDDM/dxg overhead before vLLM allocates anything. Raise these one at a time and watch the free
# memory the engine logs at startup.
export MAX_LEN=${MAX_LEN:-8192}
export MAX_SEQS=${MAX_SEQS:-2}
# 0.86, not 0.90: at 0.90 the card sat at 23.9 of 24 GiB during long prefills with the Windows desktop on the same
# GPU, and WDDM paged GPU memory to system RAM (+23 GB, the page file peaked at ~25 GB): the whole PC froze for
# seconds and the disk ran at 100%. 0.86 leaves ~1 GiB for the desktop; with HQ_ROCM_KV=octave a 150k window still
# uses a small part of the pool. Same value as the RTX 3090 single-user setup.
export GPU_UTIL=${GPU_UTIL:-0.86}
export KV_MEM=${KV_MEM-}          # empty: size the pool from GPU_UTIL (the 3090 byte pins do not apply here)
export PORT=${PORT:-19020}        # 18020 is inside a range Windows reserves for Hyper-V on many machines

# fp8 KV is slow on gfx11 (no native FP8) and a checkpoint with a kv_cache_scheme would switch to it silently,
# so the cache dtype is named, not left to "auto". The target and the drafter use the same backend.
# EXTRA_ARGS is applied last, so these replace the launcher's FLASH_ATTN defaults for CTX=fast.
if [ "${HQ_ROCM_KV:-bf16}" = bf16 ]; then
  export EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype bfloat16 ${EXTRA_ARGS:-}"
else
  export EXTRA_ARGS
fi

exec bash "$REPO/single-user/start_qwen.sh" "$@"
