// Registers the copied Octave decode (vLLM v0.33.0 csrc/attention/octave_attn.cu + octave_decode_wmma_rdna3.cu,
// compiled with every external symbol renamed by -D flags, see build) as hq::octave_decode, so a modified copy
// can run next to vLLM's own _C::octave_decode. The CDNA (gfx942) MFMA paths are stubbed: this is gfx1100 only.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>

void octave_decode(torch::Tensor out, torch::Tensor query, torch::Tensor cache, torch::Tensor block_table,
                   torch::Tensor q_to_req, torch::Tensor q_to_klen, torch::Tensor mid_o, torch::Tensor k_signs,
                   torch::Tensor v_signs, double sm_scale, int64_t num_kv_splits, int64_t fmt, int64_t query_group,
                   bool use_wmma);

int octave_decode_mfma(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, double, int64_t, int64_t, int64_t) {
  return 0;
}

bool octave_reduce_mfma(torch::Tensor, torch::Tensor, torch::Tensor, int, hipStream_t) { return false; }

TORCH_LIBRARY_FRAGMENT(hq, m) {
  m.def(
      "octave_decode(Tensor(a!) out, Tensor query, Tensor cache, Tensor block_table, Tensor q_to_req, "
      "Tensor q_to_klen, Tensor(b!) mid_o, Tensor k_signs, Tensor v_signs, float sm_scale, int num_kv_splits, "
      "int fmt, int query_group, bool use_wmma) -> ()");
}
TORCH_LIBRARY_IMPL(hq, CUDA, m) { m.impl("octave_decode", &octave_decode); }
