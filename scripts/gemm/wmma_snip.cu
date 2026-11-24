// ---------------------------------------------------------------------------------------------------------------
// WMMA variant for 9..32 activation rows (3-8 concurrent speculative requests: 4 rows each).
//
// Why: above 8 rows the dot2 GEMV is compute-bound and the stock kernels fall to ~340 GB/s (M = 12..16) and
// ~170-200 GB/s (M = 17..32, weights read twice). Here the weights are read once, through the matrix units:
// B fragments are built straight from the int4 words ((nibble | 0x4300) is the bf16 number 128 + nibble, exact), a
// second WMMA against an all-ones B gives each row's sum of activations, and the zero point and the scale are applied
// per 128-wide group in fp32, exactly as in the GEMV: out = s * (sum a * (128 + w) - (128 + z) * sum a).
//
// Mapping: a wave owns 64 output columns, 4 per lane-column of the 16x16 tile: WMMA j uses column 4 * c + j for tile
// column c, so one uint4 load per weight row gives a lane the words of its 4 columns (256 B contiguous per row for
// the 16 lanes of a half wave; lanes c and c + 16 load the same words, as the A/B operands are replicated in both half
// waves). Waves of a block split K by whole groups; the partial tiles are reduced through LDS, and gridDim.y splits K
// further for narrow outputs (second kernel adds the partials in a fixed order).
typedef short __attribute__((ext_vector_type(16))) bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) f32x8_t;
typedef unsigned int __attribute__((ext_vector_type(8))) u32x8_t;

#ifndef HQ_W2_WAVES
#define HQ_W2_WAVES 8
#endif
#ifndef HQ_W2_UN
#define HQ_W2_UN 4          // 16-wide K steps in flight per group pass (8 per group)
#endif
constexpr int W2 = HQ_W2_WAVES;

__device__ __forceinline__ f32x8_t wmma_bf16(const u32x8_t& a, const u32x8_t& b, const f32x8_t& c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(bf16x16_t, a), __builtin_bit_cast(bf16x16_t, b),
                                                      c);
}

template <int T>
__global__ void __launch_bounds__(W2 * 32)
wmma_kernel(const uint16_t* __restrict__ A, const uint32_t* __restrict__ Wq, const uint32_t* __restrict__ Zq,
            const uint16_t* __restrict__ S, uint16_t* __restrict__ out, float* __restrict__ partial, int M, int K,
            int N, int zero_off, int ksplit) {
  __shared__ float red[W2][16][64];
  const int lane = threadIdx.x & 31;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int lm = lane & 15, half = lane >> 4;
  const int n0 = (blockIdx.x * 16 + lm) * 4;
  const int groups = K / GS;
  const int slice = blockIdx.y * W2 + wave;
  const int total = ksplit * W2;
  const int g0 = (int)((long long)groups * slice / total);
  const int g1 = (int)((long long)groups * (slice + 1) / total);

  f32x8_t acc[T][4];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[t][j][i] = 0.f;

  u32x8_t ones;
#pragma unroll
  for (int i = 0; i < 8; ++i) ones[i] = ONES;
  const uint32_t* arow[T];
  bool rvalid[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    rvalid[t] = (t * 16 + lm) < M;
    arow[t] = (const uint32_t*)(A + (size_t)(rvalid[t] ? (t * 16 + lm) : 0) * K);
  }

  if (n0 < N) {
    for (int g = g0; g < g1; ++g) {
      const uint2 s2 = *(const uint2*)(S + (size_t)g * N + n0);
      const float sc[4] = {bf2f((uint16_t)(s2.x & 0xFFFF)), bf2f((uint16_t)(s2.x >> 16)),
                           bf2f((uint16_t)(s2.y & 0xFFFF)), bf2f((uint16_t)(s2.y >> 16))};
      const uint32_t zw = Zq[(size_t)g * (N / 8) + (n0 >> 3)] >> ((n0 & 7) * 4);
      float zf[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) zf[c] = (float)(128 + ((zw >> (4 * c)) & 0xF) + zero_off);

      f32x8_t d[T][4], ds[T];
#pragma unroll
      for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) ds[t][i] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int i = 0; i < 8; ++i) d[t][j][i] = 0.f;
      }
      const uint32_t* wp = Wq + (size_t)(g * ROWS_PER_GROUP) * N + n0;
#pragma unroll
      for (int kb = 0; kb < 8; kb += HQ_W2_UN) {
        u32x4 w0[HQ_W2_UN], w1[HQ_W2_UN];
#pragma unroll
        for (int u = 0; u < HQ_W2_UN; ++u) {
          w0[u] = load_w(wp + (size_t)(2 * (kb + u)) * N);
          w1[u] = load_w(wp + (size_t)(2 * (kb + u) + 1) * N);
        }
#pragma unroll
        for (int u = 0; u < HQ_W2_UN; ++u) {
          const int k0 = (g * 8 + kb + u) * 16;
          u32x8_t av[T];
#pragma unroll
          for (int t = 0; t < T; ++t) {
            const uint4 lo = *(const uint4*)(arow[t] + (k0 >> 1));
            const uint4 hi = *(const uint4*)(arow[t] + (k0 >> 1) + 4);
            const bool v = rvalid[t];
            av[t] = u32x8_t{v ? lo.x : 0u, v ? lo.y : 0u, v ? lo.z : 0u, v ? lo.w : 0u,
                            v ? hi.x : 0u, v ? hi.y : 0u, v ? hi.z : 0u, v ? hi.w : 0u};
          }
          const uint32_t wa[4] = {w0[u].x, w0[u].y, w0[u].z, w0[u].w};
          const uint32_t wb[4] = {w1[u].x, w1[u].y, w1[u].z, w1[u].w};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            u32x8_t b;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              b[i] = ((wa[j] >> (4 * i)) & 0x000F000Fu) | MAGIC;
              b[4 + i] = ((wb[j] >> (4 * i)) & 0x000F000Fu) | MAGIC;
            }
#pragma unroll
            for (int t = 0; t < T; ++t) d[t][j] = wmma_bf16(av[t], b, d[t][j]);
          }
#pragma unroll
          for (int t = 0; t < T; ++t) ds[t] = wmma_bf16(av[t], ones, ds[t]);
        }
      }
#pragma unroll
      for (int t = 0; t < T; ++t)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[t][j][i] = __fmaf_rn(sc[j], __fmaf_rn(-zf[j], ds[t][i], d[t][j][i]), acc[t][j][i]);
    }
  }

#pragma unroll
  for (int t = 0; t < T; ++t) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i) red[wave][2 * i + half][lm * 4 + j] = acc[t][j][i];
    __syncthreads();
    for (int idx = threadIdx.x; idx < 16 * 64; idx += W2 * 32) {
      const int r = idx >> 6, cl = idx & 63;
      const int m = t * 16 + r;
      const int n = blockIdx.x * 64 + cl;
      if (m < M && n < N) {
        float s = 0.f;
#pragma unroll
        for (int w = 0; w < W2; ++w) s += red[w][r][cl];
        if (ksplit == 1)
          out[(size_t)m * N + n] = f2bf(s);
        else
          partial[((size_t)blockIdx.y * M + m) * N + n] = s;
      }
    }
    __syncthreads();
  }
}

template <int T>
static void launch_wmma(const torch::Tensor& a, const torch::Tensor& qw, const torch::Tensor& qz,
                        const torch::Tensor& sc, torch::Tensor& out, int M, int K, int N, int zero_off, int ksplit,
                        hipStream_t stream) {
  const int tiles = (N + 63) / 64;
  torch::Tensor partial;
  float* pp = nullptr;
  if (ksplit > 1) {
    partial = torch::empty({ksplit, M, N}, a.options().dtype(torch::kFloat32));
    pp = partial.data_ptr<float>();
  }
  dim3 grid(tiles, ksplit);
  wmma_kernel<T><<<grid, W2 * 32, 0, stream>>>((const uint16_t*)a.data_ptr(), (const uint32_t*)qw.data_ptr(),
                                                (const uint32_t*)qz.data_ptr(), (const uint16_t*)sc.data_ptr(),
                                                (uint16_t*)out.data_ptr(), pp, M, K, N, zero_off, ksplit);
  if (ksplit > 1) {
    const int n = M * N;
    reduce_kernel<<<(n + 255) / 256, 256, 0, stream>>>(pp, (uint16_t*)out.data_ptr(), M, N, ksplit);
  }
}

torch::Tensor w4a16_wmma(torch::Tensor a, torch::Tensor qweight, torch::Tensor qzeros, torch::Tensor scales,
                         bool use_v2_format, int64_t ksplit_override) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous(), "a must be contiguous bf16");
  TORCH_CHECK(scales.scalar_type() == at::kBFloat16 && scales.is_contiguous(), "scales must be bf16");
  TORCH_CHECK(qweight.scalar_type() == at::kInt && qzeros.scalar_type() == at::kInt, "qweight/qzeros must be int32");
  const int M = a.size(0), K = a.size(1), N = qweight.size(1);
  TORCH_CHECK(M >= 1 && M <= 32, "M must be 1..32");
  TORCH_CHECK(qweight.size(0) * 8 == K, "K mismatch");
  TORCH_CHECK(K % GS == 0 && scales.size(0) * GS == K, "group size must be 128");
  TORCH_CHECK(N % 8 == 0, "N must be a multiple of 8");
  const at::cuda::OptionalCUDAGuard guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto out = torch::empty({M, N}, a.options());
  const int groups = K / GS;
  const int tiles = (N + 63) / 64;
  int ksplit = ksplit_override > 0 ? (int)ksplit_override : (HQ_KSPLIT_TARGET + tiles - 1) / tiles;
  ksplit = std::max(1, std::min(ksplit, std::max(1, groups / W2)));
  const int zero_off = use_v2_format ? 0 : 1;
  if (M <= 16)
    launch_wmma<1>(a, qweight, qzeros, scales, out, M, K, N, zero_off, ksplit, stream);
  else
    launch_wmma<2>(a, qweight, qzeros, scales, out, M, K, N, zero_off, ksplit, stream);
  return out;
}

