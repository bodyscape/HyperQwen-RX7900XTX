// Peak WMMA throughput on gfx1100: bf16 16x16x16 against iu8 16x16x16, independent accumulators, no memory traffic.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

typedef short __attribute__((ext_vector_type(16))) bf16x16_t;
typedef float __attribute__((ext_vector_type(8))) f32x8_t;
typedef int __attribute__((ext_vector_type(8))) i32x8_t;
typedef int __attribute__((ext_vector_type(4))) i32x4_t;

#define NACC 8

__global__ void k_bf16(float* out, int iters, float seed) {
  bf16x16_t a, b;
  for (int i = 0; i < 16; ++i) { a[i] = (short)(0x3F80 + threadIdx.x + i); b[i] = (short)(0x3F00 + i); }
  f32x8_t acc[NACC];
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) acc[j][i] = seed;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < NACC; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[j]);
  }
  float s = 0;
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) s += acc[j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void k_iu8(float* out, int iters, float seed) {
  i32x4_t a, b;
  for (int i = 0; i < 4; ++i) { a[i] = 0x01020304 + threadIdx.x + i; b[i] = 0x01010101 * (i + 1); }
  i32x8_t acc[NACC];
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) acc[j][i] = (int)seed;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < NACC; ++j) acc[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[j], false);
  }
  int s = 0;
  for (int j = 0; j < NACC; ++j) for (int i = 0; i < 8; ++i) s += acc[j][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = (float)s;
}

int main() {
  float* d; hipMalloc(&d, 1 << 24);
  int blocks = 96 * 8, threads = 128, iters = 4000;
  hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
  for (int rep = 0; rep < 2; ++rep) {
    k_bf16<<<blocks, threads>>>(d, iters, 0.f); hipDeviceSynchronize();
    hipEventRecord(s); k_bf16<<<blocks, threads>>>(d, iters, 0.f); hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e);
    double flop = 2.0 * 16 * 16 * 16 * NACC * (double)iters * (blocks * threads / 32);
    printf("bf16 wmma: %.2f ms  %.1f TFLOPs\n", ms, flop / ms / 1e9);
    k_iu8<<<blocks, threads>>>(d, iters, 0.f); hipDeviceSynchronize();
    hipEventRecord(s); k_iu8<<<blocks, threads>>>(d, iters, 0.f); hipEventRecord(e); hipEventSynchronize(e);
    hipEventElapsedTime(&ms, s, e);
    printf("iu8  wmma: %.2f ms  %.1f TOPS\n", ms, flop / ms / 1e9);
  }
  return 0;
}
