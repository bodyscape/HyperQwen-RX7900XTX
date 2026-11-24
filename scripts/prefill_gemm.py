"""W4A16 GEMM at prefill sizes on gfx1100: which path is fastest for M = 256..2048?"""
import torch
import vllm._custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as hy

dev = 'cuda'; dt = torch.bfloat16; g = 128
print('preferred BLAS:', torch.backends.cuda.preferred_blas_library())


def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000  # us


def dequant(qw, qz, sc, K, N):
    """exllama-shuffled int4 [K/8, N] -> bf16 [K, N] (nibble pair i of a word = K rows 2i, 2i+1)."""
    w = torch.empty(K, N, dtype=dt, device=dev)
    zero = (((qz[:, torch.arange(N, device=dev) // 8] >> ((torch.arange(N, device=dev) % 8) * 4)) & 0xF) + 1).to(dt)
    for i in range(4):
        lo = ((qw >> (4 * i)) & 0xF).to(dt); hi = ((qw >> (4 * i + 16)) & 0xF).to(dt)
        w[2 * i::8] = lo; w[2 * i + 1::8] = hi
    gi = torch.arange(K, device=dev) // g
    return (w - zero[gi]) * sc[gi]


shapes = [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv-ish', 5120, 8192)]
print('TFLOPS (higher is better): rdna3 | rdna3_wmma | triton(hybrid) | bf16 hipBLAS GEMM (weights already bf16) | dequant alone (us)')
for name, K, N in shapes:
    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int32, device=dev)
    sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
    wb = dequant(qw, qz, sc, K, N)          # [K, N] bf16
    w8 = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=dev)
    sc2 = sc.t().contiguous()
    td = timeit(lambda: dequant(qw, qz, sc, K, N), n=5)
    for M in (256, 1024, 2048):
        x = torch.randn(M, K, device=dev, dtype=dt)
        fl = 2.0 * M * K * N
        res = []
        for fn in (lambda: ops.gptq_gemm_rdna3(x, qw, qz, sc, False),
                   lambda: torch.ops._rocm_C.gptq_gemm_rdna3_wmma(x, qw, qz, sc, False),
                   lambda: hy.triton_w4a16_skinny_fmt_gemm(a=hy._pad_activation_rows(x), b_q=w8.view(torch.int32), scales=sc2, group_size=g, zp=None),
                   lambda: x @ wb):
            try:
                t = timeit(fn)
                res.append('%6.1f' % (fl / t / 1e6))
            except Exception as ex:
                res.append('  ERR ')
        print('%-8s M=%4d  %s | %s | %s | %s | dequant %.0f us' % (name, M, *res, td))
    del qw, qz, sc, wb, w8, sc2
    torch.cuda.empty_cache()
