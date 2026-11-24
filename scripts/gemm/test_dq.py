"""Dequant + hipBLASLt vs gptq_gemm_rdna3 at prefill sizes: correctness against an fp32 reference, then time."""
import os, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
import vllm._custom_ops as ops

load(name='w4a16_gemv', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/build_dq',
     extra_cuda_cflags=['-O3'], is_python_module=False)
dev = 'cuda'; dt = torch.bfloat16; g = 128


def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


def dequant_ref(qw, qz, sc, K, N):
    r = torch.arange(K // 8, device=dev)
    w = torch.empty(K, N, dtype=torch.float32, device=dev)
    for i in range(4):
        w[8 * r + 2 * i] = ((qw[r] >> (4 * i)) & 0xF).float()
        w[8 * r + 2 * i + 1] = ((qw[r] >> (4 * i + 16)) & 0xF).float()
    n = torch.arange(N, device=dev)
    z = (((qz[:, n // 8] >> ((n % 8) * 4)) & 0xF) + 1).float()
    gi = torch.arange(K, device=dev) // g
    return (w - z[gi]) * sc.float()[gi]


shapes = [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv-ish', 5120, 8192), ('o', 6144, 5120)]
print('-- correctness, M=64, max abs err / mean|ref| against fp32: rdna3 | dequant+mm')
for name, K, N in shapes:
    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    qz = torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev)
    sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
    x = torch.randn(64, K, device=dev, dtype=dt)
    ref = x.float() @ dequant_ref(qw, qz, sc, K, N)
    e1 = ((ops.gptq_gemm_rdna3(x, qw, qz, sc, False).float() - ref).abs().max() / ref.abs().mean()).item()
    e2 = ((x @ torch.ops.hq.w4a16_dequant(qw, qz, sc, False)).float() - ref).abs().max() / ref.abs().mean()
    print('  %-8s  %.4f | %.4f' % (name, e1, e2.item()))

print('\n-- time, us per call: rdna3 | dequant | mm | dequant+mm | speedup of dequant+mm')
tot = {}
for name, K, N in shapes:
    qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
    qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int32, device=dev)
    sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
    wb = torch.ops.hq.w4a16_dequant(qw, qz, sc, False)
    tdq = timeit(lambda: torch.ops.hq.w4a16_dequant(qw, qz, sc, False))
    for M in (512, 1024, 2048):
        x = torch.randn(M, K, device=dev, dtype=dt)
        a = timeit(lambda: ops.gptq_gemm_rdna3(x, qw, qz, sc, False))
        b = timeit(lambda: x @ wb)
        c = timeit(lambda: x @ torch.ops.hq.w4a16_dequant(qw, qz, sc, False))
        tot.setdefault(M, [0, 0])
        tot[M][0] += a; tot[M][1] += c
        print('  %-8s M=%4d  %8.0f | %6.0f | %8.0f | %8.0f | x%.2f' % (name, M, a, tdq, b, c, a / c))
    del qw, qz, sc, wb
    torch.cuda.empty_cache()
for M, (a, c) in tot.items():
    print('  four shapes together, M=%4d: rdna3 %.0f us, dequant+mm %.0f us, x%.2f' % (M, a, c, a / c))
