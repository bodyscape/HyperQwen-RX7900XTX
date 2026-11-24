"""Build one tuning variant of rocm/w4a16_gemv.cu and time it. usage: gemv_variant.py <tag> [-DFOO=1 ...]"""
import os, sys, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
import vllm._custom_ops as ops

tag, flags = sys.argv[1], sys.argv[2:]
bdir = f'/work/gemm/build_{tag}'
os.makedirs(bdir, exist_ok=True)
load(name=f'gemv_{tag}', sources=['/work/gemm/w4a16_gemv.cu'], build_directory=bdir,
     extra_cuda_cflags=['-O3'] + flags, is_python_module=False)
dev = 'cuda'; dt = torch.bfloat16; g = 128


def timeit(fn, n=80):
    for _ in range(8): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


shapes = [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv', 5120, 8192), ('o', 6144, 5120)]
Ms = (1, 4, 5, 8)
tot = {M: 0.0 for M in Ms}
per4 = []
bad = 0
for name, K, N in shapes:
    copies = 8
    W = [(torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev),
          torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev),
          (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)) for _ in range(copies)]
    for M in Ms:
        x = torch.randn(M, K, device=dev, dtype=dt)
        ref = ops.gptq_gemm_rdna3(x, *W[0], False).float()
        got = torch.ops.hq.w4a16_gemv(x, *W[0], False, 0).float()
        if ((got - ref).abs().max() / ref.abs().mean()).item() > 0.03:
            bad += 1
        t = timeit(lambda i: torch.ops.hq.w4a16_gemv(x, *W[i % copies], False, 0))
        tot[M] += t
        if M == 4:
            per4.append('%s %.1f' % (name, t))
    del W
    torch.cuda.empty_cache()
print('VARIANT %-10s %-40s | sum of 4 shapes, us: %s | M=4: %s | correctness failures: %d' % (
    tag, ' '.join(flags) or '(default)', '  '.join('M%d=%.1f' % (M, tot[M]) for M in Ms), ', '.join(per4), bad))
