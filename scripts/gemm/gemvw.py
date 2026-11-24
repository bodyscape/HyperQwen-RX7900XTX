import os, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
import vllm._custom_ops as ops
os.makedirs('/work/gemm/buildw', exist_ok=True)
ext = load(name='gemvw', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/buildw', extra_cuda_cflags=['-O3'], is_python_module=False)
dev = 'cuda'; dt = torch.bfloat16; g = 128
def mk(K, N, copies=6):
    o = []
    for _ in range(copies):
        qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
        qz = torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev)
        sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
        o.append((qw, qz, sc))
    return o
def timeit(fn, n=80):
    for _ in range(8): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000
shapes = [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv', 5120, 8192), ('o', 6144, 5120)]
bad = 0
for name, K, N in shapes:
    W = mk(K, N)
    print(f'== {name} K={K} N={N}')
    for M in (8, 12, 16, 24, 32):
        x = torch.randn(M, K, device=dev, dtype=dt)
        ref = ops.gptq_gemm_rdna3(x, W[0][0], W[0][1], W[0][2], False).float()
        got = torch.ops.hq.w4a16_wmma(x, W[0][0], W[0][1], W[0][2], False, 0).float()
        err = ((got - ref).abs().max() / ref.abs().mean()).item()
        if err > 0.02 or not torch.isfinite(got).all(): bad += 1
        t_st = timeit(lambda i: ops.gptq_gemm_rdna3(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False))
        t_hq = timeit(lambda i: torch.ops.hq.w4a16_wmma(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False, 0))
        print(f'  M={M:2d} stock {t_st:7.1f} us  hq {t_hq:7.1f} us  x{t_st / t_hq:4.2f}  err={err:.4f}')
    del W; torch.cuda.empty_cache()
print('BAD', bad)
