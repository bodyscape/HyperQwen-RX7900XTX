import os, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
ext = load(name='w4a16_gemv', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/build', extra_cuda_cflags=['-O3'])
dev = 'cuda'; dt = torch.bfloat16; g = 128
def mk(K, N, copies=6):
    return [(torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev),
             torch.full((K // g, N // 8), 0x77777777, dtype=torch.int32, device=dev),
             (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)) for _ in range(copies)]
def timeit(fn, n=60):
    for _ in range(8): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000
shapes = [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv-ish', 5120, 8192), ('o', 6144, 5120)]
print('us per call by K-split (0 = default heuristic), M=4 and M=5')
for name, K, N in shapes:
    W = mk(K, N)
    for M in (4, 5):
        x = torch.randn(M, K, device=dev, dtype=dt)
        row = []
        for ks in (0, 1, 2, 3, 4, 6, 8, 12, 17):
            if ks > K // 128 // 8 and ks != 0: row.append('  --'); continue
            row.append('%5.1f' % timeit(lambda i: ext.w4a16_gemv(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False, ks)))
        print('%-8s M=%d  ks: 0   1   2   3   4   6   8  12  17 | ' % (name, M) + ' '.join(row))
    del W; torch.cuda.empty_cache()
