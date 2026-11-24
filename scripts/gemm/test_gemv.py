import os, sys, time, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
import vllm._custom_ops as ops

t0 = time.time()
ext = load(name='w4a16_gemv', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/build',
           extra_cuda_cflags=['-O3'], verbose=False)
print('built/loaded in %.0fs' % (time.time() - t0))

dev = 'cuda'; dt = torch.bfloat16; g = 128
def mk(K, N, copies=6, rand_zeros=False):
    out = []
    for _ in range(copies):
        qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
        if rand_zeros:
            qz = torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev)
        else:
            qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int32, device=dev)
        sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
        out.append((qw, qz, sc))
    return out

def timeit(fn, n=60):
    for _ in range(8): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000

shapes = [('gate_up 5120->34816', 5120, 34816), ('down 17408->5120', 17408, 5120), ('qkv-ish 5120->8192', 5120, 8192), ('o 6144->5120', 6144, 5120)]
mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
bad = 0
print('\n-- correctness vs gptq_gemm_rdna3 (max rel err of |diff| / mean|ref|), random zeros too')
for name, K, N in shapes:
    for rz in (False, True):
        W = mk(K, N, copies=1, rand_zeros=rz)[0]
        for M in range(1, 9):
            x = torch.randn(M, K, device=dev, dtype=dt)
            ref = ops.gptq_gemm_rdna3(x, W[0], W[1], W[2], False).float()
            got = ext.w4a16_gemv(x, W[0], W[1], W[2], False, 0).float()
            err = ((got - ref).abs().max() / ref.abs().mean()).item()
            if err > 0.02 or not torch.isfinite(got).all():
                bad += 1
                print('  BAD %s zeros=%s M=%d err=%.4f' % (name, 'rand' if rz else 'sym', M, err))
        print('  %-22s zeros=%-4s worst-case checked M=1..8' % (name, 'rand' if rz else 'sym'))
print('correctness failures vs rdna3 (loose 2% bound):', bad)

def dequant_ref(qw, qz, sc, K, N):
    r = torch.arange(K // 8, device=dev)
    w = torch.empty(K, N, dtype=torch.float32, device=dev)
    for i in range(4):
        w[8 * r + 2 * i] = ((qw[r] >> (4 * i)) & 0xF).float()
        w[8 * r + 2 * i + 1] = ((qw[r] >> (4 * i + 16)) & 0xF).float()
    n = torch.arange(N, device=dev)
    z = (((qz[:, n // 8] >> ((n % 8) * 4)) & 0xF) + 1).float()        # [K/g, N]
    gi = torch.arange(K, device=dev) // g
    return (w - z[gi]) * sc.float()[gi]

print('\n-- both kernels against an independent fp32 reference (max abs err / mean|ref|)')
for name, K, N in shapes:
    W = mk(K, N, copies=1, rand_zeros=True)[0]
    Wf = dequant_ref(*W, K, N)
    for M in (1, 4, 5, 8):
        x = torch.randn(M, K, device=dev, dtype=dt)
        ref = x.float() @ Wf
        e_old = ((ops.gptq_gemm_rdna3(x, W[0], W[1], W[2], False).float() - ref).abs().max() / ref.abs().mean()).item()
        e_new = ((ext.w4a16_gemv(x, W[0], W[1], W[2], False, 0).float() - ref).abs().max() / ref.abs().mean()).item()
        print('  %-22s M=%d  rdna3 %.4f | new %.4f' % (name, M, e_old, e_new))
if mode == 'check':
    sys.exit(0 if bad == 0 else 1)

print('\n-- speed, us per call (weights rotated to defeat the Infinity Cache). rdna3 | new | new GB/s')
for name, K, N in shapes:
    W = mk(K, N)
    nbytes = K * N / 2 + 2 * K * N / g
    print('==', name, '(%.1f MB)' % (nbytes / 1e6))
    for M in (1, 2, 3, 4, 5, 6, 7, 8):
        x = torch.randn(M, K, device=dev, dtype=dt)
        a = timeit(lambda i: ops.gptq_gemm_rdna3(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False))
        b = timeit(lambda i: ext.w4a16_gemv(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False, 0))
        print('  M=%d  rdna3 %6.1f | new %6.1f (%4.0f GB/s)  x%.2f' % (M, a, b, nbytes / b / 1e3, a / b))
    del W; torch.cuda.empty_cache()
