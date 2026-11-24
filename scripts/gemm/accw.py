import os, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
import vllm._custom_ops as ops
os.makedirs('/work/gemm/buildw', exist_ok=True)
load(name='gemvw', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/buildw', extra_cuda_cflags=['-O3'], is_python_module=False)
dev = 'cuda'; dt = torch.bfloat16; g = 128
def dq(qw, qz, sc, K, N):
    r = torch.arange(K // 8, device=dev)
    w = torch.empty(K, N, dtype=torch.float32, device=dev)
    for i in range(4):
        w[8 * r + 2 * i] = ((qw[r] >> (4 * i)) & 0xF).float()
        w[8 * r + 2 * i + 1] = ((qw[r] >> (4 * i + 16)) & 0xF).float()
    n = torch.arange(N, device=dev)
    z = (((qz[:, n // 8] >> ((n % 8) * 4)) & 0xF) + 1).float()
    gi = torch.arange(K, device=dev) // g
    return (w - z[gi]) * sc.float()[gi]
def timeit(fn, n=80):
    for _ in range(8): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000
for name, K, N in [('gate_up', 5120, 34816), ('down', 17408, 5120), ('qkv', 5120, 8192)]:
    Ws = []
    for _ in range(6):
        qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
        qz = torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev)
        sc = (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)
        Ws.append((qw, qz, sc))
    Wf = dq(*Ws[0], K, N)
    print('==', name)
    for M in (1, 2, 4, 8, 12, 16, 32):
        x = torch.randn(M, K, device=dev, dtype=dt)
        ref = x.float() @ Wf
        sc_ = ref.abs().mean()
        e = lambda y: ((y.float() - ref).abs().max() / sc_).item()
        st = ops.gptq_gemm_rdna3(x, *Ws[0], False)
        hw = torch.ops.hq.w4a16_wmma(x, *Ws[0], False, 0)
        res = f'  M={M:2d} err vs fp32: stock {e(st):.4f} wmma {e(hw):.4f}'
        if M <= 8:
            hg = torch.ops.hq.w4a16_gemv(x, *Ws[0], False, 0); res += f' gemv {e(hg):.4f}'
        t_w = timeit(lambda i: torch.ops.hq.w4a16_wmma(x, *Ws[i % 6], False, 0))
        res += f' | us: wmma {t_w:6.1f}'
        if M <= 8:
            t_g = timeit(lambda i: torch.ops.hq.w4a16_gemv(x, *Ws[i % 6], False, 0)); res += f' gemv {t_g:6.1f}'
        print(res)
