import os, sys, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
os.makedirs('/work/gemm/buildw', exist_ok=True)
load(name='gemvw', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/buildw', extra_cuda_cflags=['-O3'], is_python_module=False)
dev = 'cuda'; dt = torch.bfloat16; g = 128
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
op = torch.ops.hq.w4a16_gemv if M <= 4 else torch.ops.hq.w4a16_wmma
def graph_us(fn, reps=24, replays=10):
    for _ in range(3): fn(0)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for i in range(reps): fn(i)
    gr.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(replays): gr.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / replays / reps * 1000
shapes = [('gdn qkvz', 5120, 16384), ('gdn out', 6144, 5120), ('attn qkv', 5120, 14336), ('attn o', 6144, 5120),
          ('gate_up', 5120, 34816), ('down', 17408, 5120), ('lm_head', 5120, 248320)]
print(f'M={M}  columns: ksplit -> us per call (GB/s); 0 = automatic')
for name, K, N in shapes:
    Ws = []
    nb = K * N // 2 + 2 * K * N // g
    copies = max(2, min(6, int(3.0e9 // nb)))
    for _ in range(copies):
        Ws.append((torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev),
                   torch.randint(-2**31, 2**31 - 1, (K // g, N // 8), dtype=torch.int32, device=dev),
                   (torch.rand(K // g, N, device=dev) * 0.02 + 0.002).to(dt)))
    x = torch.randn(M, K, device=dev, dtype=dt)
    row = f'{name:9s} {K:6d}x{N:6d} {nb/1e6:6.1f}MB |'
    for ks in (0, 1, 2, 4, 8):
        try:
            t = graph_us(lambda i: op(x, *Ws[i % copies], False, ks))
            row += f' k{ks}:{t:6.1f}({nb/t/1e3:4.0f})'
        except Exception as ex:
            row += f' k{ks}:ERR'
    print(row)
    del Ws; torch.cuda.empty_cache()
