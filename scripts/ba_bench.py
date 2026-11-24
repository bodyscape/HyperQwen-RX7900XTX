import torch
import torch.nn.functional as F
dev = 'cuda'; dt = torch.bfloat16
K, N = 5120, 96
Ws = [torch.randn(N, K, device=dev, dtype=dt) for _ in range(8)]
def timeit(fn, n=200):
    for _ in range(10): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000
print('F.linear [M,5120] x [96,5120]^T bf16 (the unquantized in_proj_ba of the 48 GDN layers), us per call')
print('  M   eager   in-graph(48 calls replayed)')
for M in (1, 2, 3, 4, 5, 6, 8):
    x = torch.randn(M, K, device=dev, dtype=dt)
    t = timeit(lambda i: F.linear(x, Ws[i % 8]))
    # 48 calls in one graph, like a decode step
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(48): F.linear(x, Ws[i % 8])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(48): F.linear(x, Ws[i % 8])
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20): g.replay()
    b.record(); torch.cuda.synchronize()
    print('  %d  %6.1f   %6.1f us/call (x48 = %.2f ms per step)' % (M, t, a.elapsed_time(b) / 20 / 48 * 1000, a.elapsed_time(b) / 20))
