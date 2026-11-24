import torch, time
dev = 'cuda'
x = torch.randn(1 << 18, device=dev)          # 1 MB: each kernel does ~2 us of real work
y = torch.empty_like(x)

def chain(n):
    for _ in range(n):
        torch.mul(x, 1.0001, out=y)           # one tiny kernel

def ev_time(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps * 1000     # us per call

print('device', torch.cuda.get_device_name(0))
# one big kernel doing the same bytes as 1 tiny kernel x N, to get the pure-work time
big = torch.randn(1 << 18, device=dev)
t1 = ev_time(lambda: torch.mul(x, 1.0001, out=y), reps=200)
print('single tiny kernel, eager, back-to-back: %.1f us each' % t1)

for n in (100, 500, 1000, 2000):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        chain(n)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        chain(n)
    t_graph = ev_time(lambda: g.replay())
    t_eager = ev_time(lambda: chain(n), reps=5)
    print('N=%4d kernels: graph replay %8.0f us (%.2f us/kernel) | eager %8.0f us (%.2f us/kernel)' % (n, t_graph, t_graph / n, t_eager, t_eager / n))

# the same total work in fewer, bigger kernels (what fusion buys): N tiny ops -> N/10 ops over 10x the data
x10 = torch.randn(10 * (1 << 18), device=dev); y10 = torch.empty_like(x10)
def chain10(n):
    for _ in range(n // 10):
        torch.mul(x10, 1.0001, out=y10)
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    chain10(1000)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=s):
    chain10(1000)
print('same work as 1000 tiny kernels, in 100 kernels of 10x size: graph replay %.0f us' % ev_time(lambda: g.replay()))
