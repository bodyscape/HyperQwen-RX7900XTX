import torch, inspect
import torch.nn.functional as F
import vllm._custom_ops as ops
from vllm.utils.platform_utils import num_compute_units
dev = 'cuda'; dt = torch.bfloat16
K, N = 5120, 96
cu = num_compute_units()
Ws = [torch.randn(N, K, device=dev, dtype=dt) for _ in range(8)]
print('wvSplitK sig:', [n for n in dir(ops) if 'wvSplitK' in n or 'wvsplitk' in n.lower()])
def timeit(fn, n=200):
    for _ in range(10): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000
print('us per call: F.linear | wvSplitK | max abs diff vs F.linear')
for M in (1, 2, 3, 4, 5, 6, 8):
    x = torch.randn(M, K, device=dev, dtype=dt)
    a = timeit(lambda i: F.linear(x, Ws[i % 8]))
    row = '  M=%d  linear %5.1f' % (M, a)
    for name in ('wvSplitK', 'wvSplitKQ'):
        f = getattr(ops, name, None)
        if f is None: continue
        try:
            out = f(Ws[0], x, cu)
            err = (out.float() - F.linear(x, Ws[0]).float()).abs().max().item()
            b = timeit(lambda i: f(Ws[i % 8], x, cu))
            row += ' | %s %5.1f (err %.3f)' % (name, b, err)
        except Exception as ex:
            row += ' | %s ERR %s' % (name, str(ex)[:40])
    print(row)
