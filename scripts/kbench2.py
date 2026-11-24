import torch
import vllm._custom_ops as ops
from vllm.utils.platform_utils import num_compute_units
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as hy

dev = 'cuda'; dt = torch.bfloat16; g = 128; cu = num_compute_units()
print('triton fn:', [n for n in dir(hy) if 'triton' in n.lower() and 'gemm' in n.lower()][:5], 'pad fn:', hasattr(hy, '_pad_activation_rows'))

def mk(K, N, copies=6):
    out = []
    for _ in range(copies):
        qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)
        qz = torch.full((K // g, N // 8), 0x77777777, dtype=torch.int32, device=dev)
        sc = torch.rand(K // g, N, device=dev, dtype=dt) * 0.01
        w8 = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=dev)
        out.append((qw, qz, sc, w8, sc.t().contiguous()))
    return out

def timeit(fn, n=50):
    for _ in range(6): fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n): fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000

shapes = [('gate_up 5120->34816', 5120, 34816), ('down 17408->5120', 17408, 5120), ('qkv-ish 5120->8192', 5120, 8192), ('o 6144->5120', 6144, 5120)]
print('us per call: rdna3 | skinny | triton(skinny fmt) | best routed')
tot = {M: [0.0, 0.0] for M in (1, 2, 3, 4, 5, 6, 8)}
for name, K, N in shapes:
    W = mk(K, N)
    print('\n==', name)
    for M in (1, 2, 3, 4, 5, 6, 8):
        x = torch.randn(M, K, device=dev, dtype=dt)
        r = {}
        for label, fn in (
            ('rdna3', lambda i: ops.gptq_gemm_rdna3(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False)),
            ('skinny', lambda i: ops.wvSplitK_int4_g(W[i % 6][3], x, W[i % 6][4], cu, g, None, None)),
            ('triton', lambda i: hy.triton_w4a16_skinny_fmt_gemm(a=hy._pad_activation_rows(x), b_q=W[i % 6][3].view(torch.int32), scales=W[i % 6][4], group_size=g, zp=None)),
        ):
            try:
                r[label] = timeit(fn)
            except Exception as ex:
                r[label] = None
        best = min(v for v in r.values() if v)
        print('  M=%d  rdna3 %6s | skinny %6s | triton %6s | best %6.1f' % (M, *(('%.1f' % r[k]) if r[k] else 'n/a' for k in ('rdna3', 'skinny', 'triton')), best))
    del W; torch.cuda.empty_cache()
