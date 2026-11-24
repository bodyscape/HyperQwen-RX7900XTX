import torch, sys, time
import vllm._custom_ops as ops
from vllm.utils.platform_utils import num_compute_units

dev = 'cuda'
dt = torch.bfloat16
g = 128
cu = num_compute_units()
print('device', torch.cuda.get_device_name(0), 'CUs', cu)
has_wmma = hasattr(torch.ops._rocm_C, 'gptq_gemm_rdna3_wmma')
print('wmma op present:', has_wmma)

def mk(K, N, copies=6):
    out = []
    for _ in range(copies):
        qw = torch.randint(-2**31, 2**31 - 1, (K // 8, N), dtype=torch.int32, device=dev)          # exllama layout
        qz = torch.full((K // g, N // 8), 0x77777777 - 2**32 if 0x77777777 >= 2**31 else 0x77777777, dtype=torch.int32, device=dev)
        sc = (torch.rand(K // g, N, device=dev, dtype=dt) * 0.01)
        w8 = torch.randint(-128, 127, (N, K // 2), dtype=torch.int8, device=dev)                     # skinny layout
        sc2 = sc.t().contiguous()                                                                    # [N, K/g]
        out.append((qw, qz, sc, w8, sc2))
    return out

def timeit(fn, n=60):
    for _ in range(8):
        fn(0)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for i in range(n):
        fn(i)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000  # us

shapes = [('gate_up 5120->34816', 5120, 34816), ('down 17408->5120', 17408, 5120), ('qkv-ish 5120->8192', 5120, 8192)]
Ms = [8, 12, 16, 24, 32]
for name, K, N in shapes:
    W = mk(K, N)
    nbytes = K * N / 2 + 2 * K * N / g
    print('\n== %s  (weights %.1f MB)' % (name, nbytes / 1e6))
    print('  M | rdna3 us (GB/s) | wmma us (GB/s) | skinny us (GB/s)')
    for M in Ms:
        x = torch.randn(M, K, device=dev, dtype=dt)
        res = []
        for label, fn in (
            ('rdna3', lambda i: ops.gptq_gemm_rdna3(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False)),
            ('wmma', (lambda i: torch.ops._rocm_C.gptq_gemm_rdna3_wmma(x, W[i % 6][0], W[i % 6][1], W[i % 6][2], False)) if has_wmma else None),
            ('skinny', (lambda i: ops.wvSplitK_int4_g(W[i % 6][3], x, W[i % 6][4], cu, g, None, None)) if M <= 5 or True else None),
        ):
            if fn is None:
                res.append('   n/a      '); continue
            try:
                t = timeit(fn)
                res.append('%7.1f (%4.0f)' % (t, nbytes / t / 1e3))
            except Exception as ex:
                res.append('ERR %s' % str(ex)[:24])
        print('  %2d | %s' % (M, ' | '.join(res)))
    del W
    torch.cuda.empty_cache()
