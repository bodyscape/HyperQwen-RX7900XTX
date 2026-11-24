import sys, math, torch
import vllm._C  # noqa
from vllm.v1.attention.ops import rocm_octave as sq

dev = 'cuda'
Hq, Hkv, D, ROPE = 24, 4, 256, 64
scale = D ** -0.5
fmt = sq.OctaveFormat.from_cache_dtype('octave_k3v3_compact', D, ROPE)
slot = fmt.slot_bytes
sq.register_rope_dim(D, ROPE)
k_signs = sq.sign_bits(D).to(dev)
v_signs = k_signs
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 2176
print('slot bytes', slot, 'block size', BS, 'kernel_code', fmt.kernel_code)

def run(kv_len, num_q, splits, qgroup, iters=40):
    nblk = (kv_len + BS - 1) // BS
    cache = torch.randint(0, 255, (nblk, Hkv, BS, slot), dtype=torch.uint8, device=dev)
    # plausible fp16 scales so softmax does not blow up: overwrite scale bytes
    sc = (torch.rand(nblk, Hkv, BS, fmt.num_scales, device=dev) * 0.5 + 0.1).to(torch.float16).view(torch.uint8).view(nblk, Hkv, BS, -1)
    cache[..., fmt.off_scales:fmt.off_scales + sc.shape[-1]] = sc
    bt = torch.arange(nblk, dtype=torch.int32, device=dev).view(1, -1)
    q = torch.randn(num_q, Hq, D, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    q_to_req = torch.zeros(num_q, dtype=torch.int32, device=dev)
    q_to_klen = (kv_len - num_q + 1 + torch.arange(num_q, device=dev)).to(torch.int32)
    mid = torch.empty(num_q, Hq, splits, D + 2, dtype=torch.float32, device=dev)
    def f():
        torch.ops._C.octave_decode(out, q, cache, bt, q_to_req, q_to_klen, mid, k_signs, v_signs,
                                   scale, splits, fmt.kernel_code, qgroup, True)
    for _ in range(5): f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    t = s.elapsed_time(e) / iters * 1000
    del cache
    return t, out.float().abs().mean().item()

nb = lambda L: L * Hkv * slot
print('\nper-layer decode time (us); roofline at 800 GB/s in brackets')
print('%8s %5s %6s %6s | %9s %9s' % ('kv_len', 'num_q', 'qgrp', 'splits', 'time us', 'GB/s'))
for L in (8000, 32000, 70000, 125000):
    for nq, qg in ((1, 1), (4, 4)):
        for sp in (8, 16, 32, 64, 128):
            try:
                t, _ = run(L, nq, sp, qg)
                print('%8d %5d %6d %6d | %9.1f %9.0f   [%.0f us]' % (L, nq, qg, sp, t, nb(L) / t / 1e3, nb(L) / 800e3))
            except Exception as ex:
                print('%8d %5d %6d %6d | ERR %s' % (L, nq, qg, sp, str(ex)[:60]))
    print()
