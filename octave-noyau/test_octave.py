"""hq::octave_decode (copied/modified kernel) against vLLM's _C::octave_decode: same output, and time per layer.

usage: test_octave.py <lib.so>
"""
import sys, torch
import vllm._C  # noqa
from vllm.v1.attention.ops import rocm_octave as sq

torch.ops.load_library(sys.argv[1])
dev = 'cuda'
Hq, Hkv, D, ROPE, BS = 24, 4, 256, 64, 2176
scale = D ** -0.5
fmt = sq.OctaveFormat.from_cache_dtype('octave_k3v3_compact', D, ROPE)
sq.register_rope_dim(D, ROPE)
ks = sq.sign_bits(D).to(dev)
torch.manual_seed(0)


def make(kv_len):
    nblk = (kv_len + BS - 1) // BS
    k = torch.randn(nblk * BS, Hkv, D, device=dev) * 0.5
    v = torch.randn(nblk * BS, Hkv, D, device=dev) * 0.5
    slots = sq.reference_quantize(k, v, fmt)            # a real Octave cache, not random bytes
    cache = slots.view(nblk, BS, Hkv, -1).permute(0, 2, 1, 3).contiguous()
    return cache, torch.arange(nblk, dtype=torch.int32, device=dev).view(1, -1)


def run(op, cache, bt, kv_len, nq, splits=192, iters=30):
    q = (torch.randn(nq, Hq, D, device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    q2r = torch.zeros(nq, dtype=torch.int32, device=dev)
    q2k = (kv_len - nq + 1 + torch.arange(nq, device=dev)).to(torch.int32)
    mid = torch.empty(nq, Hq, splits, D + 2, dtype=torch.float32, device=dev)
    f = lambda: op(out, q, cache, bt, q2r, q2k, mid, ks, ks, scale, splits, fmt.kernel_code, 4 if nq > 1 else 1, True)
    f(); torch.cuda.synchronize()
    res = out.float().clone()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): f()
    e.record(); torch.cuda.synchronize()
    return res, s.elapsed_time(e) / iters * 1000, q, q2r, q2k


for kv_len in (8000, 32000, 125000):
    cache, bt = make(kv_len)
    for nq in (1, 2, 4):
        torch.manual_seed(nq)
        a, ta, q, q2r, q2k = run(torch.ops._C.octave_decode, cache, bt, kv_len, nq)
        torch.manual_seed(nq)
        b, tb, *_ = run(torch.ops.hq.octave_decode, cache, bt, kv_len, nq)
        ref = sq.reference_attention(q, cache, bt, q2r, q2k, fmt, scale) if kv_len <= 8000 else None
        err_ab = (a - b).abs().max().item()
        err_ref = ((b - ref).abs().max() / ref.abs().mean()).item() if ref is not None else float('nan')
        print('kv %6d  q=%d | vLLM %7.1f us | hq %7.1f us | x%.2f | max|vLLM-hq| %.2e | hq vs torch reference %.4f'
              % (kv_len, nq, ta, tb, ta / tb, err_ab, err_ref))
    del cache
    torch.cuda.empty_cache()
