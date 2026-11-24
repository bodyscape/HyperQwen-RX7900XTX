import vllm._C  # registers torch.ops._C
import sys, torch, math, time
from vllm.v1.attention.ops import rocm_octave as sq
dev = 'cuda'
HQ, HKV, D, BS, CH = 24, 4, 256, 2176, 2048
fmt = sq.OctaveFormat.from_cache_dtype('octave_k3v3_compact', D, 64)
ks = torch.ops._C
signs = sq.sign_bits(D).to(dev)
scale = 1.0 / math.sqrt(D)
def run(P, reps=5):
    T = P + CH
    nb = (T + BS - 1) // BS
    key = torch.randn(T, HKV, D, device=dev, dtype=torch.float16)
    val = torch.randn(T, HKV, D, device=dev, dtype=torch.float16)
    slots = sq.reference_quantize(key, val, fmt)          # (T, H, slot)
    cache = torch.zeros(nb, HKV, BS, fmt.slot_bytes, dtype=torch.uint8, device=dev)
    pos = torch.arange(T, device=dev)
    cache[pos // BS, :, pos % BS] = slots
    bt = torch.arange(nb, device=dev, dtype=torch.int32).view(1, nb)
    q = torch.randn(CH, HQ, D, device=dev, dtype=torch.float16)
    k = key[P:].contiguous(); v = val[P:].contiguous()
    ks.octave_rotate(q, signs, False, False); ks.octave_rotate(k, signs, False, False); ks.octave_rotate(v, signs, False, False)
    o = torch.empty_like(q)
    qsl = torch.tensor([0, CH], device=dev, dtype=torch.int32)
    sl = torch.tensor([T], device=dev, dtype=torch.int32)
    f = lambda: ks.octave_prefill(o, q, k, v, cache, bt, qsl, sl, CH, scale, fmt.kernel_code)
    f(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): f()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / reps
    pairs = CH * P + CH * (CH + 1) / 2
    tf = 4 * HQ * D * pairs / ms / 1e9
    print(f'prefix {P:6d} chunk {CH}: {ms:7.2f} ms  {tf:5.1f} TFLOPs (QK+PV)  [{pairs*1.0/1e6:.0f}M pairs]')
    del cache, slots, key, val
    torch.cuda.empty_cache()
for P in [int(x) for x in sys.argv[1:]] or [0, 4352, 13056, 26112, 52224]:
    run(P)
