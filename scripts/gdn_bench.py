"""Time the two Triton kernels of the speculative GDN decode path (one layer, batch 1, 4 tokens), in isolation.
GPU time is taken by replaying a captured HIP graph of N back-to-back calls (no host launch cost) and also eagerly."""
import torch, time, sys
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update

dev = 'cuda'; bf = torch.bfloat16
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1      # sequences
T = 4                                                   # tokens per sequence (3 drafts + 1)
HK, HV, DK, DV = 16, 48, 128, 128
DIM = 2 * HK * DK + HV * DV
NB = 64
ssm_dt = torch.float16

conv_mem = torch.randn(NB, 3, DIM, device=dev, dtype=bf)
conv_state = conv_mem.transpose(-1, -2)                  # [NB, DIM, 3]
ssm_state = torch.randn(NB, HV, DV, DK, device=dev, dtype=ssm_dt) * 0.01
w = torch.randn(DIM, 4, device=dev, dtype=bf) * 0.1
bias = None
x = torch.randn(B * T, DIM, device=dev, dtype=bf)
qsl = torch.arange(0, B * T + 1, T, device=dev, dtype=torch.int32)
idx = (torch.arange(B * (T + 1), device=dev, dtype=torch.int32) % (NB - 1)).view(B, T + 1)[:, :T].contiguous()
idx = torch.stack([torch.arange(b * T, b * T + T, device=dev, dtype=torch.int32) % NB for b in range(B)])
acc = torch.full((B,), 2, device=dev, dtype=torch.int32)
a = torch.randn(B * T, HV, device=dev, dtype=bf)
b_ = torch.randn(B * T, HV, device=dev, dtype=bf)
A_log = torch.randn(HV, device=dev, dtype=torch.float32)
dt_bias = torch.randn(HV, device=dev, dtype=bf)
q = torch.randn(1, B * T, HK, DK, device=dev, dtype=bf)
k = torch.randn(1, B * T, HK, DK, device=dev, dtype=bf)
v = torch.randn(1, B * T, HV, DV, device=dev, dtype=bf)


def conv():
    return causal_conv1d_update(x, conv_state, w, bias, 'silu', conv_state_indices=idx[:, 0], num_accepted_tokens=acc,
                                query_start_loc=qsl, max_query_len=T, validate_data=False)


def gdr():
    return fused_sigmoid_gating_delta_rule_update(A_log=A_log, a=a, b=b_, dt_bias=dt_bias, q=q, k=k, v=v,
                                                  initial_state=ssm_state, inplace_final_state=True, cu_seqlens=qsl,
                                                  ssm_state_indices=idx, num_accepted_tokens=acc,
                                                  use_qk_l2norm_in_kernel=True)


def eager_us(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


def graph_us(fn, reps=48, replays=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(replays): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / replays / reps * 1000


print(f'B={B} T={T}: state {HV*DV*DK*2/1e6:.2f} MB per sequence slot (fp16)')
for name, fn in (('causal_conv1d_update (spec)', conv), ('fused_sigmoid_gating_delta_rule_update', gdr)):
    try:
        print(f'  {name:42s} eager {eager_us(fn):7.1f} us/call   graph {graph_us(fn):7.1f} us/call')
    except Exception as ex:
        print(f'  {name}: ERROR {type(ex).__name__}: {str(ex)[:200]}')
