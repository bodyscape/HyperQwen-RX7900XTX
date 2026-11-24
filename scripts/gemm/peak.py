import torch
dev='cuda'; dt=torch.bfloat16
def t(fn,n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n
for M,K,N in [(2048,5120,34816),(2048,17408,5120),(2048,5120,16384),(2048,6144,5120),(2048,5120,14336),(4096,5120,34816),(4096,17408,5120),(8192,8192,8192),(1024,5120,34816)]:
    a=torch.randn(M,K,device=dev,dtype=dt); b=torch.randn(K,N,device=dev,dtype=dt)
    ms=t(lambda: a@b)
    print(f'M={M:5d} K={K:6d} N={N:6d}: {ms*1000:8.0f} us  {2*M*K*N/ms/1e9:6.1f} TFLOPs')
