import torch, time
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
dev='cuda'; dt=torch.float16
HQ,HKV,D,CH=24,4,256,2048
def bench(fn, reps=5):
    fn(); torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/reps
for P in (4352, 26112, 52224):
    q=torch.randn(1,HQ,CH,D,device=dev,dtype=dt)
    k=torch.randn(1,HKV,P,D,device=dev,dtype=dt); v=torch.randn(1,HKV,P,D,device=dev,dtype=dt)
    kx=k.repeat_interleave(HQ//HKV,1); vx=v.repeat_interleave(HQ//HKV,1)
    fl=4*HQ*D*CH*P
    for name,be in (('flash',SDPBackend.FLASH_ATTENTION),('efficient',SDPBackend.EFFICIENT_ATTENTION),('math',None)):
        try:
            if be is None:
                if P>10000: continue
                ctx=sdpa_kernel(SDPBackend.MATH)
            else: ctx=sdpa_kernel(be)
            with ctx:
                ms=bench(lambda: F.scaled_dot_product_attention(q,kx,vx))
            print(f'P={P:6d} {name:9s} expanded-kv: {ms:7.2f} ms  {fl/ms/1e9:5.1f} TFLOPs')
        except Exception as ex:
            print(f'P={P:6d} {name:9s}: ERR {str(ex)[:90]}')
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            ms=bench(lambda: F.scaled_dot_product_attention(q,k,v,enable_gqa=True))
        print(f'P={P:6d} flash gqa native: {ms:7.2f} ms  {fl/ms/1e9:5.1f} TFLOPs')
    except Exception as ex: print('gqa ERR',str(ex)[:90])
    del q,k,v,kx,vx; torch.cuda.empty_cache()
