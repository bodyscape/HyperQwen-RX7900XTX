import os, torch
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load
os.makedirs('/work/gemm/buildw', exist_ok=True)
load(name='gemvw', sources=['/work/gemm/w4a16_gemv.cu'], build_directory='/work/gemm/buildw', extra_cuda_cflags=['-O3'], is_python_module=False)
dev='cuda'; dt=torch.bfloat16; g=128
def t(fn,n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/n*1000
def ref_dq(qw,qz,sc,K,N):
    r=torch.arange(K//8,device=dev); w=torch.empty(K,N,dtype=torch.float32,device=dev)
    for i in range(4):
        w[8*r+2*i]=((qw[r]>>(4*i))&0xF).float(); w[8*r+2*i+1]=((qw[r]>>(4*i+16))&0xF).float()
    n=torch.arange(N,device=dev); z=(((qz[:,n//8]>>((n%8)*4))&0xF)+1).float(); gi=torch.arange(K,device=dev)//g
    return ((w-z[gi])*sc.float()[gi]).to(dt)
for name,K,N in [('gate_up',5120,34816),('down',17408,5120),('qkvz',5120,16384),('o',6144,5120)]:
    qw=torch.randint(-2**31,2**31-1,(K//8,N),dtype=torch.int32,device=dev)
    qz=torch.randint(-2**31,2**31-1,(K//g,N//8),dtype=torch.int32,device=dev)
    sc=(torch.rand(K//g,N,device=dev)*0.02+0.002).to(dt)
    out=torch.ops.hq.w4a16_dequant(qw,qz,sc,False)
    ok = torch.equal(out, ref_dq(qw,qz,sc,K,N))
    us=t(lambda: torch.ops.hq.w4a16_dequant(qw,qz,sc,False))
    byt=K*N*2 + K*N//2
    a=torch.randn(2048,K,device=dev,dtype=dt)
    us_mm=t(lambda: a@out)
    print(f'{name:8s} {K}x{N}: dequant {us:7.0f} us ({byt/us/1e3:4.0f} GB/s) exact={ok} | matmul M=2048 {us_mm:7.0f} us -> dequant share {100*us/(us+us_mm):4.1f}%')
