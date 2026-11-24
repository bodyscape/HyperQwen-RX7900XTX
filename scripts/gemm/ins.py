p='E:/git/HyperQwen-7900XTX/hyperqwen/rocm/w4a16_gemv.cu'
s=open(p,newline='').read()
crlf='\r\n' in s
s=s.replace('\r\n','\n')
snip=open('E:/git/HyperQwen-7900XTX/scripts/gemm/wmma_snip.cu').read()
marker="// Registered ops, not pybind functions"
assert marker in s and 'wmma_kernel' not in s
s=s.replace(marker,snip+marker)
s=s.replace('  m.def("w4a16_dequant(','  m.def("w4a16_wmma(Tensor a, Tensor qweight, Tensor qzeros, Tensor scales, bool use_v2_format, int ksplit) -> Tensor");\n  m.def("w4a16_dequant(')
s=s.replace('  m.impl("w4a16_dequant"','  m.impl("w4a16_wmma", &w4a16_wmma);\n  m.impl("w4a16_dequant"')
if crlf: s=s.replace('\n','\r\n')
open(p,'w',newline='').write(s)
g=open('E:/git/HyperQwen-7900XTX/scripts/gemm/gemv16.py').read()
g=g.replace("for M in (4, 8, 9, 10, 12, 14, 16):","for M in (8, 12, 16, 24, 32):")
g=g.replace("torch.ops.hq.w4a16_gemv(","torch.ops.hq.w4a16_wmma(")
g=g.replace("build16","buildw").replace("'gemv16'","'gemvw'")
open('E:/git/HyperQwen-7900XTX/scripts/gemm/gemvw.py','w').write(g)
