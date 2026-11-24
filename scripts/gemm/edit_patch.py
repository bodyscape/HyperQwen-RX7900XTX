import re
p='E:/git/HyperQwen-RX7900XTX/hyperqwen/patches/rocm-w4a16-gemv.patch'
s=open(p,newline='').read()
crlf='\r\n' in s
s=s.replace('\r\n','\n')
old_route='''+    if (
+        _HQ_GEMV
+        and a.shape[0] <= 8
+        and a.dtype == torch.bfloat16
+        and a.is_contiguous()
+        and a.data_ptr() % 16 == 0
+        and w_s.dtype == torch.bfloat16
+        and w_s.shape[0] * 128 == a.shape[1]
+        and w_s.is_contiguous()
+    ):
+        return torch.ops.hq.w4a16_gemv(a, w_q, w_zp, w_s, False, 0)
'''
new_route='''+    if (
+        _HQ_GEMV
+        and a.shape[0] <= 32
+        and a.dtype == torch.bfloat16
+        and a.is_contiguous()
+        and a.data_ptr() % 16 == 0
+        and w_s.dtype == torch.bfloat16
+        and w_s.shape[0] * 128 == a.shape[1]
+        and w_s.is_contiguous()
+    ):
+        if _HQ_WMMA and a.shape[0] > _HQ_WMMA_MIN_M:
+            return torch.ops.hq.w4a16_wmma(a, w_q, w_zp, w_s, False, 0)
+        if a.shape[0] <= 8:
+            return torch.ops.hq.w4a16_gemv(a, w_q, w_zp, w_s, False, 0)
'''
assert old_route in s
s=s.replace(old_route,new_route)
old_dq='''+_HQ_DQ = _HQ_GEMV and _HQ_DQ_MIN_M > 0 and hasattr(torch.ops.hq, "w4a16_dequant")
'''
new_dq=old_dq+'''+# 5..32 rows (2-8 concurrent speculative requests): a WMMA kernel on the same layout that reads the weights once, ~2x
+# faster than gptq_gemm_rdna3 at 12-32 rows. HQ_W4A16_WMMA=0 turns it off; below HQ_W4A16_WMMA_MIN_M+1 rows the GEMV is
+# kept (it is a little faster at 1-4 rows).
+_HQ_WMMA = (
+    _HQ_GEMV
+    and hasattr(torch.ops.hq, "w4a16_wmma")
+    and __import__("os").environ.get("HQ_W4A16_WMMA", "1") != "0"
+)
+_HQ_WMMA_MIN_M = int(__import__("os").environ.get("HQ_W4A16_WMMA_MIN_M", "4"))
'''
assert old_dq in s
s=s.replace(old_dq,new_dq)
s=s.replace("Route W4A16 GEMMs on RDNA3 (ROCm, bf16): up to 8 rows to a single-read GEMV, 1024+ rows to dequant + hipBLASLt.",
 "Route W4A16 GEMMs on RDNA3 (ROCm, bf16): 1-4 rows to a single-read GEMV, 5-32 rows to a single-read WMMA kernel,\n1024+ rows to dequant + hipBLASLt.")
s=s.replace("# --- HyperQwen: a single-read W4A16 GEMV for up to 8 activation rows","# --- HyperQwen: single-read W4A16 kernels for up to 32 activation rows")
s=s.replace("HQ_W4A16_DQ_MIN_M=0 turns the prefill path off.\n","HQ_W4A16_DQ_MIN_M=0 turns the prefill path off.\n\nThe WMMA kernel (rocm/w4a16_gemv.cu, w4a16_wmma) covers 5-32 rows: one layer, us per call at M=16, gate_up 263 -> 131,\ndown 192 -> 83, qkv 77 -> 48, o 64 -> 38; at M=32 gate_up 510 -> 206 (weights read once instead of twice).\nHQ_W4A16_WMMA=0 turns it off, HQ_W4A16_WMMA_MIN_M moves the GEMV/WMMA switch (default: WMMA above 4 rows).\n",1)
# recompute hunk headers
lines=s.split('\n')
hunks=[i for i,l in enumerate(lines) if l.startswith('@@ ')]
delta=0
for hi,start in enumerate(hunks):
    end=hunks[hi+1] if hi+1<len(hunks) else len(lines)
    body=[l for l in lines[start+1:end]]
    # drop trailing empty line artifact
    while body and body[-1]=='' : body.pop()
    old=sum(1 for l in body if l[:1] in (' ','-'))
    new=sum(1 for l in body if l[:1] in (' ','+'))
    m=re.match(r'@@ -(\d+),\d+ \+(\d+),\d+ @@(.*)',lines[start])
    os_,ns_,rest=int(m.group(1)),int(m.group(2)),m.group(3)
    lines[start]=f'@@ -{os_},{old} +{os_+delta},{new} @@{rest}'
    delta+=new-old
s='\n'.join(lines)
if crlf: s=s.replace('\n','\r\n')
open(p,'w',newline='').write(s)
print('ok')
