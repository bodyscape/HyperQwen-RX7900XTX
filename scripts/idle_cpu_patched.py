"""Idle CPU of a GPU-initialised process, and which libhsa-runtime64 it really loaded. usage: idle_cpu_patched.py"""
import os, time, torch
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libhsa-runtime64" in l or "librocdxg" in l})
time.sleep(3)
t0 = time.process_time(); w0 = time.time()
time.sleep(10)
cpu = time.process_time() - t0; wall = time.time() - w0
print("LIBS", libs)
print("RESULT idle: %.2f cores busy (%.2f cpu-s in %.1f s)" % (cpu / wall, cpu, wall))
# a real round trip still works and how long a small sync takes (latency sanity)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
s = time.perf_counter()
for _ in range(200):
    (a @ a); torch.cuda.synchronize()
print("RESULT 200 x (matmul + synchronize): %.1f ms each" % ((time.perf_counter() - s) / 200 * 1000))
