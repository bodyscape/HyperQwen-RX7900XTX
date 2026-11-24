"""Does a process that only initialises the GPU runtime and then sleeps burn CPU? usage: idle_cpu_probe.py <init|noinit> """
import os, sys, time
mode = sys.argv[1]
import torch
if mode == "init":
    x = torch.zeros(1, device="cuda")   # creates the HIP context and the HSA runtime threads
    torch.cuda.synchronize()
time.sleep(3)                            # let any start-up work finish
t0 = time.process_time(); w0 = time.time()
time.sleep(10)
cpu = time.process_time() - t0; wall = time.time() - w0
print("RESULT %-7s HSA_ENABLE_INTERRUPT=%-5s : %.2f cores busy while idle (%.1f cpu-s in %.1f s)"
      % (mode, os.environ.get("HSA_ENABLE_INTERRUPT", "unset"), cpu / wall, cpu, wall))
