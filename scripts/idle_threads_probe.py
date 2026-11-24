"""Which threads of an idle, GPU-initialised process use CPU, and what are they doing? (Linux /proc)"""
import os, time, torch
x = torch.zeros(1, device="cuda"); torch.cuda.synchronize()
time.sleep(3)
pid = os.getpid()
def snap():
    d = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            s = open(f"/proc/{pid}/task/{tid}/stat").read()
            name = s[s.index("(") + 1:s.rindex(")")]
            f = s[s.rindex(")") + 2:].split()
            d[tid] = (name, int(f[11]) + int(f[12]), int(f[11]), int(f[12]))   # utime+stime, utime, stime (ticks)
        except Exception:
            pass
    return d
a = snap(); time.sleep(5); b = snap()
hz = os.sysconf("SC_CLK_TCK")
print("THREADS (cpu cores over 5 s: user / system), only those above 5%:")
for tid, (name, tot, ut, st) in sorted(b.items(), key=lambda kv: -(kv[1][1] - a.get(kv[0], (0, 0, 0, 0))[1])):
    t0 = a.get(tid, (name, 0, 0, 0))
    dt = (tot - t0[1]) / hz / 5
    if dt > 0.05:
        try:
            wchan = open(f"/proc/{pid}/task/{tid}/wchan").read().strip()
        except Exception:
            wchan = "?"
        print("  tid %-6s %-18s %.2f cores  (user %.2f, sys %.2f)  wchan=%s"
              % (tid, name, dt, (ut - t0[2]) / hz / 5, (st - t0[3]) / hz / 5, wchan))
