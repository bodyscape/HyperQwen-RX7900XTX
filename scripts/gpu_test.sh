echo "--- /dev/dxg:"; ls -l /dev/dxg 2>&1
echo "--- /usr/lib/wsl/lib:"; ls /usr/lib/wsl/lib 2>&1 | head
cat > /tmp/t.py <<'EOF'
import time, torch
print("torch", torch.__version__, "hip", torch.version.hip)
ok = torch.cuda.is_available()
print("cuda.is_available:", ok)
if not ok:
    raise SystemExit(2)
p = torch.cuda.get_device_properties(0)
print("device:", p.name, getattr(p, "gcnArchName", "?"), "total GiB: %.2f" % (p.total_memory / 2**30))
free, total = torch.cuda.mem_get_info()
print("mem_get_info free/total GiB: %.2f / %.2f" % (free / 2**30, total / 2**30))
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(3): (a @ b)
torch.cuda.synchronize(); t = time.time()
n = 20
for _ in range(n): (a @ b)
torch.cuda.synchronize(); dt = time.time() - t
print("fp16 matmul 8192^3: %.1f TFLOPS" % (2 * 8192**3 * n / dt / 1e12))
x = torch.empty(2**30 // 2, dtype=torch.float16, device="cuda")  # 1 GiB
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t = time.time()
n = 20
for _ in range(n): y.copy_(x)
torch.cuda.synchronize(); dt = time.time() - t
print("D2D copy: %.0f GB/s (read+write counted)" % (2 * 2**30 * n / dt / 1e9))
from vllm.platforms import current_platform
print("vllm platform:", current_platform.__class__.__name__, "pin_memory:", current_platform.is_pin_memory_available())
EOF
python3 /tmp/t.py 2>&1 | grep -v Warning
