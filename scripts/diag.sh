D=/usr/local/lib/python3.12/dist-packages
echo "--- files:"
ls -l $D/_rocm_sdk_core/lib/librocdxg* $D/_rocm_sdk_core/lib/libhsa-runtime64* $D/_rocm_sdk_devel/lib/librocdxg* $D/_rocm_sdk_devel/lib/libhsa-runtime64* 2>&1
echo "--- sha256:"
sha256sum $D/_rocm_sdk_core/lib/librocdxg.so.1 $D/_rocm_sdk_devel/lib/librocdxg.so.1.1.0 $D/_rocm_sdk_core/lib/libhsa-runtime64.so.1 $D/_rocm_sdk_devel/lib/libhsa-runtime64.so.1.21.0 2>&1
echo "--- who exports / imports hsaKmtGetDefaultHostGpu:"
for f in $D/_rocm_sdk_core/lib/librocdxg.so.1 $D/_rocm_sdk_devel/lib/librocdxg.so.1.1.0 $D/_rocm_sdk_core/lib/libhsa-runtime64.so.1 $D/_rocm_sdk_devel/lib/libhsa-runtime64.so.1.21.0; do
  printf '%s: ' "$f"; (nm -D "$f" 2>/dev/null || objdump -T "$f" 2>/dev/null) | grep -c hsaKmtGetDefaultHostGpu
done
echo "--- rocm_sdk version:"
python3 - <<'EOF' 2>&1 | tail -5
import importlib.metadata as m
for p in ("rocm", "rocm-sdk-core", "rocm-sdk-devel", "torch", "vllm"):
    try: print(p, m.version(p))
    except Exception as e: print(p, "n/a")
EOF
echo "--- librocdxg strings (version):"
strings $D/_rocm_sdk_core/lib/librocdxg.so.1 2>/dev/null | grep -iE 'rocdxg.*[0-9]+\.[0-9]+\.[0-9]+|^1\.[0-9]+\.[0-9]+$' | head -5
