D=/usr/local/lib/python3.12/dist-packages/vllm
echo "--- _C archs / octave strings:"
ls -la $D/_C*.so 2>&1 | head -3
(command -v llvm-objdump >/dev/null && llvm-objdump --offloading $D/_C.abi3.so 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort | uniq -c) || echo "no llvm-objdump"
for f in $D/_C.abi3.so; do echo "octave symbols in $f: $(strings -a $f | grep -ci octave)"; done
echo "--- torch.ops._C octave attrs:"
/app/venv/bin/python - <<'EOF' 2>&1 | grep -vE 'UserWarning|raw_cnt|agent.cpp|pin_memory'
import torch, vllm._C
for n in ("octave_cache_store", "octave_decode", "octave_prefill", "octave_rotate", "gptq_gemm_rdna3"):
    try:
        getattr(torch.ops._C, n); print(n, "present")
    except Exception as e:
        print(n, "MISSING:", type(e).__name__)
import vllm, os, glob
print([os.path.basename(p) for p in glob.glob(os.path.dirname(vllm.__file__) + "/**/*octave*", recursive=True)][:12])
EOF
echo "--- PYTORCH_ROCM_ARCH / env:"; env | grep -iE 'rocm_arch|gfx|AMDGPU_TARGETS' | head
