echo "--- librocdxg / hsa / dxcore:"
find / -xdev \( -name 'librocdxg*' -o -name 'libhsa-runtime64*' -o -name 'libdxcore*' \) 2>/dev/null | head -20
echo "--- python/torch/vllm:"
python3 -c "import sys,torch,vllm;print(sys.version.split()[0],torch.__version__,torch.version.hip,vllm.__version__)" 2>&1 | tail -2
echo "--- /dev/dxg:"
ls -l /dev/dxg 2>&1
echo "--- ROCm dirs:"
ls /opt 2>&1 | head
