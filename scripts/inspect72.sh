echo "--- librocdxg / libhsa / libdxcore:"
find / -xdev \( -name 'librocdxg*' -o -name 'libhsa-runtime64*' -o -name 'libdxcore*' \) 2>/dev/null | head -20
echo "--- python/torch/vllm:"
python3 -c "import sys,torch,vllm;print(sys.version.split()[0],torch.__version__,torch.version.hip,vllm.__version__)" 2>&1 | tail -2
echo "--- ROCm install:"
ls -d /opt/rocm* 2>&1 | head; cat /opt/rocm/.info/version 2>&1 | head -2
echo "--- LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "--- os:"; . /etc/os-release; echo "$PRETTY_NAME"; ldd --version | head -1
echo "--- tools:"; which curl wget dpkg apt-get 2>&1
