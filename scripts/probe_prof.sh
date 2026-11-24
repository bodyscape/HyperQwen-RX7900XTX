cd /app
echo "--- profiler flags:"
venv/bin/vllm serve --help=all 2>/dev/null | grep -i -E 'profil' | head -20
echo "--- profiler env vars in vllm/envs.py:"
grep -n -i 'PROFILER' /usr/local/lib/python3.12/dist-packages/vllm/envs.py | head -12
echo "--- ProfilerConfig:"
grep -n -E 'class ProfilerConfig|profiler:|torch_profiler_dir|delay_iterations|max_iterations|torch_profiler_with' /usr/local/lib/python3.12/dist-packages/vllm/config/profiler.py | head -20
echo "--- key env:"
env | grep -E '^(SPEC|DRAFT|CUDAGRAPH|GPU_UTIL|MAX_|VLLM_SPEC|HSA|ROCPROF)' | sort
echo "--- spec config in launched cmd:"
ps -eo args | grep -E 'vllm serve' | grep -v grep | head -1 | tr ' ' '\n' | grep -A1 -E 'speculative-config|compilation-config|async' | head -8
