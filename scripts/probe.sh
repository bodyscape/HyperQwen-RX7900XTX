export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/wsl/lib
run() { echo "=== $1"; shift; "$@" 2>&1 | grep -v -i warn | tail -6; echo "exit=${PIPESTATUS[0]}"; }
T='import torch; print("is_available", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")'
run "A: HSA_ENABLE_DXG_DETECTION=1" env HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0 python3 -c "$T"
run "B: no DXG env vars" env ROCPROFILER_REGISTER_ENABLED=0 python3 -c "$T"
run "C: HSA_ENABLE_DXG_DETECTION=1, core libs first" env HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0 LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:/usr/lib/wsl/lib python3 -c "$T"
