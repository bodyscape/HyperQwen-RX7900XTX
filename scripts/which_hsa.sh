#!/bin/bash
for p in $(pgrep -f 'vllm serve|EngineCore'); do
  n=$(grep -c 'rocr/out/libhsa' /proc/$p/maps)
  s=$(grep -c '/opt/rocm.*/libhsa-runtime64' /proc/$p/maps)
  echo "pid $p ($(ps -o comm= -p $p)): patched libhsa mapped=$n  stock libhsa mapped=$s"
done
