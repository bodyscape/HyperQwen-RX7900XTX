#!/bin/bash
# Long context on both servers at the same time: labd at ~45k tokens (same frozen corpus, same six tasks), then 3 needles at ~120k.
cd /app
export PATH=/app/venv/bin:$PATH
STEP=${1:-labd}
if [ "$STEP" = "labd" ]; then
  (VLLM_API=http://127.0.0.1:19020 timeout 560 python bench/labd_bench.py rocm45 --ctx 40000 > /tmp/lb_rocm.log 2>&1) &
  P1=$!
  (VLLM_API=http://IP_3090:18020 timeout 560 python bench/labd_bench.py rtx45 --ctx 40000 > /tmp/lb_3090.log 2>&1) &
  P2=$!
  wait $P1 $P2
  echo "--- 7900 XTX:"; grep -E '^LABD|rror' /tmp/lb_rocm.log | tail -8
  echo "--- RTX 3090:"; grep -E '^LABD|rror' /tmp/lb_3090.log | tail -8
else
  (python /tmp/needle_compare.py http://127.0.0.1:19020 3150 rocm > /tmp/nd_rocm.log 2>&1) &
  P1=$!
  (python /tmp/needle_compare.py http://IP_3090:18020 3150 rtx3090 > /tmp/nd_3090.log 2>&1) &
  P2=$!
  wait $P1 $P2
  echo "--- 7900 XTX:"; cat /tmp/nd_rocm.log | tail -5
  echo "--- RTX 3090:"; cat /tmp/nd_3090.log | tail -5
fi

