#!/bin/bash
# The repo's single-stream cohort (bench/run_benchmarks.sh single, concurrency 1) on both servers at the same time.
cd /app
export PATH=/app/venv/bin:$PATH
sed 's/for C in 1 2 4 8/for C in 1/' bench/run_benchmarks.sh > bench/rb_c1.sh
(HOST=127.0.0.1 PORT=19020 OUT=/tmp/co_rocm timeout 900 bash bench/rb_c1.sh single > /tmp/co_rocm.log 2>&1) &
P1=$!
(HOST=IP_3090 PORT=18020 OUT=/tmp/co_3090 timeout 900 bash bench/rb_c1.sh single > /tmp/co_3090.log 2>&1) &
P2=$!
wait $P1 $P2
echo "--- 7900 XTX:"; grep -E '^ROW|rror' /tmp/co_rocm.log | tail -4
echo "--- RTX 3090:"; grep -E '^ROW|rror' /tmp/co_3090.log | tail -4
