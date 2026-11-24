#!/bin/bash
# The repo's cohort at concurrency 1, 2, 4 and 8 (real prompts, 8 requests of 1024 tokens, default sampling and greedy),
# on both servers at the same time.
cd /app
export PATH=/app/venv/bin:$PATH
(HOST=127.0.0.1 PORT=19020 OUT=/tmp/pa_rocm timeout 1500 bash bench/run_benchmarks.sh single > /tmp/pa_rocm.log 2>&1) &
P1=$!
(HOST=IP_3090 PORT=18020 OUT=/tmp/pa_3090 timeout 1500 bash bench/run_benchmarks.sh single > /tmp/pa_3090.log 2>&1) &
P2=$!
wait $P1 $P2
echo "--- 7900 XTX (MAX_SEQS=8):"; grep -E '^ROW|rror' /tmp/pa_rocm.log
echo "--- RTX 3090:"; grep -E '^ROW|rror' /tmp/pa_3090.log
