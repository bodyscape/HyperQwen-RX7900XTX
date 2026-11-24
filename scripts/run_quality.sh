#!/bin/bash
# GSM8K (100 questions, repo's quality_battery.py) on both servers at the same time; one result file per server.
cd /app
export PATH=/app/venv/bin:$PATH
N=${1:-100}
VLLM_API=http://127.0.0.1:19020 timeout 900 python bench/quality_battery.py rocm7900 --gsm-only --gsm-n $N > /tmp/q_rocm.log 2>&1 &
P1=$!
VLLM_API=http://IP_3090:18020 timeout 900 python bench/quality_battery.py rtx3090 --gsm-only --gsm-n $N > /tmp/q_3090.log 2>&1 &
P2=$!
wait $P1 $P2
echo "--- 7900 XTX:"; grep -E 'GSM8K|Error|error|Traceback' /tmp/q_rocm.log | tail -3
echo "--- RTX 3090:"; grep -E 'GSM8K|Error|error|Traceback' /tmp/q_3090.log | tail -3
