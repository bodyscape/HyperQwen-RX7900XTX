#!/bin/bash
# Prepare the comparison inside the hyperqwen-rocm container: GSM8K test data, pyarrow, reachability of both servers.
cd /app
/app/venv/bin/python -c 'import pyarrow; print("pyarrow", pyarrow.__version__)' 2>&1 | tail -1
for u in http://127.0.0.1:19020 http://IP_3090:18020; do
  echo "$u/health -> $(curl -s -m 5 -o /dev/null -w '%{http_code}' $u/health)"
done
mkdir -p bench/quality-data
if [ ! -f bench/quality-data/gsm8k/main/test-00000-of-00001.parquet ]; then
  export HF_HUB_ENABLE_HF_TRANSFER=0
  hf download openai/gsm8k --repo-type dataset --include "main/test-*" --local-dir bench/quality-data/gsm8k 2>&1 | tail -2
fi
ls -la bench/quality-data/gsm8k/main/ 2>&1 | tail -3
ls -la /cache/bench/labd_corpus.txt
