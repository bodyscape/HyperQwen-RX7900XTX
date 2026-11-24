#!/bin/bash
# One process per variant: each build registers the same hq:: ops.
cd /work/gemm
cp /repo_rocm/w4a16_gemv.cu /work/gemm/w4a16_gemv.cu
run() { /app/venv/bin/python -I gemv_variant.py "$@" 2>&1 | grep -E '^VARIANT|Error|error:' | head -3; }
run base
run nt -DHQ_NT=1
run rif16 -DHQ_RIF=16
run nt16 -DHQ_NT=1 -DHQ_RIF=16
run w4 -DHQ_WAVES=4
run nt_w4 -DHQ_NT=1 -DHQ_WAVES=4
run kt512 -DHQ_KSPLIT_TARGET=512
run nt_kt512 -DHQ_NT=1 -DHQ_KSPLIT_TARGET=512
