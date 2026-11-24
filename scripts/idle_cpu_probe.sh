#!/bin/bash
cd /work
export HIP_VISIBLE_DEVICES=0 HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0
P="/app/venv/bin/python -I /work/idle_cpu_probe.py"
$P noinit 2>&1 | grep RESULT
$P init 2>&1 | grep RESULT
HSA_ENABLE_INTERRUPT=1 $P init 2>&1 | grep RESULT
HSA_ENABLE_INTERRUPT=0 $P init 2>&1 | grep RESULT
