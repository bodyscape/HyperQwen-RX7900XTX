#!/bin/bash
# Container entrypoint. First argument selects what to run:
#   single   single-user/start_qwen.sh  (MTP speculative decoding, low latency)
#   batch    batch/start_qwen.sh        (throughput)
#   turing   start-turing.sh            (RTX 20-series / sm75, fp16; see docs/turing.md)
#   rocm     start-rocm.sh              (AMD Radeon RX 7900 XTX / gfx1100; Dockerfile.rocm image, see docs/rocm.md)
#   prepare  docker/prepare.sh          (download + requantize the model into /app/models)
#   verify   verify.sh [args]
#   <anything else> is exec'd as a command (e.g. bash)
# Before serving, docker/prepare.sh runs (idempotent: a state check and
# seconds when the model is already prepared, the download + requantization
# otherwise; PREPARE=0 skips it — this is what makes a bare `docker run` with
# an empty models volume work), then verify.sh --no-server, which aborts on
# FAIL (patches missing, ...); VERIFY=0 skips that.
set -e
cd /app
cmd=${1:-single}; shift || true
# rocm on WSL2 (Docker Desktop): the card is /dev/dxg, and ROCm only looks for it with these set. They have to
# be in the environment before verify.sh below, which asks torch for the GPU; start-rocm.sh sets them again,
# for a launch that does not come through this script.
if [ "$cmd" = rocm ] && [ -e /dev/dxg ]; then
  export HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION:-1}
  export ROCPROFILER_REGISTER_ENABLED=${ROCPROFILER_REGISTER_ENABLED:-0}
fi
case "$cmd" in
  single|batch|turing|rocm)
    echo "entrypoint: first-boot stages: [1/3] docker/prepare.sh [2/3] verify.sh --no-server [3/3] start_qwen.sh ($cmd)"
    echo "entrypoint: stage 1 is minutes on first boot (download + requantize), seconds once prepared"
    echo "entrypoint: use log timestamps as the elapsed timer; total first boot can exceed 30 min"
    if [ "${PREPARE:-1}" != "0" ]; then
      bash docker/prepare.sh
    fi
    # Preparation may create the fast variant. Resolve it now, once, so the
    # verifier and launcher use the same model; batch retains its base default.
    if [ "$cmd" = single ] || [ "$cmd" = rocm ]; then
      REPO=/app
      source "$REPO/single-user/select_model.sh"
      export MODEL
    fi
    if [ "${VERIFY:-1}" != "0" ]; then
      bash verify.sh --no-server || { echo "entrypoint: verify.sh FAILED — fix the above or set VERIFY=0"; exit 1; }
    fi
    case "$cmd" in
      single) exec bash single-user/start_qwen.sh "$@" ;;
      turing) exec bash start-turing.sh "$@" ;;
      rocm)   exec bash start-rocm.sh "$@" ;;
      *)      exec bash batch/start_qwen.sh "$@" ;;
    esac ;;
  prepare) exec bash docker/prepare.sh "$@" ;;
  verify)  exec bash verify.sh "$@" ;;
  *)       exec "$cmd" "$@" ;;
esac
