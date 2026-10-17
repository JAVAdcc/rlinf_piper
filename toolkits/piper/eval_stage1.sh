#!/usr/bin/env bash
# Evaluate the verified Piper H50 FP32 Stage 1 checkpoint on this workstation.
set -euo pipefail

piper_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
piper_python=${PIPER_PYTHON:-"$piper_repo/.venv-piper/bin/python"}
piper_checkpoint=${PIPER_STAGE1_PATH:-"$piper_repo/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"}
piper_mode=${1:-infer}
if (($#)); then shift; fi
cd -- "$piper_repo"
export PYTHONPATH="$piper_repo${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export JAX_PLATFORMS=cpu

case "$piper_mode" in
  infer)
    exec "$piper_python" -m toolkits.piper.run_policy infer \
      --checkpoint "$piper_checkpoint" \
      --dataset "${PIPER_DATA_PATH:-/home/descfly/code/rlinf-piper-check/datasets/usb_iter2_rlt}" \
      "$@"
    ;;
  rollout)
    exec "$piper_python" -m toolkits.piper.run_policy rollout \
      --checkpoint "$piper_checkpoint" \
      --hardware "$piper_repo/examples/embodiment/config/piper_usb_hardware.yaml" \
      --task 'pick up the two USB cables and insert them into the USB port on the outlet.' \
      --interactive --num-episodes 20 --steps 3600 --replan-steps 50 \
      "$@"
    ;;
  *)
    printf 'Usage: %s [infer|rollout] [run_policy options]\n' "$0" >&2
    exit 2
    ;;
esac
