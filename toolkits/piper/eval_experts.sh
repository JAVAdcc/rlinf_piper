#!/usr/bin/env bash
# Explicitly select one of the two C50 heads for fixed-weight evaluation.
set -euo pipefail
piper_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
piper_python=${PIPER_PYTHON:-"$piper_repo/.venv-piper/bin/python"}
cd -- "$piper_repo"
export PYTHONPATH="$piper_repo${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 JAX_PLATFORMS=cpu
piper_mode=infer-reference
if (($#)) && [[ "$1" != --* ]]; then
  piper_mode=$1
  shift
fi
for piper_arg in "$@"; do
  if [[ "$piper_arg" == --resume || "$piper_arg" == --resume=* ]]; then
    exec "$piper_python" -m toolkits.piper.run_policy rollout "$@"
  fi
done
: "${PIPER_EXPERTS_PATH:?Set PIPER_EXPERTS_PATH to the extracted experts.yaml}"
case "$piper_mode" in
  infer-reference|infer-phase1|infer-phase2)
    : "${PIPER_DATA_PATH:?Set PIPER_DATA_PATH to one recorded LeRobot episode}"
    piper_args=(infer --dataset "$PIPER_DATA_PATH")
    if [[ "$piper_mode" != infer-reference ]]; then
      piper_args+=(--infer-actor --infer-expert "${piper_mode#infer-}")
    fi
    ;;
  rollout)
    piper_args=(rollout --hardware "$piper_repo/examples/embodiment/config/piper_usb_hardware.yaml"
      --task 'pick up the two USB cables and insert them into the USB port on the outlet.'
      --interactive --record-dataset --num-episodes 1 --steps 7200 --replan-steps 50
      --save-rlt-reference --reset-after-episode initial)
    ;;
  *) printf 'Usage: %s [infer-reference|infer-phase1|infer-phase2|rollout] [options]\n' "$0" >&2; exit 2 ;;
esac
exec "$piper_python" -m toolkits.piper.run_policy "${piper_args[@]}" \
  --checkpoint "${PIPER_STAGE1_PATH:-$piper_repo/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor}" \
  --experts "$PIPER_EXPERTS_PATH" --actor-mode manual-critical --allow-offline-actor "$@"
