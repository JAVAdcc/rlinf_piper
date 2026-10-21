#!/usr/bin/env bash
# Prepare or run an operator-controlled shared RLT actor evaluation.
set -euo pipefail
piper_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
piper_python=${PIPER_PYTHON:-"$piper_repo/.venv-piper/bin/python"}
piper_checkpoint=${PIPER_STAGE1_PATH:-"$piper_repo/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor"}
: "${PIPER_ACTOR_PATH:?Set PIPER_ACTOR_PATH to the extracted shared actor directory}"
piper_mode=${1:-infer-reference}
if (($#)); then shift; fi
cd -- "$piper_repo"
export PYTHONPATH="$piper_repo${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export JAX_PLATFORMS=cpu
case "$piper_mode" in
  infer-reference|infer-actor)
    : "${PIPER_DATA_PATH:?Set PIPER_DATA_PATH to a local LeRobot episode for hardware-free inference}"
    piper_args=(infer --dataset "$PIPER_DATA_PATH")
    if [[ "$piper_mode" == infer-actor ]]; then piper_args+=(--infer-actor); fi
    ;;
  rollout)
    piper_args=(rollout --hardware "$piper_repo/examples/embodiment/config/piper_usb_hardware.yaml"
      --task 'pick up the two USB cables and insert them into the USB port on the outlet.'
      --interactive --num-episodes 1 --steps 3600 --replan-steps 50
      --record-dataset --reset-after-episode none)
    ;;
  *)
    printf 'Usage: %s [infer-reference|infer-actor|rollout] [run_policy options]\n' "$0" >&2
    exit 2
    ;;
esac
exec "$piper_python" -m toolkits.piper.run_policy "${piper_args[@]}" \
  --checkpoint "$piper_checkpoint" --actor "$PIPER_ACTOR_PATH" \
  --actor-mode manual-critical --allow-offline-actor "$@"
