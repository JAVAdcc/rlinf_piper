#!/usr/bin/env bash
# Native RLT learning at operator-confirmed Piper episode boundaries.
set -euo pipefail
piper_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
piper_python=${PIPER_PYTHON:-"$piper_repo/.venv-piper/bin/python"}
cd -- "$piper_repo"
export PYTHONPATH="$piper_repo${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
# Resume restores all collection, training and checkpoint bindings.
for piper_arg in "$@"; do
  if [[ "$piper_arg" == --resume || "$piper_arg" == --resume=* ]]; then
    exec "$piper_python" -m toolkits.piper.run_policy rollout "$@"
  fi
done
: "${PIPER_ACTOR_PATH:?Set PIPER_ACTOR_PATH to an actor with a warmed-up critic}"
piper_replay_args=()
if [[ -n "${PIPER_OFFLINE_CACHE:-}" ]]; then
  piper_replay_args+=(--offline-cache "$PIPER_OFFLINE_CACHE")
fi
exec "$piper_python" -m toolkits.piper.run_policy rollout \
  --online-config "$piper_repo/examples/embodiment/config/piper_rlt_online.yaml" \
  --checkpoint "${PIPER_STAGE1_PATH:-$piper_repo/checkpoints/piper_stage1_h50_fp32_20261003/global_step_50000/actor}" \
  --actor "$PIPER_ACTOR_PATH" \
  --actor-mode manual-critical --allow-offline-actor \
  --hardware "$piper_repo/examples/embodiment/config/piper_usb_hardware.yaml" \
  --task 'pick up the two USB cables and insert them into the USB port on the outlet.' \
  --interactive --record-dataset --num-episodes 100 --steps 7200 --replan-steps 50 \
  --reset-after-episode initial "${piper_replay_args[@]}" "$@"
