#!/usr/bin/env bash
# Fixed C50 evaluation; online training has a separate entrypoint.
set -euo pipefail
piper_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
piper_python=${PIPER_PYTHON:-"$piper_repo/.venv-piper/bin/python"}
export PIPER_ACTOR_PATH=${PIPER_C50_ACTOR_PATH:-"$piper_repo/checkpoints/piper_shared_c50_bc5000_20261022/actor"}
"$piper_python" - "$PIPER_ACTOR_PATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
if not (root / "piper_rlt_actor.json").exists():
    root /= "actor"
model = json.loads((root / "piper_rlt_actor.json").read_text())["model"]
if (model["num_action_chunks"], model["ref_num_action_chunks"]) != (50, 50):
    raise SystemExit("C50 evaluation requires actor C50 and reference H50.")
PY
piper_mode=${1:-rollout}
if (($#)); then shift; fi
if [[ "$piper_mode" == rollout ]]; then
  exec bash "$piper_repo/toolkits/piper/eval_shared.sh" rollout "$@" --replan-steps 50
fi
exec bash "$piper_repo/toolkits/piper/eval_shared.sh" "$piper_mode" "$@"
