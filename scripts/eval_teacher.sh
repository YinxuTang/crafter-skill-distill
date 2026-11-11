#!/bin/bash
# Teacher eval — runs entirely inside this package (uses .venv).
#   N_EP=1 bash scripts/eval_teacher.sh
#   N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt bash scripts/eval_teacher.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CKPT="${CKPT:-$ROOT/artifacts/teacher/agent-e250.pt}"
N_EP="${N_EP:-1}"
SEED="${SEED:-0}"
OUT="${OUT:-$ROOT/artifacts/results/teacher_eval_smoke.json}"
mkdir -p "$(dirname "$OUT")"

EXTRA=()
if [[ -n "${SEEDS_FILE:-}" ]]; then
  EXTRA+=(--seeds_file "$SEEDS_FILE")
fi

cd "$ROOT/third_party/Achievement-Distillation"
echo "Evaluating CKPT=$CKPT episodes=$N_EP seeds_file=${SEEDS_FILE:-none}"
"$PYTHON" -u eval_score.py \
  --exp_name ppo_ad \
  --ckpt "$CKPT" \
  --num_episodes "$N_EP" \
  --seed "$SEED" \
  --out "$OUT" \
  "${EXTRA[@]}"
echo "Result: $OUT"
