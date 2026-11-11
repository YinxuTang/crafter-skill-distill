#!/bin/bash
# Collect teacher trajectories into this package (GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

CKPT="${CKPT:-$ROOT/artifacts/teacher/agent-e250.pt}"
SPLIT="${SPLIT:-exploration}"
SEEDS_DIR="${SEEDS_DIR:-$ROOT/artifacts/teacher/seeds}"
SEEDS_FILE="${SEEDS_FILE:-$SEEDS_DIR/${SPLIT}_seeds.txt}"
OUT="${OUT:-$ROOT/artifacts/teacher/raw/teacher_e250_${SPLIT}}"
MAX_STEPS="${MAX_STEPS:-10000}"
FORK_HORIZON="${FORK_HORIZON:-9}"

mkdir -p "$(dirname "$SEEDS_FILE")" "$(dirname "$OUT")"
if [[ ! -f "$SEEDS_FILE" ]]; then
  "$PYTHON" -m skill_distill.make_seed_splits \
    --out-dir "$SEEDS_DIR" \
    --exclude "$ROOT/artifacts/teacher/matched_seeds_20.txt"
fi

"$PYTHON" -m skill_distill.collect_teacher \
  --ckpt "$CKPT" \
  --config "$ROOT/third_party/Achievement-Distillation/configs/ppo_ad.yaml" \
  --seeds-file "$SEEDS_FILE" \
  --max-steps "$MAX_STEPS" \
  --fork-horizon "$FORK_HORIZON" \
  --out "$OUT"
