#!/bin/bash
# Collect teacher trajectories for distillation (GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/third_party/Achievement-Distillation:${PYTHONPATH:-}"
if [[ -x "$HOME/rl2lang/envs/ad/bin/python" ]]; then
  export PATH="$HOME/rl2lang/envs/ad/bin:$PATH"
fi
export LD_LIBRARY_PATH="${HOME}/.local/nvidia-lib-fix:/usr/local/cuda-12.6/compat:${LD_LIBRARY_PATH:-}"

CKPT="${CKPT:-$ROOT/artifacts/teacher/agent-e250.pt}"
SPLIT="${SPLIT:-exploration}"
SEEDS_DIR="${SEEDS_DIR:-$ROOT/artifacts/teacher/seeds}"
SEEDS_FILE="${SEEDS_FILE:-$SEEDS_DIR/${SPLIT}_seeds.txt}"
OUT="${OUT:-$ROOT/artifacts/teacher/raw/teacher_e250_${SPLIT}}"
MAX_STEPS="${MAX_STEPS:-10000}"
FORK_HORIZON="${FORK_HORIZON:-9}"

mkdir -p "$(dirname "$SEEDS_FILE")" "$(dirname "$OUT")"
if [[ ! -f "$SEEDS_FILE" ]]; then
  python -m skill_distill.make_seed_splits \
    --out-dir "$SEEDS_DIR" \
    --exclude "$ROOT/artifacts/teacher/matched_seeds_20.txt"
fi

python -m skill_distill.collect_teacher \
  --ckpt "$CKPT" \
  --config "$ROOT/third_party/Achievement-Distillation/configs/ppo_ad.yaml" \
  --seeds-file "$SEEDS_FILE" \
  --max-steps "$MAX_STEPS" \
  --fork-horizon "$FORK_HORIZON" \
  --out "$OUT"
