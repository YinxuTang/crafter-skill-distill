#!/bin/bash
# Smoke / formal eval for packaged PPO-AD teacher.
# Usage:
#   N_EP=1 bash scripts/eval_teacher.sh
#   N_EP=20 SEEDS_FILE=artifacts/teacher/matched_seeds_20.txt bash scripts/eval_teacher.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AD="$ROOT/third_party/Achievement-Distillation"

if [[ -x "$HOME/rl2lang/envs/ad/bin/python" ]]; then
  export PATH="$HOME/rl2lang/envs/ad/bin:$PATH"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
  export CUDA_VISIBLE_DEVICES=0
fi
if [[ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.261.03 ]]; then
  mkdir -p "$HOME/.local/nvidia-lib"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.261.03 "$HOME/.local/nvidia-lib/libnvidia-ml.so.1"
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.261.03 "$HOME/.local/nvidia-lib/libcuda.so.1"
  export LD_LIBRARY_PATH="$HOME/.local/nvidia-lib:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${HOME}/.local/nvidia-lib-fix:/usr/local/cuda-12.6/compat:${LD_LIBRARY_PATH:-}"

CKPT="${CKPT:-$ROOT/artifacts/teacher/agent-e250.pt}"
N_EP="${N_EP:-1}"
SEED="${SEED:-0}"
OUT="${OUT:-$ROOT/artifacts/results/teacher_eval_smoke.json}"
mkdir -p "$(dirname "$OUT")"

EXTRA=()
if [[ -n "${SEEDS_FILE:-}" ]]; then
  EXTRA+=(--seeds_file "$SEEDS_FILE")
fi

cd "$AD"
echo "Evaluating CKPT=$CKPT episodes=$N_EP seeds_file=${SEEDS_FILE:-none}"
python -u eval_score.py \
  --exp_name ppo_ad \
  --ckpt "$CKPT" \
  --num_episodes "$N_EP" \
  --seed "$SEED" \
  --out "$OUT" \
  "${EXTRA[@]}"
echo "Result: $OUT"
