#!/bin/bash
# Earlier skill-library pipeline (segment→pair→fork→extract→summarize→validate).
# Distinct from the v4 Hypothesis-DSL discovery that produced the 72-rule book.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/third_party/Achievement-Distillation:${PYTHONPATH:-}"
if [[ -x "$HOME/rl2lang/envs/ad/bin/python" ]]; then
  export PATH="$HOME/rl2lang/envs/ad/bin:$PATH"
fi

TRAIN_RAW="${TRAIN_RAW:-$ROOT/artifacts/teacher/raw/teacher_e250_train}"
VALIDATION_RAW="${VALIDATION_RAW:-$ROOT/artifacts/teacher/raw/teacher_e250_validation}"
WORK_DIR="${WORK_DIR:-$ROOT/artifacts/skill_build/v1}"
LIBRARY_DIR="${LIBRARY_DIR:-$ROOT/artifacts/skill_library}"
mkdir -p "$WORK_DIR" "$LIBRARY_DIR"

python -m skill_distill.segment "$TRAIN_RAW" --out "$WORK_DIR/train.events.jsonl"
python -m skill_distill.segment "$VALIDATION_RAW" --out "$WORK_DIR/validation.events.jsonl"
python -m skill_distill.pair --events "$WORK_DIR/train.events.jsonl" --out "$WORK_DIR/train.pairs.jsonl"
python -m skill_distill.fork --events "$WORK_DIR/train.events.jsonl" --out "$WORK_DIR/train.forks.jsonl" --skip-unreplayable
python -m skill_distill.extract \
  --pairs "$WORK_DIR/train.pairs.jsonl" \
  --out "$WORK_DIR/skills.candidate.jsonl" \
  --prompts-out "$WORK_DIR/summarizer.prompts.jsonl"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-openai}"
else
  SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-template}"
fi
python -m skill_distill.summarize \
  --skills "$WORK_DIR/skills.candidate.jsonl" \
  --pairs "$WORK_DIR/train.pairs.jsonl" \
  --backend "$SUMMARIZER_BACKEND" \
  --out "$WORK_DIR/skills.natural-language.jsonl" \
  --book "$WORK_DIR/skill-book.pre-validation.txt"
python -m skill_distill.validate \
  --skills "$WORK_DIR/skills.natural-language.jsonl" \
  --events "$WORK_DIR/train.events.jsonl" \
  --forks "$WORK_DIR/train.forks.jsonl" \
  --validation-events "$WORK_DIR/validation.events.jsonl" \
  --out "$LIBRARY_DIR/skills.v1.jsonl" \
  --report "$LIBRARY_DIR/skills.v1.validation.json" \
  --book "$LIBRARY_DIR/skill-book.v1.txt"

echo "Skill library: $LIBRARY_DIR/skills.v1.jsonl"
