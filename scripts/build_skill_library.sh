#!/bin/bash
# Earlier skill-library pipeline (in-repo paths only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_env.sh"

TRAIN_RAW="${TRAIN_RAW:-$ROOT/artifacts/teacher/raw/teacher_e250_train}"
VALIDATION_RAW="${VALIDATION_RAW:-$ROOT/artifacts/teacher/raw/teacher_e250_validation}"
WORK_DIR="${WORK_DIR:-$ROOT/artifacts/skill_build/v1}"
LIBRARY_DIR="${LIBRARY_DIR:-$ROOT/artifacts/skill_library}"
mkdir -p "$WORK_DIR" "$LIBRARY_DIR"

"$PYTHON" -m skill_distill.segment "$TRAIN_RAW" --out "$WORK_DIR/train.events.jsonl"
"$PYTHON" -m skill_distill.segment "$VALIDATION_RAW" --out "$WORK_DIR/validation.events.jsonl"
"$PYTHON" -m skill_distill.pair --events "$WORK_DIR/train.events.jsonl" --out "$WORK_DIR/train.pairs.jsonl"
"$PYTHON" -m skill_distill.fork --events "$WORK_DIR/train.events.jsonl" --out "$WORK_DIR/train.forks.jsonl" --skip-unreplayable
"$PYTHON" -m skill_distill.extract \
  --pairs "$WORK_DIR/train.pairs.jsonl" \
  --out "$WORK_DIR/skills.candidate.jsonl" \
  --prompts-out "$WORK_DIR/summarizer.prompts.jsonl"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-openai}"
else
  SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-template}"
fi
"$PYTHON" -m skill_distill.summarize \
  --skills "$WORK_DIR/skills.candidate.jsonl" \
  --pairs "$WORK_DIR/train.pairs.jsonl" \
  --backend "$SUMMARIZER_BACKEND" \
  --out "$WORK_DIR/skills.natural-language.jsonl" \
  --book "$WORK_DIR/skill-book.pre-validation.txt"
"$PYTHON" -m skill_distill.validate \
  --skills "$WORK_DIR/skills.natural-language.jsonl" \
  --events "$WORK_DIR/train.events.jsonl" \
  --forks "$WORK_DIR/train.forks.jsonl" \
  --validation-events "$WORK_DIR/validation.events.jsonl" \
  --out "$LIBRARY_DIR/skills.v1.jsonl" \
  --report "$LIBRARY_DIR/skills.v1.validation.json" \
  --book "$LIBRARY_DIR/skill-book.v1.txt"

echo "Skill library: $LIBRARY_DIR/skills.v1.jsonl"
