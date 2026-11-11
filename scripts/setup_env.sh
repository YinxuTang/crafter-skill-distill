#!/bin/bash
# Create an in-repo virtualenv so nothing outside this folder is required.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  echo "Creating $VENV"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

# Torch: prefer CUDA wheel when requested
if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    pip install "torch>=2.0" --index-url "$TORCH_INDEX_URL"
  else
    pip install "torch>=2.0"
  fi
fi

pip install -r "$ROOT/requirements.txt"
pip install -e "$ROOT/third_party/Achievement-Distillation"

echo
echo "OK: activate with:"
echo "  source $ROOT/.venv/bin/activate"
echo "  export PYTHONPATH=$ROOT:$ROOT/third_party/Achievement-Distillation"
