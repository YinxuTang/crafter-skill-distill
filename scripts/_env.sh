#!/bin/bash
# In-repo runtime only. Does not append inherited PYTHONPATH (avoids rl2lang / mlx leakage).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Drop inherited path pollution from workspace shells.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

if [[ -x "$ROOT/envs/ad/bin/python" ]]; then
  # Bundled virtualenv copy (large; local/dev). Use its site-packages explicitly
  # because bin/python may be a symlink to /usr/bin/python3.
  PY_BASE="/usr/bin/python3.11"
  if [[ ! -x "$PY_BASE" ]]; then PY_BASE="/usr/bin/python3"; fi
  SITE="$ROOT/envs/ad/lib/python3.11/site-packages"
  if [[ ! -d "$SITE" ]]; then
    # detect versioned site-packages
    SITE="$(echo "$ROOT"/envs/ad/lib/python*/site-packages | awk '{print $1}')"
  fi
  export VIRTUAL_ENV="$ROOT/envs/ad"
  export PATH="$ROOT/envs/ad/bin:$PATH"
  export PYTHONPATH="$SITE:$ROOT:$ROOT/third_party/Achievement-Distillation"
  PYTHON="$PY_BASE"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="$ROOT/.venv"
  export PATH="$ROOT/.venv/bin:$PATH"
  export PYTHONPATH="$ROOT:$ROOT/third_party/Achievement-Distillation"
  PYTHON="$ROOT/.venv/bin/python"
else
  echo "Missing in-repo Python environment." >&2
  echo "Create one with:  bash scripts/setup_env.sh" >&2
  echo "Or place a copied env at: envs/ad/" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
  export CUDA_VISIBLE_DEVICES=0
fi
if [[ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.261.03 ]]; then
  mkdir -p "$HOME/.local/nvidia-lib"
  ln -sfn /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.261.03 "$HOME/.local/nvidia-lib/libnvidia-ml.so.1" 2>/dev/null || true
  ln -sfn /usr/lib/x86_64-linux-gnu/libcuda.so.535.261.03 "$HOME/.local/nvidia-lib/libcuda.so.1" 2>/dev/null || true
  export LD_LIBRARY_PATH="$HOME/.local/nvidia-lib:${LD_LIBRARY_PATH:-}"
fi
export LD_LIBRARY_PATH="${HOME}/.local/nvidia-lib-fix:/usr/local/cuda-12.6/compat:${LD_LIBRARY_PATH:-}"
