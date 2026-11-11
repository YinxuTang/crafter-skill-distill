#!/usr/bin/env python3
"""Load teacher checkpoint onto GPU/CPU and print a short OK line."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Achievement-Distillation"))
import torch

ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "artifacts/teacher/agent-e250.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"loading {ckpt} on {device}")
obj = torch.load(str(ckpt), map_location=device, weights_only=False)
keys = list(obj.keys()) if isinstance(obj, dict) else [type(obj).__name__]
print(f"OK keys={keys[:8]} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
