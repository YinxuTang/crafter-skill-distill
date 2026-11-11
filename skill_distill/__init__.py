"""Crafter trajectory-to-skill distillation utilities."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AD_ROOT = PROJECT_ROOT / "third_party" / "Achievement-Distillation"
BALROG_ROOT = PROJECT_ROOT / "third_party" / "BALROG"

for _path in (str(AD_ROOT), str(BALROG_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

__all__ = ["AD_ROOT", "BALROG_ROOT", "PROJECT_ROOT"]
