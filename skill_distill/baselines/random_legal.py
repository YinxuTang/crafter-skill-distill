"""B0: sample a legal-looking action uniformly from the discrete action set.

Crafter does not expose a clean legality mask in Decision DB rows, so MVP uses
the full action alphabet as the legal set. This remains a weak reference bar.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Sequence

from skill_distill.crafter import ACTIONS


def predict_action(
    row: Dict[str, Any],
    rng: Optional[random.Random] = None,
    actions: Optional[Sequence[str]] = None,
) -> str:
    del row  # state-independent weak baseline
    pool = list(actions) if actions is not None else list(ACTIONS)
    generator = rng or random
    return generator.choice(pool)
