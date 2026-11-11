"""Extract revision counterexample packs for hypothesis refine."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from skill_distill.hypothesis.dsl import (
    evaluate_behavior,
    evaluate_condition,
    select_independent_behavior_rows,
)


CONTEXT_FIELDS = [
    "action",
    "front",
    "health",
    "food",
    "drink",
    "energy",
    "is_night",
    "sleeping",
    "next_interaction_target",
    "next_interaction_type",
    "next_achievement",
    "steps_to_achievement",
]


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return True


def _state_snapshot(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = {key: row.get(key) for key in CONTEXT_FIELDS if _is_present(row.get(key))}
    snapshot["inventory"] = {
        key[len("inv_") :]: int(value)
        for key, value in row.items()
        if key.startswith("inv_") and _is_present(value) and float(value) > 0
    }
    snapshot["achievements"] = sorted(
        key[len("ach_") :]
        for key, value in row.items()
        if key.startswith("ach_") and _is_present(value) and bool(value)
    )
    nearby = {}
    for key, value in row.items():
        if not key.startswith("visible_") or not _is_present(value) or not bool(value):
            continue
        target = key[len("visible_") :]
        nearby[target] = {
            "distance": row.get(f"dist_{target}"),
            "direction": row.get(f"dir_{target}"),
        }
    snapshot["nearby"] = nearby
    snapshot["actual_behavior_outcome"] = result.get("actual", row.get("action"))
    snapshot["why_counterexample"] = "condition_matched_but_behavior_missed"
    return snapshot


def extract_counterexamples(
    hypothesis: Dict[str, Any],
    frame: pd.DataFrame,
    split: str = "revision",
    limit: int = 20,
) -> Dict[str, Any]:
    work = frame[frame["split"] == split]
    misses: List[Dict[str, Any]] = []
    hits = 0
    censored = 0
    support_rows: List[Dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        if not evaluate_condition(hypothesis["condition"], row):
            continue
        result = evaluate_behavior(hypothesis["behavior"], row)
        if not result.get("eligible", True):
            censored += 1
            continue
        support_rows.append(row)
    support_rows = select_independent_behavior_rows(hypothesis["behavior"], support_rows)
    support = len(support_rows)
    for row in support_rows:
        result = evaluate_behavior(hypothesis["behavior"], row)
        if result["hit"]:
            hits += 1
            continue
        misses.append(_state_snapshot(row, result))

    outcome_hist = Counter(item.get("actual_behavior_outcome") for item in misses)
    # Diversify by actual outcome before truncating.
    selected: List[Dict[str, Any]] = []
    by_outcome: Dict[Any, List[Dict[str, Any]]] = {}
    for item in misses:
        by_outcome.setdefault(item.get("actual_behavior_outcome"), []).append(item)
    while len(selected) < limit and by_outcome:
        empty = []
        for outcome, bucket in by_outcome.items():
            if not bucket:
                empty.append(outcome)
                continue
            selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
        for outcome in empty:
            by_outcome.pop(outcome, None)

    fidelity = (hits / support) if support else None
    return {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "split": split,
        "support": support,
        "hits": hits,
        "misses": len(misses),
        "censored": censored,
        "fidelity": fidelity,
        "miss_outcome_histogram": dict(outcome_hist),
        "examples": selected,
    }


def build_counterexample_feedback(
    hypotheses: Sequence[Dict[str, Any]],
    frame: pd.DataFrame,
    split: str = "revision",
    limit_per_hyp: int = 15,
) -> List[Dict[str, Any]]:
    packs = []
    for hypo in hypotheses:
        if not hypo.get("condition") or not hypo.get("behavior"):
            continue
        if hypo.get("error"):
            packs.append(
                {
                    "hypothesis_id": hypo.get("hypothesis_id"),
                    "error": hypo.get("error"),
                    "examples": [],
                }
            )
            continue
        packs.append(
            extract_counterexamples(
                {
                    "hypothesis_id": hypo.get("hypothesis_id"),
                    "condition": hypo["condition"],
                    "behavior": hypo["behavior"],
                },
                frame,
                split=split,
                limit=limit_per_hyp,
            )
        )
    return packs
