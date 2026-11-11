"""Evaluate hypotheses against Decision DB rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from skill_distill.hypothesis.dsl import (
    DSLError,
    evaluate_behavior,
    evaluate_condition,
    select_independent_behavior_rows,
    validate_hypothesis,
)


def evaluate_hypothesis_on_frame(
    hypothesis: Dict[str, Any],
    frame: pd.DataFrame,
    split: Optional[str] = "exploration",
) -> Dict[str, Any]:
    work = frame if split is None else frame[frame["split"] == split].copy()
    validation = validate_hypothesis(hypothesis, columns=list(work.columns))
    support_rows = []
    condition_matches = 0
    censored = 0
    for record in work.to_dict(orient="records"):
        if not evaluate_condition(hypothesis["condition"], record):
            continue
        condition_matches += 1
        result = evaluate_behavior(hypothesis["behavior"], record)
        if not result.get("eligible", True):
            censored += 1
            continue
        support_rows.append(record)
    support_rows = select_independent_behavior_rows(hypothesis["behavior"], support_rows)
    hits = sum(
        1 for row in support_rows if evaluate_behavior(hypothesis["behavior"], row)["hit"]
    )
    support = len(support_rows)
    seeds = {int(row["seed"]) for row in support_rows}
    fidelity = (hits / support) if support else None
    action_hist = Counter(row.get("action") for row in support_rows)
    return {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "validation": validation,
        "split": split,
        "n_rows_scanned": int(len(work)),
        "condition_matches": condition_matches,
        "censored": censored,
        "support": support,
        "independent_seeds": len(seeds),
        "hits": hits,
        "fidelity": fidelity,
        "support_action_histogram": dict(action_hist),
        "behavior": hypothesis.get("behavior"),
        "condition": hypothesis.get("condition"),
    }


def load_hypotheses(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    return [payload]


DEFAULT_DEMO_HYPOTHESES: List[Dict[str, Any]] = [
    {
        "hypothesis_id": "demo_sleep_when_energy_low",
        "condition": {"field": "energy", "op": "<=", "value": 2},
        "behavior": {"type": "action", "action": "Sleep"},
        "rationale": "MVP demo: low energy -> Sleep",
    },
    {
        "hypothesis_id": "demo_do_when_front_tree",
        "condition": {
            "all": [
                {"field": "front", "op": "==", "value": "tree"},
                {"field": "visible_tree", "op": "==", "value": True},
            ]
        },
        "behavior": {"type": "action", "action": "Do"},
        "rationale": "MVP demo: facing tree -> Do (likely mechanics-like)",
    },
    {
        "hypothesis_id": "demo_place_table_with_wood",
        "condition": {
            "all": [
                {"field": "has_wood", "op": "==", "value": True},
                {"field": "inv_wood", "op": ">=", "value": 2},
                {"field": "has_wood_pickaxe", "op": "==", "value": False},
            ]
        },
        "behavior": {"type": "action", "action": "Place Table"},
        "rationale": "MVP demo: enough wood and no pickaxe yet -> Place Table",
    },
    {
        "hypothesis_id": "demo_forbidden_seed_should_fail",
        "condition": {"field": "seed", "op": "==", "value": 1},
        "behavior": {"type": "action", "action": "Do"},
        "rationale": "Negative control: forbidden field",
    },
]


def evaluate_many(
    hypotheses: Iterable[Dict[str, Any]],
    frame: pd.DataFrame,
    split: str = "exploration",
) -> List[Dict[str, Any]]:
    results = []
    for hypothesis in hypotheses:
        try:
            results.append(evaluate_hypothesis_on_frame(hypothesis, frame, split=split))
        except DSLError as exc:
            results.append(
                {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "error": str(exc),
                    "condition": hypothesis.get("condition"),
                    "behavior": hypothesis.get("behavior"),
                }
            )
    return results
