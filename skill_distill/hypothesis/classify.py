#!/usr/bin/env python3
"""Classify hypotheses vs B0/B3 on their Decision DB support sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from skill_distill.baselines.mechanics import default_data_yaml, load_data_yaml, predict_action as mech_predict
from skill_distill.baselines.random_legal import predict_action as random_predict
from skill_distill.hypothesis.dsl import (
    evaluate_behavior,
    evaluate_condition,
    select_independent_behavior_rows,
    validate_hypothesis,
)


def load_evaluated(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("candidates") or payload.get("hypotheses") or []


def behavior_hit(behavior: Dict[str, Any], row: Dict[str, Any], predicted_action: Optional[str] = None) -> bool:
    """Compare a baseline action prediction against hypothesis behavior."""
    btype = behavior.get("type")
    if btype == "action":
        return predicted_action == behavior.get("action")
    if btype == "action_distribution":
        return predicted_action in set(behavior.get("actions") or [])
    if btype == "next_interaction_target":
        # Baselines here are action policies; for target rules, approximate by whether
        # the baseline chooses Do while front/nearby matches target, else count as miss.
        target = behavior.get("target")
        if predicted_action != "Do":
            return False
        if row.get("front") == target:
            return True
        dist = row.get(f"dist_{target}")
        return bool(row.get(f"visible_{target}")) and dist is not None and int(dist) <= 1
    return False


def _is_temporal_behavior(behavior: Dict[str, Any]) -> bool:
    return behavior.get("type") in {"next_interaction_target", "next_achievement"}


def _temporal_base_rate(behavior: Dict[str, Any], rows: List[Dict[str, Any]]) -> float:
    eligible_rows = []
    for row in rows:
        result = evaluate_behavior(behavior, row)
        if result.get("eligible", True):
            eligible_rows.append(row)
    eligible_rows = select_independent_behavior_rows(behavior, eligible_rows)
    if not eligible_rows:
        return 0.0
    return (
        sum(1 for row in eligible_rows if evaluate_behavior(behavior, row)["hit"])
        / len(eligible_rows)
    )


def classify_one(
    hypothesis: Dict[str, Any],
    frame: pd.DataFrame,
    data: Dict[str, Any],
    rng: random.Random,
    split: str = "exploration",
) -> Dict[str, Any]:
    hypo = {
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "condition": hypothesis.get("condition") or hypothesis.get("metrics", {}).get("condition"),
        "behavior": hypothesis.get("behavior"),
        "rationale": hypothesis.get("rationale"),
    }
    # evaluated_round files already nest condition at top-level.
    if hypo["condition"] is None and "metrics" in hypothesis:
        hypo["condition"] = hypothesis.get("condition")
    if not hypo["condition"]:
        hypo["condition"] = hypothesis.get("condition")

    validate_hypothesis(hypo, columns=list(frame.columns))
    work = frame[frame["split"] == split]
    work_rows = work.to_dict(orient="records")
    condition_matches = 0
    censored = 0
    support_rows = []
    for row in work_rows:
        if not evaluate_condition(hypo["condition"], row):
            continue
        condition_matches += 1
        result = evaluate_behavior(hypo["behavior"], row)
        if not result.get("eligible", True):
            censored += 1
            continue
        support_rows.append(row)
    support_rows = select_independent_behavior_rows(hypo["behavior"], support_rows)
    support = len(support_rows)
    if support == 0:
        return {
            **hypo,
            "label": "reject_unsupported",
            "reason": "zero_support",
            "condition_matches": condition_matches,
            "censored": censored,
            "support": 0,
            "fidelity": None,
        }

    teacher_hits = sum(
        1 for row in support_rows if evaluate_behavior(hypo["behavior"], row)["hit"]
    )
    fidelity = teacher_hits / support

    if _is_temporal_behavior(hypo["behavior"]):
        # Action baselines cannot predict future targets. Compare temporal
        # hypotheses to unconditional target prevalence instead.
        b0_agree = _temporal_base_rate(hypo["behavior"], work_rows)
        b3_agree = b0_agree
        baseline_kind = "unconditional_target_prevalence"
    else:
        b0_preds = [random_predict(row, rng=rng) for row in support_rows]
        b3_preds = [mech_predict(row, data=data) for row in support_rows]
        b0_agree = sum(
            1
            for row, pred in zip(support_rows, b0_preds)
            if behavior_hit(hypo["behavior"], row, pred)
        ) / support
        b3_agree = sum(
            1
            for row, pred in zip(support_rows, b3_preds)
            if behavior_hit(hypo["behavior"], row, pred)
        ) / support
        baseline_kind = "B0_random_legal_and_B3_mechanics"

    lift_b0 = fidelity - b0_agree
    lift_b3 = fidelity - b3_agree
    seeds = {int(row["seed"]) for row in support_rows}

    failed_gates = []
    if support < 30:
        failed_gates.append("support_lt_30")
    if len(seeds) < 5:
        failed_gates.append("seeds_lt_5")
    if fidelity < 0.45:
        failed_gates.append("fidelity_lt_0.45")

    thin_evidence = support < 30 or len(seeds) < 5
    if thin_evidence and fidelity >= 0.55:
        label = "promising_undercovered"
        reason = "high_fidelity_but_thin_evidence"
    elif thin_evidence:
        label = "reject_undercovered"
        reason = "thin_evidence_without_high_fidelity"
    elif fidelity < 0.45:
        label = "reject_low_fidelity"
        reason = "adequate_evidence_but_low_fidelity"
    elif not _is_temporal_behavior(hypo["behavior"]) and lift_b3 <= 0.05 and fidelity >= 0.75:
        label = "mechanics_like"
        reason = "high_fidelity_but_close_to_mechanics"
    elif lift_b0 >= 0.15 and lift_b3 >= 0.10 and fidelity >= 0.55:
        label = "teacher_candidate"
        reason = "beats_random_and_mechanics"
    elif lift_b0 >= 0.10 and fidelity >= 0.60:
        label = "teacher_candidate"
        reason = "beats_random_enough"
    elif not _is_temporal_behavior(hypo["behavior"]) and fidelity >= 0.75:
        label = "mechanics_like"
        reason = "high_fidelity_default_mechanics_bucket"
    else:
        label = "reject_insufficient_lift"
        reason = "insufficient_lift"

    return {
        "hypothesis_id": hypo["hypothesis_id"],
        "condition": hypo["condition"],
        "behavior": hypo["behavior"],
        "rationale": hypo.get("rationale"),
        "label": label,
        "reason": reason,
        "failed_gates": failed_gates,
        "condition_matches": condition_matches,
        "censored": censored,
        "support": support,
        "independent_seeds": len(seeds),
        "fidelity": fidelity,
        "b0_agreement_on_support": b0_agree,
        "b3_agreement_on_support": b3_agree,
        "baseline_kind": baseline_kind,
        "lift_vs_b0": lift_b0,
        "lift_vs_b3": lift_b3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluated",
        default="data/skill_distill/v3/hypotheses/round1/evaluated_round1.json",
    )
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument("--split", default="exploration")
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/hypotheses/round1/classified.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluated = load_evaluated(Path(args.evaluated))
    # Prefer full hypothesis objects from evaluated file.
    frame = pd.read_csv(args.decisions, low_memory=False)
    data = load_data_yaml(str(default_data_yaml()))
    rng = random.Random(args.seed)

    results = []
    for item in evaluated:
        if item.get("error"):
            results.append(
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "label": "reject_invalid",
                    "reason": f"invalid_dsl: {item.get('error')}",
                }
            )
            continue
        results.append(classify_one(item, frame, data, rng, split=args.split))

    summary = {
        "n": len(results),
        "label_counts": {
            key: sum(1 for row in results if row.get("label") == key)
            for key in sorted({row.get("label") for row in results})
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
