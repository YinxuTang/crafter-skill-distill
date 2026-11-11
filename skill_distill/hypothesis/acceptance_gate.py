#!/usr/bin/env python3
"""Formal acceptance gate: program-only promotion on acceptance split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from skill_distill.baselines.mechanics import default_data_yaml, load_data_yaml, predict_action as mech_predict
from skill_distill.baselines.random_legal import predict_action as random_predict
from skill_distill.hypothesis.classify import behavior_hit, classify_one
from skill_distill.hypothesis.evaluate import evaluate_hypothesis_on_frame


def gate_one(
    skill: Dict[str, Any],
    frame: pd.DataFrame,
    data: Dict[str, Any],
    rng: random.Random,
    min_support: int,
    min_seeds: int,
    min_fidelity: float,
    min_lift_b0: float,
) -> Dict[str, Any]:
    hypo = {
        "hypothesis_id": skill.get("skill_id") or skill.get("hypothesis_id"),
        "condition": skill["condition"],
        "behavior": skill["behavior"],
        "rationale": skill.get("rationale"),
    }
    discovery_label = skill.get("label")
    metrics = evaluate_hypothesis_on_frame(hypo, frame, split="acceptance")
    classified = classify_one(hypo, frame, data, rng, split="acceptance")

    support = metrics.get("support") or 0
    seeds = metrics.get("independent_seeds") or 0
    fidelity = metrics.get("fidelity")
    lift_b0 = classified.get("lift_vs_b0")
    lift_b3 = classified.get("lift_vs_b3")
    auto_label = classified.get("label")

    reasons = []
    if support < min_support:
        reasons.append(f"support<{min_support}")
    if seeds < min_seeds:
        reasons.append(f"seeds<{min_seeds}")
    if fidelity is None or fidelity < min_fidelity:
        reasons.append(f"fidelity<{min_fidelity}")
    if lift_b0 is None or lift_b0 < min_lift_b0:
        reasons.append(f"lift_b0<{min_lift_b0}")

    if not reasons:
        if auto_label == "mechanics_like" or discovery_label == "mechanics_like":
            status = "accepted_mechanics_like"
        elif auto_label == "teacher_candidate":
            status = "accepted_teacher_skill"
        else:
            status = "accepted_with_caution"
            reasons.append(f"auto_label={auto_label}")
    else:
        status = "rejected_on_acceptance"

    return {
        "skill_id": hypo["hypothesis_id"],
        "discovery_label": discovery_label,
        "acceptance_label": auto_label,
        "status": status,
        "reject_reasons": reasons,
        "acceptance_metrics": {
            "support": support,
            "independent_seeds": seeds,
            "fidelity": fidelity,
            "lift_vs_b0": lift_b0,
            "lift_vs_b3": lift_b3,
            "b0_agreement_on_support": classified.get("b0_agreement_on_support"),
            "b3_agreement_on_support": classified.get("b3_agreement_on_support"),
            "support_action_histogram": metrics.get("support_action_histogram"),
        },
        "condition": hypo["condition"],
        "behavior": hypo["behavior"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills",
        default="data/skill_distill/v3/hypotheses/skill_draft_v1.json",
    )
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/hypotheses/acceptance_gate_report.json",
    )
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-fidelity", type=float, default=0.50)
    parser.add_argument("--min-lift-b0", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(Path(args.skills).read_text(encoding="utf-8"))
    skills = payload.get("skills") or payload.get("results") or []
    frame = pd.read_csv(args.decisions, low_memory=False)
    if "acceptance" not in set(frame["split"].unique()):
        raise SystemExit(
            "Decision DB has no acceptance split. Collect acceptance trajectories "
            "and rebuild the DB first."
        )

    data = load_data_yaml(str(default_data_yaml()))
    rng = random.Random(args.seed)
    results = [
        gate_one(
            skill,
            frame,
            data,
            rng,
            min_support=args.min_support,
            min_seeds=args.min_seeds,
            min_fidelity=args.min_fidelity,
            min_lift_b0=args.min_lift_b0,
        )
        for skill in skills
    ]
    summary = {
        "n": len(results),
        "status_counts": {
            key: sum(1 for row in results if row["status"] == key)
            for key in sorted({row["status"] for row in results})
        },
        "thresholds": {
            "min_support": args.min_support,
            "min_seeds": args.min_seeds,
            "min_fidelity": args.min_fidelity,
            "min_lift_b0": args.min_lift_b0,
        },
        "results": results,
        "note": (
            "Acceptance is measured by the program harness only. LLM never sees "
            "acceptance rows and cannot self-promote."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
