#!/usr/bin/env python3
"""Evaluate kept skills on a holdout Decision DB split (e.g. revision)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from skill_distill.hypothesis.evaluate import evaluate_hypothesis_on_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills",
        default="data/skill_distill/v3/hypotheses/skill_draft_v1.json",
    )
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument("--split", default="revision")
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/hypotheses/holdout_revision.json",
    )
    args = parser.parse_args()

    skills = json.loads(Path(args.skills).read_text(encoding="utf-8"))["skills"]
    frame = pd.read_csv(args.decisions, low_memory=False)

    rows: List[Dict[str, Any]] = []
    for skill in skills:
        hypo = {
            "hypothesis_id": skill["skill_id"],
            "condition": skill["condition"],
            "behavior": skill["behavior"],
        }
        discovery = {
            "label": skill.get("label"),
            "fidelity": skill.get("fidelity"),
            "support": skill.get("support"),
            "independent_seeds": skill.get("independent_seeds"),
        }
        metrics = evaluate_hypothesis_on_frame(hypo, frame, split=args.split)
        holdout_fid = metrics.get("fidelity")
        disc_fid = discovery.get("fidelity")
        drop = None if holdout_fid is None or disc_fid is None else holdout_fid - disc_fid
        status = "pass"
        if metrics.get("support", 0) < 10 or metrics.get("independent_seeds", 0) < 3:
            status = "insufficient_holdout_support"
        elif holdout_fid is not None and holdout_fid < 0.45:
            status = "fail_fidelity"
        elif drop is not None and drop < -0.15:
            status = "fail_large_drop"
        rows.append(
            {
                "skill_id": skill["skill_id"],
                "label": skill.get("label"),
                "discovery": discovery,
                "holdout_split": args.split,
                "holdout": {
                    "support": metrics.get("support"),
                    "independent_seeds": metrics.get("independent_seeds"),
                    "fidelity": holdout_fid,
                    "support_action_histogram": metrics.get("support_action_histogram"),
                },
                "fidelity_drop": drop,
                "status": status,
            }
        )

    summary = {
        "split": args.split,
        "n_skills": len(rows),
        "status_counts": {
            key: sum(1 for row in rows if row["status"] == key)
            for key in sorted({row["status"] for row in rows})
        },
        "results": rows,
        "note": (
            "This is an informal holdout checkup, not formal Acceptance. "
            "Acceptance is measured later by the program harness on acceptance data."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
