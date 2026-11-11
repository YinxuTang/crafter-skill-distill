#!/usr/bin/env python3
"""Mine revision counterexamples for kept skills (makes revision useful in the loop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skill_distill.hypothesis.counterexamples import build_counterexample_feedback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skills",
        default="data/skill_distill/v3/hypotheses/skill_draft_v1.json",
    )
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/hypotheses/revision_counterexamples_v1.json",
    )
    parser.add_argument("--limit-per-hyp", type=int, default=15)
    args = parser.parse_args()

    skills = json.loads(Path(args.skills).read_text(encoding="utf-8"))["skills"]
    hypos = [
        {
            "hypothesis_id": skill["skill_id"],
            "condition": skill["condition"],
            "behavior": skill["behavior"],
        }
        for skill in skills
    ]
    frame = pd.read_csv(args.decisions, low_memory=False)
    packs = build_counterexample_feedback(
        hypos, frame, split="revision", limit_per_hyp=args.limit_per_hyp
    )
    summary = {
        "n_skills": len(packs),
        "packs": packs,
        "note": (
            "These revision counterexamples are for refine feedback only. "
            "They must not be used as acceptance promotion evidence."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(
        {
            "n_skills": len(packs),
            "per_skill": [
                {
                    "id": p.get("hypothesis_id"),
                    "support": p.get("support"),
                    "misses": p.get("misses"),
                    "fidelity": p.get("fidelity"),
                    "n_examples": len(p.get("examples") or []),
                }
                for p in packs
            ],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
