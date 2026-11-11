#!/usr/bin/env python3
"""Program-only acceptance gate for frozen hypotheses."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from skill_distill.baselines.mechanics import default_data_yaml, load_data_yaml
from skill_distill.hypothesis.classify import classify_one, load_evaluated


def acceptance_decision(classified: Dict[str, Any]) -> Dict[str, Any]:
    discovery_label = classified.get("label")
    if discovery_label == "teacher_candidate":
        decision = "accepted_teacher_skill"
    elif discovery_label == "mechanics_like":
        decision = "accepted_mechanics_like"
    elif discovery_label == "promising_undercovered":
        decision = "promising_undercovered"
    else:
        decision = "rejected_on_acceptance"
    return {
        **classified,
        "acceptance_classification": discovery_label,
        "label": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="acceptance")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.split != "acceptance":
        parser.error("the formal acceptance gate must run on split=acceptance")

    evaluated = load_evaluated(Path(args.evaluated))
    frame = pd.read_csv(args.decisions, low_memory=False)
    data = load_data_yaml(str(default_data_yaml()))
    rng = random.Random(args.seed)

    results = []
    for item in evaluated:
        if item.get("error"):
            results.append(
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "label": "rejected_on_acceptance",
                    "acceptance_classification": "reject_invalid",
                    "reason": f"invalid_dsl: {item.get('error')}",
                }
            )
            continue
        results.append(
            acceptance_decision(
                classify_one(item, frame, data, rng, split=args.split)
            )
        )

    summary = {
        "split": args.split,
        "n": len(results),
        "label_counts": {
            label: sum(1 for row in results if row.get("label") == label)
            for label in sorted({row.get("label") for row in results})
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
