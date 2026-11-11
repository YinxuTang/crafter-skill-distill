#!/usr/bin/env python3
"""Run B0/B3 baselines and optional demo hypotheses on Decision DB / critical bank."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from skill_distill.baselines.mechanics import default_data_yaml, load_data_yaml, predict_action as mech_predict
from skill_distill.baselines.random_legal import predict_action as random_predict
from skill_distill.hypothesis.evaluate import DEFAULT_DEMO_HYPOTHESES, evaluate_many
from skill_distill.hypothesis.predicates import schema_summary


def agreement(frame: pd.DataFrame, preds: List[str]) -> Dict[str, Any]:
    actual = frame["action"].tolist()
    hits = sum(int(a == p) for a, p in zip(actual, preds))
    n = len(actual)
    return {"n": n, "hits": hits, "agreement": (hits / n) if n else None}


def load_critical_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((int(row["seed"]), int(row["step"])))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument(
        "--critical",
        default="data/skill_distill/v3/critical_states/rollout_bank_wide/critical_correct.jsonl",
    )
    parser.add_argument("--split", default="exploration")
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/baselines/mvp_baseline_report.json",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.decisions, low_memory=False)
    work = frame[frame["split"] == args.split]
    if args.sample_rows and len(work) > args.sample_rows:
        work = work.sample(n=args.sample_rows, random_state=args.seed)

    data = load_data_yaml(str(default_data_yaml()))
    rng = random.Random(args.seed)
    rows = work.to_dict(orient="records")
    random_preds = [random_predict(row, rng=rng) for row in rows]
    mech_preds = [mech_predict(row, data=data) for row in rows]

    report: Dict[str, Any] = {
        "split": args.split,
        "n_eval_rows": len(work),
        "data_yaml": str(default_data_yaml()),
        "schema": schema_summary(frame.columns),
        "baselines": {
            "B0_random_legal": agreement(work, random_preds),
            "B3_mechanics_data_yaml": agreement(work, mech_preds),
        },
    }

    critical_keys = load_critical_keys(Path(args.critical))
    if critical_keys:
        # Match by seed/step on full split (not only sampled rows).
        full = frame[frame["split"] == args.split].copy()
        mask = [
            (int(seed), int(step)) in critical_keys
            for seed, step in zip(full["seed"].tolist(), full["step"].tolist())
        ]
        crit = full.loc[mask]
        # Critical bank came from live rollouts, so seed/step overlap with logged
        # Decision DB may be partial. Still report whatever matches.
        if len(crit):
            crit_rows = crit.to_dict(orient="records")
            report["critical_overlap_n"] = len(crit)
            report["baselines_on_critical_overlap"] = {
                "B0_random_legal": agreement(
                    crit, [random_predict(row, rng=rng) for row in crit_rows]
                ),
                "B3_mechanics_data_yaml": agreement(
                    crit, [mech_predict(row, data=data) for row in crit_rows]
                ),
            }
        else:
            report["critical_overlap_n"] = 0
            report["note"] = (
                "No seed/step overlap between live rollout critical bank and logged "
                "Decision DB; baseline agreement is reported on Decision DB sample only."
            )

    hypo_results = evaluate_many(DEFAULT_DEMO_HYPOTHESES, frame, split=args.split)
    report["demo_hypotheses"] = hypo_results

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
