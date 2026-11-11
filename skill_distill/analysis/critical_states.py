#!/usr/bin/env python3
"""Build action- and target-level critical state banks from estimates + Decision DB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_estimates(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_action_bank(
    estimates: List[Dict[str, Any]],
    percentile: float,
) -> Dict[str, Any]:
    valid = [row for row in estimates if "teacher_gap_td" in row]
    if not valid:
        return {"n": 0, "rows": []}
    abs_gaps = sorted(abs(row["teacher_gap_td"]) for row in valid)
    threshold = abs_gaps[min(len(abs_gaps) - 1, int(len(abs_gaps) * percentile / 100.0))]
    rows = sorted(
        [row for row in valid if abs(row["teacher_gap_td"]) >= threshold],
        key=lambda item: -abs(item["teacher_gap_td"]),
    )
    buckets = {
        "large_positive": [row for row in rows if row["teacher_gap_td"] > 0],
        "large_negative": [row for row in rows if row["teacher_gap_td"] < 0],
        "high_entropy": [
            row
            for row in rows
            if row.get("entropy") is not None and float(row["entropy"]) >= 1.5
        ],
    }
    return {
        "n_valid": len(valid),
        "threshold_abs_gap_td": threshold,
        "n_critical": len(rows),
        "bucket_sizes": {key: len(value) for key, value in buckets.items()},
        "rows": rows,
        "buckets": buckets,
    }


def build_target_bank(decisions: pd.DataFrame, min_support: int = 30) -> Dict[str, Any]:
    work = decisions[decisions["split"] == "exploration"].copy()
    work = work[work["next_interaction_target"].notna()]
    work = work[work["next_interaction_target"].astype(str) != ""]
    work["success"] = work["next_interaction_success"].astype(bool)

    # Observational: early steps that commit to a rare/hard upcoming interaction.
    grouped = (
        work.groupby(["next_interaction_type", "next_interaction_target"], dropna=False)
        .agg(
            n=("success", "size"),
            success_rate=("success", "mean"),
            mean_step=("step", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_value=("value", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= min_support].sort_values(
        ["success_rate", "n"], ascending=[True, False]
    )

    hard_targets = grouped.head(40).to_dict(orient="records")
    examples = []
    for record in hard_targets[:20]:
        subset = work[
            (work["next_interaction_type"] == record["next_interaction_type"])
            & (work["next_interaction_target"] == record["next_interaction_target"])
        ]
        # Prefer early decision points with high entropy / low success path.
        subset = subset.sort_values(["step", "entropy"], ascending=[True, False]).head(5)
        for _, row in subset.iterrows():
            examples.append(
                {
                    "episode_id": row["episode_id"],
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "action": row["action"],
                    "front": row.get("front"),
                    "entropy": float(row["entropy"]) if pd.notna(row["entropy"]) else None,
                    "value": float(row["value"]) if pd.notna(row["value"]) else None,
                    "next_interaction_type": row["next_interaction_type"],
                    "next_interaction_target": row["next_interaction_target"],
                    "next_interaction_success": bool(row["success"]),
                    "target_success_rate": float(record["success_rate"]),
                    "target_support": int(record["n"]),
                }
            )
    return {
        "method": "observational_hindsight_target",
        "min_support": min_support,
        "hard_targets": hard_targets,
        "examples": examples,
        "n_examples": len(examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--estimates",
        default="data/skill_distill/v3/critical_states/action_value_estimates.jsonl",
    )
    parser.add_argument("--decisions", default="data/skill_distill/v3/decision_db/decisions.csv")
    parser.add_argument("--out-dir", default="data/skill_distill/v3/critical_states")
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--min-support", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    estimates = load_estimates(Path(args.estimates)) if Path(args.estimates).exists() else []
    action_bank = build_action_bank(estimates, args.percentile)
    decisions = pd.read_csv(args.decisions, low_memory=False)
    target_bank = build_target_bank(decisions, min_support=args.min_support)

    with open(out_dir / "action_critical_states.jsonl", "w", encoding="utf-8") as handle:
        for row in action_bank["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    with open(out_dir / "target_critical_states.jsonl", "w", encoding="utf-8") as handle:
        for row in target_bank["examples"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    summary = {
        "action_bank": {
            key: value for key, value in action_bank.items() if key not in {"rows", "buckets"}
        },
        "target_bank": {
            "method": target_bank["method"],
            "min_support": target_bank["min_support"],
            "n_hard_targets": len(target_bank["hard_targets"]),
            "n_examples": target_bank["n_examples"],
            "hard_targets_top10": target_bank["hard_targets"][:10],
        },
    }
    (out_dir / "critical_banks_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "hard_targets.json").write_text(
        json.dumps(target_bank["hard_targets"], indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
