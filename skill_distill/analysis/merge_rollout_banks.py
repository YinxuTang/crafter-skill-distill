#!/usr/bin/env python3
"""Merge sharded rollout_critical_bank outputs and reclassify with a global threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from skill_distill.analysis.rollout_critical_bank import (
    bank_quality_report,
    classify_row,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True, help="Directory containing shard_*/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--gap-percentile", type=float, default=70.0)
    parser.add_argument("--min-abs-gap", type=float, default=0.05)
    parser.add_argument("--min-sign-consistency", type=float, default=0.8)
    parser.add_argument("--min-alt-stability", type=float, default=0.6)
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    rows: List[Dict[str, Any]] = []
    missing = []
    for shard_id in range(args.num_shards):
        path = shard_root / f"shard_{shard_id}" / "rollout_estimates.jsonl"
        if not path.exists():
            missing.append(str(path))
            continue
        rows.extend(load_jsonl(path))
    if missing:
        raise SystemExit(f"missing shard outputs: {missing}")

    # Deduplicate by (seed, step) keeping first.
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (int(row["seed"]), int(row["step"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped

    abs_gaps = np.array([abs(row["teacher_gap_rollout"]) for row in rows], dtype=np.float64)
    percentile_threshold = (
        float(np.percentile(abs_gaps, args.gap_percentile)) if len(abs_gaps) else args.min_abs_gap
    )
    abs_gap_threshold = max(args.min_abs_gap, percentile_threshold)

    for row in rows:
        row["abs_gap_threshold_used"] = abs_gap_threshold
        row["class"] = classify_row(
            row,
            abs_gap_threshold=abs_gap_threshold,
            min_sign_consistency=args.min_sign_consistency,
            min_alt_stability=args.min_alt_stability,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rollout_estimates.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    banks = {
        "critical_correct": [],
        "critical_negative": [],
        "ambiguous_large": [],
        "indifferent": [],
        "unstable": [],
    }
    for row in rows:
        banks[row["class"]].append(row)
    for name, items in banks.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in sorted(items, key=lambda item: -abs(item["teacher_gap_rollout"])):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    action_hit_counts: Dict[str, int] = {}
    for row in rows:
        action = row.get("action")
        action_hit_counts[action] = action_hit_counts.get(action, 0) + 1

    quality = bank_quality_report(rows)
    summary = {
        "n_states": len(rows),
        "n_shards_merged": args.num_shards,
        "sample_mode": "action_stratified_merged",
        "action_hit_counts": action_hit_counts,
        "n_seeds_evaluated": len({int(row["seed"]) for row in rows}),
        "abs_gap_threshold": abs_gap_threshold,
        "gap_percentile": args.gap_percentile,
        "min_abs_gap": args.min_abs_gap,
        "min_sign_consistency": args.min_sign_consistency,
        "min_alt_stability": args.min_alt_stability,
        "quality": quality,
        "judgment_rubric": {
            "critical_correct": (
                "gap>0, |gap|>=threshold, 95% CI excludes 0, "
                "sign_consistency>=min, best_alt_stability>=min"
            ),
            "critical_negative": "same reliability checks but gap<0 (bad-habit candidate)",
            "ambiguous_large": "|gap| large but CI crosses 0 — need more samples",
            "indifferent": "|gap| below threshold — action swap likely unimportant",
            "unstable": "large/signed but unstable alt identity or sign flips",
            "bank_pass_if": quality["pass_checks"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
