#!/usr/bin/env python3
"""Compute Progression and official Crafter geometric score for a BALROG run dir."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skill_distill.evaluate_ablation import load_run, summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Directory containing default_run_*.json")
    parser.add_argument("--name", default="run")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()
    episodes = load_run(args.run)
    summary = summarize(args.name, episodes)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"n={summary['n']}")
        print(f"Progression: {summary['progression_pct']:.2f}% ± {summary['progression_se_pct']:.2f}")
        print(f"Official Score: {summary['official_score']:.2f}")


if __name__ == "__main__":
    main()
