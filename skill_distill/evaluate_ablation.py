#!/usr/bin/env python3
"""Summarize Crafter ablations and compute paired bootstrap intervals."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def episode_files(path: Path) -> List[Path]:
    return sorted(
        item
        for item in path.rglob("default_run_*.json")
        if "summary" not in item.name
    )


def load_run(path: str) -> List[Dict[str, Any]]:
    return [json.loads(item.read_text()) for item in episode_files(Path(path).expanduser())]


def official_score(episodes: List[Dict[str, Any]]) -> float:
    if not episodes:
        return 0.0
    keys = sorted(episodes[0].get("achievements", {}))
    rates = np.array(
        [
            100.0
            * np.mean([int(episode["achievements"].get(key, 0)) > 0 for episode in episodes])
            for key in keys
        ]
    )
    return float(np.exp(np.mean(np.log1p(rates))) - 1.0)


def summarize(name: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    progressions = np.array([float(item.get("progression", 0.0)) for item in episodes])
    total_actions = sum(sum(item.get("action_frequency", {}).values()) for item in episodes)
    do_actions = sum(item.get("action_frequency", {}).get("Do", 0) for item in episodes)
    achievements = {}
    if episodes:
        for key in episodes[0].get("achievements", {}):
            achievements[key] = 100.0 * float(
                np.mean([int(item["achievements"].get(key, 0)) > 0 for item in episodes])
            )
    return {
        "name": name,
        "n": len(episodes),
        "seeds": [int(item["seed"]) for item in episodes],
        "progression_pct": 100.0 * float(progressions.mean()) if len(progressions) else 0.0,
        "progression_se_pct": (
            100.0 * float(progressions.std(ddof=1) / np.sqrt(len(progressions)))
            if len(progressions) > 1
            else 0.0
        ),
        "official_score": official_score(episodes),
        "do_action_pct": 100.0 * do_actions / total_actions if total_actions else 0.0,
        "mean_input_tokens": float(np.mean([item.get("input_tokens", 0) for item in episodes]))
        if episodes
        else 0.0,
        "mean_output_tokens": float(np.mean([item.get("output_tokens", 0) for item in episodes]))
        if episodes
        else 0.0,
        "achievement_unlock_pct": achievements,
    }


def paired_bootstrap(
    baseline: List[Dict[str, Any]],
    variant: List[Dict[str, Any]],
    samples: int = 10000,
    seed: int = 20261021,
) -> Dict[str, Any]:
    baseline_by_seed = {int(item["seed"]): float(item["progression"]) for item in baseline}
    variant_by_seed = {int(item["seed"]): float(item["progression"]) for item in variant}
    seeds = sorted(set(baseline_by_seed) & set(variant_by_seed))
    if not seeds:
        return {"n": 0, "error": "no matched seeds"}
    deltas = np.array(
        [100.0 * (variant_by_seed[value] - baseline_by_seed[value]) for value in seeds]
    )
    rng = np.random.RandomState(seed)
    draws = rng.choice(deltas, size=(samples, len(deltas)), replace=True).mean(axis=1)
    return {
        "n": len(seeds),
        "mean_delta_progression_points": float(deltas.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "probability_positive": float(np.mean(draws > 0)),
    }


def markdown_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Crafter skill ablation",
        "",
        "| Run | n | Progression | Official score | Do share | Mean tokens (in/out) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            f"| {run['name']} | {run['n']} | {run['progression_pct']:.2f}% "
            f"± {run['progression_se_pct']:.2f} | {run['official_score']:.2f} | "
            f"{run['do_action_pct']:.2f}% | {run['mean_input_tokens']:.0f}/"
            f"{run['mean_output_tokens']:.0f} |"
        )
    lines.extend(["", "## Paired progression deltas", ""])
    for name, comparison in result["paired_vs_baseline"].items():
        if comparison.get("n", 0):
            lines.append(
                f"- {name}: {comparison['mean_delta_progression_points']:.2f} points, "
                f"95% CI [{comparison['ci95'][0]:.2f}, {comparison['ci95'][1]:.2f}], "
                f"P(delta>0)={comparison['probability_positive']:.3f}."
            )
        else:
            lines.append(f"- {name}: {comparison.get('error', 'not available')}.")
    return "\n".join(lines) + "\n"


def parse_run(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    return tuple(value.split("=", 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    loaded = {name: load_run(path) for name, path in args.run}
    if args.baseline not in loaded:
        raise SystemExit(f"baseline {args.baseline!r} not found in --run")
    result = {
        "baseline": args.baseline,
        "runs": [summarize(name, episodes) for name, episodes in loaded.items()],
        "paired_vs_baseline": {
            name: paired_bootstrap(
                loaded[args.baseline], episodes, samples=args.bootstrap_samples
            )
            for name, episodes in loaded.items()
            if name != args.baseline
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
