#!/usr/bin/env python3
"""Greedy predicate ablation and exact deduplication for hypothesis rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from skill_distill.hypothesis.classify import load_evaluated
from skill_distill.hypothesis.dsl import (
    evaluate_behavior,
    evaluate_condition,
    select_independent_behavior_rows,
    validate_hypothesis,
)


def _score(
    condition: Dict[str, Any],
    behavior: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    matched = []
    censored = 0
    for row in rows:
        if not evaluate_condition(condition, row):
            continue
        result = evaluate_behavior(behavior, row)
        if not result.get("eligible", True):
            censored += 1
            continue
        matched.append(row)
    matched = select_independent_behavior_rows(behavior, matched)
    support = len(matched)
    hits = sum(1 for row in matched if evaluate_behavior(behavior, row)["hit"])
    return {
        "support": support,
        "independent_seeds": len({int(row["seed"]) for row in matched}),
        "hits": hits,
        "fidelity": hits / support if support else None,
        "censored": censored,
    }


def _acceptable_ablation(
    candidate_scores: Dict[str, Dict[str, Any]],
    original_scores: Dict[str, Dict[str, Any]],
    *,
    min_fidelity: float,
    max_fidelity_drop: float,
) -> bool:
    for split, original in original_scores.items():
        candidate = candidate_scores[split]
        original_fidelity = original.get("fidelity")
        candidate_fidelity = candidate.get("fidelity")
        if original_fidelity is None or candidate_fidelity is None:
            return False
        fidelity_floor = max(min_fidelity, original_fidelity - max_fidelity_drop)
        if candidate_fidelity < fidelity_floor:
            return False
        if candidate["support"] < original["support"]:
            return False
        if candidate["independent_seeds"] < original["independent_seeds"]:
            return False
    return True


def simplify_one(
    hypothesis: Dict[str, Any],
    rows_by_split: Dict[str, Sequence[Dict[str, Any]]],
    *,
    min_fidelity: float = 0.55,
    max_fidelity_drop: float = 0.03,
) -> Dict[str, Any]:
    condition = hypothesis.get("condition")
    behavior = hypothesis.get("behavior")
    if not condition or not behavior:
        return {**hypothesis, "simplification": {"status": "skipped_invalid"}}
    validate_hypothesis(hypothesis, columns=list(next(iter(rows_by_split.values()))[0].keys()))

    original_scores = {
        split: _score(condition, behavior, rows)
        for split, rows in rows_by_split.items()
    }
    current_condition = condition
    current_scores = original_scores
    removed: List[Dict[str, Any]] = []

    while isinstance(current_condition, dict) and set(current_condition) == {"all"}:
        children = list(current_condition["all"])
        if len(children) <= 1:
            break
        choices = []
        for index, predicate in enumerate(children):
            remaining = children[:index] + children[index + 1 :]
            trial_condition = remaining[0] if len(remaining) == 1 else {"all": remaining}
            trial_scores = {
                split: _score(trial_condition, behavior, rows)
                for split, rows in rows_by_split.items()
            }
            if not _acceptable_ablation(
                trial_scores,
                original_scores,
                min_fidelity=min_fidelity,
                max_fidelity_drop=max_fidelity_drop,
            ):
                continue
            support_gain = sum(
                trial_scores[split]["support"] - current_scores[split]["support"]
                for split in rows_by_split
            )
            mean_fidelity = sum(
                trial_scores[split]["fidelity"] for split in rows_by_split
            ) / len(rows_by_split)
            choices.append(
                (support_gain, mean_fidelity, index, predicate, trial_condition, trial_scores)
            )
        if not choices:
            break
        _, _, _, predicate, current_condition, current_scores = max(
            choices,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        removed.append(predicate)

    return {
        **hypothesis,
        "condition": current_condition,
        "simplification": {
            "status": "simplified" if removed else "unchanged",
            "removed_predicates": removed,
            "original_metrics": original_scores,
            "simplified_metrics": current_scores,
            "min_fidelity": min_fidelity,
            "max_total_fidelity_drop": max_fidelity_drop,
        },
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        normalized = {key: _canonical(item) for key, item in value.items()}
        if set(normalized) in ({"all"}, {"any"}):
            key = next(iter(normalized))
            normalized[key] = sorted(
                normalized[key],
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        return normalized
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _predicate_set(condition: Any) -> set[str] | None:
    if not isinstance(condition, dict):
        return None
    if "field" in condition and "op" in condition:
        return {json.dumps(_canonical(condition), sort_keys=True, separators=(",", ":"))}
    if set(condition) == {"all"}:
        predicates: set[str] = set()
        for child in condition["all"]:
            child_set = _predicate_set(child)
            if child_set is None:
                return None
            predicates |= child_set
        return predicates
    return None


def _metrics_rank(metrics: Dict[str, Dict[str, Any]]) -> Tuple[float, int]:
    fidelities = [
        item["fidelity"]
        for item in metrics.values()
        if item.get("fidelity") is not None
    ]
    support = sum(int(item.get("support") or 0) for item in metrics.values())
    return (min(fidelities) if fidelities else -1.0, support)


def _broader_rule_subsumes(
    broader: Dict[str, Any],
    narrower: Dict[str, Any],
    *,
    max_specificity_gain: float = 0.03,
) -> bool:
    if _canonical(broader.get("behavior")) != _canonical(narrower.get("behavior")):
        return False
    broad_set = _predicate_set(broader.get("condition"))
    narrow_set = _predicate_set(narrower.get("condition"))
    if broad_set is None or narrow_set is None or not broad_set < narrow_set:
        return False
    broad_metrics = broader.get("simplification", {}).get("simplified_metrics", {})
    narrow_metrics = narrower.get("simplification", {}).get("simplified_metrics", {})
    if set(broad_metrics) != set(narrow_metrics):
        return False
    for split in broad_metrics:
        broad = broad_metrics[split]
        narrow = narrow_metrics[split]
        if broad.get("fidelity") is None or narrow.get("fidelity") is None:
            return False
        if broad["fidelity"] + max_specificity_gain < narrow["fidelity"]:
            return False
        if int(broad.get("support") or 0) < int(narrow.get("support") or 0):
            return False
    return True


def deduplicate(hypotheses: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_rule: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, List[str]] = {}
    for hypothesis in hypotheses:
        key = json.dumps(
            {
                "condition": _canonical(hypothesis.get("condition")),
                "behavior": _canonical(hypothesis.get("behavior")),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = by_rule.get(key)
        if existing is None:
            by_rule[key] = hypothesis
            aliases[key] = []
            continue
        aliases[key].append(str(hypothesis.get("hypothesis_id")))
        existing_metrics = existing.get("simplification", {}).get("simplified_metrics", {})
        new_metrics = hypothesis.get("simplification", {}).get("simplified_metrics", {})

        if _metrics_rank(new_metrics) > _metrics_rank(existing_metrics):
            aliases[key].append(str(existing.get("hypothesis_id")))
            by_rule[key] = hypothesis

    exact_results = []
    for key, hypothesis in by_rule.items():
        winner_id = str(hypothesis.get("hypothesis_id"))
        exact_results.append(
            {
                **hypothesis,
                "deduplication": {
                    "merged_hypothesis_ids": sorted(
                        {
                            alias
                            for alias in aliases[key]
                            if alias and alias != "None" and alias != winner_id
                        }
                    )
                },
            }
        )

    # Prefer a broader rule when it has the same behavior and the narrower
    # condition improves fidelity by no more than max_specificity_gain.
    ordered = sorted(
        exact_results,
        key=lambda item: (
            len(_predicate_set(item.get("condition")) or set()) or 10_000,
            -_metrics_rank(
                item.get("simplification", {}).get("simplified_metrics", {})
            )[1],
        ),
    )
    kept: List[Dict[str, Any]] = []
    for candidate in ordered:
        broader = next(
            (
                existing
                for existing in kept
                if _broader_rule_subsumes(existing, candidate)
            ),
            None,
        )
        if broader is None:
            kept.append(candidate)
            continue
        merged = broader["deduplication"]["merged_hypothesis_ids"]
        merged.append(str(candidate.get("hypothesis_id")))
        merged.extend(candidate["deduplication"]["merged_hypothesis_ids"])
        broader["deduplication"]["merged_hypothesis_ids"] = sorted(set(merged))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluated", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["exploration", "revision"])
    parser.add_argument("--min-fidelity", type=float, default=0.55)
    parser.add_argument("--max-fidelity-drop", type=float, default=0.03)
    args = parser.parse_args()

    frame = pd.read_csv(args.decisions, low_memory=False)
    rows_by_split = {
        split: frame[frame["split"] == split].to_dict(orient="records")
        for split in args.splits
    }
    hypotheses = load_evaluated(Path(args.evaluated))
    simplified = []
    for hypothesis in hypotheses:
        if hypothesis.get("error"):
            continue
        simplified.append(
            simplify_one(
                hypothesis,
                rows_by_split,
                min_fidelity=args.min_fidelity,
                max_fidelity_drop=args.max_fidelity_drop,
            )
        )
    deduped = deduplicate(simplified)
    payload = {
        "source": args.evaluated,
        "splits": args.splits,
        "n_input": len(hypotheses),
        "n_simplified": sum(
            item.get("simplification", {}).get("status") == "simplified"
            for item in simplified
        ),
        "n_output": len(deduped),
        "candidates": deduped,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
