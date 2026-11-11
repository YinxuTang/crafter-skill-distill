#!/usr/bin/env python3
"""Build compact, factual evidence packs without inserting policy knowledge."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from skill_distill.crafter import read_jsonl, write_jsonl


def _anchor(event: Dict[str, Any]) -> Dict[str, Any]:
    return next(step for step in event["steps"] if step["step"] == event["anchor_step"])


def _compact_step(step: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": step["step"],
        "observation": {
            "world": step["text_obs"]["long_term_context"],
            "status_inventory": step["text_obs"]["short_term_context"],
        },
        "action": step["action_name"],
        "reward": step["reward"],
        "inventory_delta": step.get("inventory_delta", {}),
        "achievement_delta": step.get("unlocked", []),
    }


def compact_event(event: Dict[str, Any], fork: Dict[str, Any] = None) -> Dict[str, Any]:
    anchor = _anchor(event)
    local_steps = [
        step
        for step in event["steps"]
        if event["anchor_step"] - 8 <= step["step"] <= event["anchor_step"] + 2
    ]
    result = {
        "evidence_id": event["event_id"],
        "outcome": "success" if event["success"] else "failure",
        "event_type": event["event_type"],
        "episode_id": event["episode_id"],
        "seed": event["seed"],
        "step_span": [event["step_start"], event["step_stop"]],
        "anchor_step": event["anchor_step"],
        "anchor_before": anchor["before"],
        "anchor_after": anchor["after"],
        "local_trace": [_compact_step(step) for step in local_steps],
    }
    if fork:
        result["counterfactual_branches"] = [
            {
                "first_action": branch["first_action_name"],
                "actions": branch.get("action_names", []),
                "score": branch["score"],
                "teacher_action": branch["is_teacher_action"],
            }
            for branch in fork["branches"]
        ]
    return result


def diverse_sample(events: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_seed[int(event["seed"])].append(event)
    selected = []
    while len(selected) < limit:
        advanced = False
        for seed in sorted(by_seed):
            if by_seed[seed] and len(selected) < limit:
                selected.append(by_seed[seed].pop(0))
                advanced = True
        if not advanced:
            break
    return selected


def build_packs(
    events: List[Dict[str, Any]],
    forks: List[Dict[str, Any]],
    max_positive: int = 6,
    max_negative: int = 6,
) -> List[Dict[str, Any]]:
    forks_by_event = {fork["event_id"]: fork for fork in forks}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[event["goal"]].append(event)
    packs = []
    for goal, group in sorted(grouped.items()):
        positives = diverse_sample([event for event in group if event["success"]], max_positive)
        negatives = diverse_sample([event for event in group if not event["success"]], max_negative)
        if not positives:
            continue
        selected = positives + negatives
        packs.append(
            {
                "schema_version": 2,
                "pack_id": f"crafter:{goal}",
                "observed_goal_label": goal,
                "selection": {
                    "positive_available": sum(event["success"] for event in group),
                    "negative_available": sum(not event["success"] for event in group),
                    "positive_included": len(positives),
                    "negative_included": len(negatives),
                    "seed_count_available": len({event["seed"] for event in group}),
                },
                "evidence": [
                    compact_event(event, forks_by_event.get(event["event_id"]))
                    for event in selected
                ],
                "instructions": {
                    "facts_only": True,
                    "external_game_knowledge_allowed": False,
                    "citation_unit": "evidence_id",
                },
            }
        )
    return packs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--forks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-positive", type=int, default=6)
    parser.add_argument("--max-negative", type=int, default=6)
    args = parser.parse_args()
    packs = build_packs(
        read_jsonl(args.events),
        read_jsonl(args.forks),
        max_positive=args.max_positive,
        max_negative=args.max_negative,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, packs)
    print(json.dumps({"packs": len(packs)}, indent=2))


if __name__ == "__main__":
    main()
