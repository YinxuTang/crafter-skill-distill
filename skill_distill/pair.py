#!/usr/bin/env python3
"""Match successful Crafter event windows with comparable failed attempts."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from skill_distill.crafter import read_jsonl, write_jsonl


def anchor_record(event: Dict[str, Any]) -> Dict[str, Any]:
    for step in event["steps"]:
        if step["step"] == event["anchor_step"]:
            return step
    raise ValueError(f"anchor missing for {event['event_id']}")


def state_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    a = anchor_record(left)["before"]
    b = anchor_record(right)["before"]
    inventory_keys = set(a["inventory"]) | set(b["inventory"])
    inventory = sum(
        min(abs(int(a["inventory"].get(key, 0)) - int(b["inventory"].get(key, 0))), 3)
        for key in inventory_keys
        if key not in ("health", "food", "drink", "energy")
    )
    vitals = sum(
        abs(int(a["vitals"][key]) - int(b["vitals"][key])) / 3.0
        for key in a["vitals"]
    )
    achievements = sum(
        bool(value) != bool(b["achievements"].get(key, 0))
        for key, value in a["achievements"].items()
    )
    front_a = anchor_record(left)["text_obs"]["long_term_context"].split("You face ")[-1]
    front_b = anchor_record(right)["text_obs"]["long_term_context"].split("You face ")[-1]
    facing_penalty = 0.0 if front_a == front_b else 3.0
    seed_penalty = 4.0 if left["seed"] == right["seed"] else 0.0
    return float(inventory + vitals + achievements + facing_penalty + seed_penalty)


def pair_events(events: List[Dict[str, Any]], max_negatives: int = 3) -> List[Dict[str, Any]]:
    successes = [event for event in events if event["success"]]
    failures_by_goal: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        if not event["success"]:
            failures_by_goal.setdefault(event["goal"], []).append(event)

    pairs = []
    for positive in successes:
        candidates = failures_by_goal.get(positive["goal"], [])
        ranked: List[Tuple[float, Dict[str, Any]]] = sorted(
            ((state_distance(positive, negative), negative) for negative in candidates),
            key=lambda item: item[0],
        )
        if not ranked:
            continue
        pairs.append(
            {
                "schema_version": 1,
                "pair_id": f"pair:{positive['event_id']}",
                "goal": positive["goal"],
                "positive": positive,
                "negatives": [
                    {"distance": distance, "event": event}
                    for distance, event in ranked[:max_negatives]
                ],
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-negatives", type=int, default=3)
    args = parser.parse_args()
    events = read_jsonl(args.events)
    pairs = pair_events(events, max_negatives=args.max_negatives)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, pairs)
    print(json.dumps({"pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
