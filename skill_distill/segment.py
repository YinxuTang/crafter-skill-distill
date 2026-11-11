#!/usr/bin/env python3
"""Segment Crafter trajectories into grounded success and failure event windows."""

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from skill_distill.crafter import read_jsonl, write_jsonl


CRAFT_GOALS = {
    "Place Table": "place_table",
    "Place Furnace": "place_furnace",
    "Place Plant": "place_plant",
    "Place Stone": "place_stone",
    "Make Wood Pickaxe": "make_wood_pickaxe",
    "Make Stone Pickaxe": "make_stone_pickaxe",
    "Make Iron Pickaxe": "make_iron_pickaxe",
    "Make Wood Sword": "make_wood_sword",
    "Make Stone Sword": "make_stone_sword",
    "Make Iron Sword": "make_iron_sword",
}
FRONT_GOALS = {
    "tree": "collect_wood",
    "stone": "collect_stone",
    "coal": "collect_coal",
    "iron": "collect_iron",
    "diamond": "collect_diamond",
    "water": "collect_drink",
    "cow": "eat_cow",
    "zombie": "defeat_zombie",
    "skeleton": "defeat_skeleton",
}


def _front(text: str) -> Optional[str]:
    marker = "You face "
    for line in text.splitlines():
        if line.startswith(marker) and line.endswith(" at your front."):
            return line[len(marker) : -len(" at your front.")]
    return None


def _event_at(record: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    events = []
    for achievement in record.get("unlocked", []):
        events.append(("achievement", achievement, True))

    action = record["action_name"]
    expected = CRAFT_GOALS.get(action)
    front = _front(record["text_obs"]["long_term_context"])
    if action == "Do" and front in FRONT_GOALS:
        expected = FRONT_GOALS[front]
    if expected and expected not in record.get("unlocked", []):
        changed = any(int(value) > 0 for value in record.get("inventory_delta", {}).values())
        if not changed:
            events.append(("failed_attempt", expected, False))

    before_vitals = record["before"]["vitals"]
    after_vitals = record["after"]["vitals"]
    for vital in ("health", "food", "drink", "energy"):
        if before_vitals[vital] > 3 and after_vitals[vital] <= 3:
            events.append(("vital_danger", f"recover_{vital}", False))
        if before_vitals[vital] <= 3 and after_vitals[vital] > before_vitals[vital]:
            events.append(("vital_recovery", f"recover_{vital}", True))
    if record["after"]["dead"] and not record["before"]["dead"]:
        events.append(("death", "survive", False))
    if record["after"]["sleeping"] and not record["before"]["sleeping"]:
        events.append(("sleep_start", "recover_energy", True))
    return events


def _evidence_step(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": record["step"],
        "obs_hash": record["obs_hash"],
        "action_id": record["action_id"],
        "action_name": record["action_name"],
        "reward": record["reward"],
        "before": record["before"],
        "after": record["after"],
        "text_obs": record["text_obs"],
        "unlocked": record.get("unlocked", []),
        "inventory_delta": record.get("inventory_delta", {}),
        "teacher": record.get("teacher", {}),
        "counterfactuals": record.get("counterfactuals", []),
    }


def segment_episode(records: List[Dict[str, Any]], pre: int = 32, post: int = 8) -> List[Dict[str, Any]]:
    if not records:
        return []
    windows = []
    for anchor, record in enumerate(records):
        for event_type, goal, success in _event_at(record):
            start = max(0, anchor - pre)
            stop = min(len(records), anchor + post + 1)
            episode_id = record["episode_id"]
            windows.append(
                {
                    "schema_version": 1,
                    "event_id": f"{episode_id}:{record['step']}:{event_type}:{goal}",
                    "episode_id": episode_id,
                    "seed": record["seed"],
                    "goal": goal,
                    "event_type": event_type,
                    "success": success,
                    "anchor_step": record["step"],
                    "step_start": records[start]["step"],
                    "step_stop": records[stop - 1]["step"] + 1,
                    "action_prefix": [item["action_id"] for item in records[:anchor]],
                    "steps": [_evidence_step(item) for item in records[start:stop]],
                }
            )
    return windows


def find_episode_files(inputs: Iterable[str]) -> List[str]:
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(str(item) for item in path.glob("episodes/*.jsonl"))
            files.extend(str(item) for item in path.glob("*.jsonl"))
        else:
            files.extend(glob.glob(value))
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pre", type=int, default=32)
    parser.add_argument("--post", type=int, default=8)
    args = parser.parse_args()

    events = []
    for path in find_episode_files(args.inputs):
        events.extend(segment_episode(read_jsonl(path), pre=args.pre, post=args.post))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, events)
    counts: Dict[str, int] = {}
    for event in events:
        key = f"{event['goal']}:{'success' if event['success'] else 'failure'}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"events": len(events), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
