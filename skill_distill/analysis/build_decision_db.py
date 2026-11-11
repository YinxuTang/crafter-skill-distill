#!/usr/bin/env python3
"""Flatten teacher trajectories into a queryable Decision Database."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from skill_distill.crafter import ACTIONS, VITALS, read_jsonl


SEE_RE = re.compile(
    r"^- (?P<name>[a-z_]+) (?P<dist>\d+) steps? to your (?P<dir>[a-z-]+)\.?$"
)
FACE_RE = re.compile(r"^You face (?P<front>.+) at your front\.$")
STATUS_RE = re.compile(r"^- (?P<name>[a-z_]+): (?P<value>\d+)(?:/9)?$")

INTERACTION_ACTIONS = {
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
}

DEFAULT_VISIBLE_OBJECTS = (
    "tree",
    "stone",
    "water",
    "coal",
    "iron",
    "diamond",
    "cow",
    "zombie",
    "skeleton",
    "table",
    "furnace",
    "plant",
    "grass",
    "sand",
    "path",
    "lava",
    "sapling",
    "arrow",
)

MAX_LOOKAHEAD = 64


def parse_long_term(text: str) -> Dict[str, Any]:
    is_night = None
    sleeping = False
    dead = False
    visible: Dict[str, Dict[str, Any]] = {}
    front = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("It is nighttime"):
            is_night = True
        elif line.startswith("It is daytime"):
            is_night = False
        elif line.startswith("You are sleeping"):
            sleeping = True
        elif line.startswith("You died"):
            dead = True
        match = SEE_RE.match(line)
        if match:
            name = match.group("name")
            visible[name] = {
                "distance": int(match.group("dist")),
                "direction": match.group("dir"),
            }
            continue
        match = FACE_RE.match(line)
        if match:
            front = match.group("front")
    return {
        "is_night": is_night,
        "text_sleeping": sleeping,
        "text_dead": dead,
        "front": front,
        "visible": visible,
    }


def parse_short_term(text: str) -> Dict[str, Any]:
    vitals: Dict[str, int] = {}
    inventory: Dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        match = STATUS_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        value = int(match.group("value"))
        if name in VITALS:
            vitals[name] = value
        else:
            inventory[name] = value
    return {"vitals": vitals, "inventory": inventory}


def interaction_label(action_name: str, front: Optional[str]) -> Tuple[str, str]:
    if action_name == "Do":
        return "Do", front or "unknown"
    if action_name == "Sleep":
        return "Sleep", "energy"
    if action_name.startswith("Place "):
        return "Place", action_name[len("Place ") :].lower().replace(" ", "_")
    if action_name.startswith("Make "):
        return "Make", action_name[len("Make ") :].lower().replace(" ", "_")
    return action_name, "none"


def attach_hindsight(rows: List[Dict[str, Any]], max_lookahead: int = MAX_LOOKAHEAD) -> None:
    n = len(rows)
    next_type = ["NONE"] * n
    next_target = ["NONE"] * n
    next_success = [None] * n
    steps_to = [None] * n
    for index, row in enumerate(rows):
        limit = min(n, index + max_lookahead + 1)
        found = None
        for future in range(index, limit):
            candidate = rows[future]
            if candidate["action"] not in INTERACTION_ACTIONS:
                continue
            itype, itarget = interaction_label(candidate["action"], candidate.get("front"))
            success = None
            if candidate["action"] == "Do":
                success = bool(candidate.get("inventory_delta")) or bool(candidate.get("unlocked"))
            elif candidate["action"].startswith(("Place ", "Make ")) or candidate["action"] == "Sleep":
                success = bool(candidate.get("unlocked")) or bool(candidate.get("inventory_delta"))
            found = (itype, itarget, success, future - index)
            break
        if found is None:
            continue
        itype, itarget, success, delta = found
        next_type[index] = itype
        next_target[index] = itarget
        next_success[index] = success
        steps_to[index] = delta
    for index, row in enumerate(rows):
        row["next_interaction_type"] = next_type[index]
        row["next_interaction_target"] = next_target[index]
        row["next_interaction_success"] = next_success[index]
        row["steps_to_interaction"] = steps_to[index]


def attach_future_achievements(
    rows: List[Dict[str, Any]],
    max_lookahead: int = MAX_LOOKAHEAD,
) -> None:
    """Attach the next observed achievement transition within a fixed horizon.

    These are hindsight labels for hypothesis behaviors, never condition fields.
    A row is censored when no event is observed and the recorded episode ends
    before the full horizon without a terminal transition.
    """
    n = len(rows)
    episode_terminated = bool(rows[-1].get("done")) if rows else False
    for index, row in enumerate(rows):
        limit = min(n, index + max_lookahead + 1)
        found_achievement = None
        found_delta = None
        for future in range(index, limit):
            unlocked = rows[future].get("unlocked") or []
            if unlocked:
                found_achievement = str(unlocked[0])
                found_delta = int(future - index)
                break

        full_horizon_observed = index + max_lookahead < n
        label_observed = found_achievement is not None
        row["next_achievement"] = found_achievement or "NONE"
        row["steps_to_achievement"] = found_delta
        row["achievement_horizon"] = int(max_lookahead)
        row["achievement_horizon_complete"] = bool(
            label_observed or full_horizon_observed or episode_terminated
        )


def attach_achievement_history(rows: List[Dict[str, Any]]) -> None:
    last_unlocked = None
    last_step = None
    for row in rows:
        unlocked = row.get("unlocked") or []
        if unlocked:
            last_unlocked = unlocked[-1]
            last_step = row["step"]
        row["last_unlocked_achievement"] = last_unlocked
        row["steps_since_achievement"] = (
            None if last_step is None else int(row["step"] - last_step)
        )


def flatten_step(
    record: Dict[str, Any],
    split: str,
    visible_objects: Sequence[str],
) -> Dict[str, Any]:
    long_text = record["text_obs"]["long_term_context"]
    short_text = record["text_obs"]["short_term_context"]
    parsed_long = parse_long_term(long_text)
    parsed_short = parse_short_term(short_text)
    before = record["before"]
    after = record["after"]
    teacher = record.get("teacher", {})
    inventory = before.get("inventory", {})
    vitals = before.get("vitals", {})
    # Prefer structured state; fall back to text parse.
    for key in VITALS:
        if key not in vitals and key in parsed_short["vitals"]:
            vitals = {**parsed_short["vitals"], **vitals}
    visible = parsed_long["visible"]

    row: Dict[str, Any] = {
        "split": split,
        "episode_id": record["episode_id"],
        "episode_idx": record["episode_idx"],
        "seed": record["seed"],
        "step": record["step"],
        "obs_hash": record.get("obs_hash"),
        "action": record["action_name"],
        "action_id": record["action_id"],
        "argmax_action": record.get("argmax_action_name"),
        "argmax_action_id": record.get("argmax_action_id"),
        "selected_is_argmax": teacher.get("selected_is_argmax"),
        "sampling": teacher.get("sampling"),
        "reward": record.get("reward"),
        "done": record.get("done"),
        "front": parsed_long["front"],
        "is_night": parsed_long["is_night"],
        "sleeping": before.get("sleeping", parsed_long["text_sleeping"]),
        "dead": before.get("dead", parsed_long["text_dead"]),
        "daylight": before.get("daylight"),
        "player_pos_x": before.get("player_pos", [None, None])[0],
        "player_pos_y": before.get("player_pos", [None, None])[1],
        "facing_x": before.get("player_facing", [None, None])[0],
        "facing_y": before.get("player_facing", [None, None])[1],
        "entropy": teacher.get("entropy"),
        "value": teacher.get("value"),
        "log_prob": teacher.get("log_prob"),
        "memory_norm": teacher.get("memory_norm"),
        "unlocked": list(record.get("unlocked") or []),
        "inventory_delta": dict(record.get("inventory_delta") or {}),
        "has_inventory_delta": bool(record.get("inventory_delta")),
        "has_unlock": bool(record.get("unlocked")),
        "text_long": long_text,
        "text_short": short_text,
    }
    for vital in VITALS:
        row[vital] = int(vitals.get(vital, inventory.get(vital, -1)))
    for item, count in inventory.items():
        if item in VITALS:
            continue
        row[f"inv_{item}"] = int(count)
        row[f"has_{item}"] = int(count) > 0
    for name in visible_objects:
        info = visible.get(name)
        row[f"visible_{name}"] = info is not None
        row[f"dist_{name}"] = None if info is None else int(info["distance"])
        row[f"dir_{name}"] = None if info is None else info["direction"]
    # Achievement flags before the action.
    for key, value in before.get("achievements", {}).items():
        row[f"ach_{key}"] = int(value) > 0
    return row


def find_episode_files(raw_root: Path, splits: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for split in splits:
        episode_dir = raw_root / split / "episodes"
        if not episode_dir.exists():
            continue
        for path in sorted(episode_dir.glob("*.jsonl")):
            files.append((split, path))
    return files


def build_rows(
    raw_root: Path,
    splits: Sequence[str],
    visible_objects: Sequence[str],
    max_lookahead: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = find_episode_files(raw_root, splits)
    for split, path in files:
        records = read_jsonl(str(path))
        episode_rows = [flatten_step(record, split, visible_objects) for record in records]
        attach_hindsight(episode_rows, max_lookahead=max_lookahead)
        attach_future_achievements(episode_rows, max_lookahead=max_lookahead)
        attach_achievement_history(episode_rows)
        rows.extend(episode_rows)
        print(
            f"[{split}] {path.name}: steps={len(episode_rows)}",
            flush=True,
        )
    return rows


def write_outputs(rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    jsonl_path = out_dir / "decisions.jsonl"
    csv_path = out_dir / "decisions.csv"
    parquet_path = out_dir / "decisions.parquet"

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    frame.to_csv(csv_path, index=False)

    parquet_ok = False
    parquet_error = None
    try:
        frame.to_parquet(parquet_path, index=False)
        parquet_ok = True
    except Exception as exc:  # noqa: BLE001 - optional dependency
        parquet_error = str(exc)

    summary = {
        "rows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()) if len(frame) else 0,
        "seeds": int(frame["seed"].nunique()) if len(frame) else 0,
        "splits": frame["split"].value_counts().to_dict() if len(frame) else {},
        "actions": frame["action"].value_counts().to_dict() if len(frame) else {},
        "next_interaction_type": (
            frame["next_interaction_type"].value_counts().to_dict() if len(frame) else {}
        ),
        "next_achievement": (
            frame["next_achievement"].value_counts().to_dict() if len(frame) else {}
        ),
        "achievement_horizon": (
            sorted(frame["achievement_horizon"].dropna().unique().tolist())
            if len(frame)
            else []
        ),
        "outputs": {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "parquet": str(parquet_path) if parquet_ok else None,
            "parquet_error": parquet_error,
        },
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/skill_distill/v3/raw",
        help="Directory containing split subfolders with episodes/",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["exploration", "revision", "final_test"],
    )
    parser.add_argument(
        "--out-dir",
        default="data/skill_distill/v3/decision_db",
    )
    parser.add_argument("--max-lookahead", type=int, default=MAX_LOOKAHEAD)
    args = parser.parse_args()

    rows = build_rows(
        raw_root=Path(args.raw_root).expanduser().resolve(),
        splits=args.splits,
        visible_objects=DEFAULT_VISIBLE_OBJECTS,
        max_lookahead=args.max_lookahead,
    )
    summary = write_outputs(rows, Path(args.out_dir).expanduser().resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
