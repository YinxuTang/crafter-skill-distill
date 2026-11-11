#!/usr/bin/env python3
"""Reclassify discovery rules by evidence status and semantic knowledge kind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


MOVE_TO_DIRECTION = {
    "Move North": "north",
    "Move South": "south",
    "Move East": "east",
    "Move West": "west",
}


def condition_atoms(condition: Any) -> List[Dict[str, Any]]:
    if not isinstance(condition, dict):
        return []
    if "field" in condition and "op" in condition:
        return [condition]
    atoms: List[Dict[str, Any]] = []
    for key in ("all", "any"):
        for child in condition.get(key) or []:
            atoms.extend(condition_atoms(child))
    if "not" in condition:
        atoms.extend(condition_atoms(condition["not"]))
    return atoms


def _condition_requires(condition: Any, field: str, value: Any) -> bool:
    return any(
        atom.get("field") == field
        and atom.get("op") == "=="
        and atom.get("value") == value
        for atom in condition_atoms(condition)
    )


def _referenced_targets(condition: Any) -> Set[str]:
    targets = set()
    for atom in condition_atoms(condition):
        field = str(atom.get("field") or "")
        for prefix in ("dir_", "visible_", "dist_"):
            if field.startswith(prefix):
                targets.add(field[len(prefix) :])
    return targets


def _movement_directions(behavior: Dict[str, Any]) -> Set[str]:
    if behavior.get("type") == "action":
        direction = MOVE_TO_DIRECTION.get(behavior.get("action"))
        return {direction} if direction else set()
    if behavior.get("type") == "action_distribution":
        return {
            direction
            for action in behavior.get("actions") or []
            if (direction := MOVE_TO_DIRECTION.get(action))
        }
    return set()


def _follows_observed_direction(condition: Any, behavior: Dict[str, Any]) -> bool:
    directions = _movement_directions(behavior)
    if not directions:
        return False
    for atom in condition_atoms(condition):
        field = str(atom.get("field") or "")
        if not field.startswith("dir_"):
            continue
        value = str(atom.get("value") or "")
        if atom.get("op") == "==" and any(direction in value for direction in directions):
            return True
        if atom.get("op") == "contains" and value in directions:
            return True
    return False


def _evidence_status(row: Dict[str, Any]) -> str:
    original = str(row.get("label") or "")
    if original == "promising_undercovered":
        return "undercovered"
    if original.startswith("reject") or original == "reject":
        return "reject"
    return "keep"


def _knowledge_kind(row: Dict[str, Any]) -> tuple[str, str]:
    condition = row.get("condition") or {}
    behavior = row.get("behavior") or {}
    btype = behavior.get("type")
    fidelity = float(row.get("fidelity") or 0.0)
    lift_b3 = row.get("lift_vs_b3")
    lift_b3 = float(lift_b3) if lift_b3 is not None else None

    if btype == "next_achievement":
        return (
            "progression_pattern",
            "Observed future achievement pattern; requires a progression baseline before "
            "calling it teacher-specific policy.",
        )

    if btype == "next_interaction_target":
        target = behavior.get("target")
        if _condition_requires(condition, "front", target):
            return (
                "basic_affordance",
                "The predicted target is already the object at front; this is structurally trivial.",
            )

    if _follows_observed_direction(condition, behavior):
        targets = _referenced_targets(condition)
        if len(targets) <= 1:
            return (
                "basic_navigation",
                "The action follows the only referenced object's direction.",
            )
        return (
            "policy_skill",
            "Movement is conditioned on multiple referenced objects, so it may encode priority.",
        )

    if lift_b3 is not None and lift_b3 <= 0.10 and fidelity >= 0.65:
        return (
            "basic_mechanics",
            "High fidelity but no material lift over the mechanics baseline.",
        )

    if btype == "next_interaction_target":
        return (
            "policy_skill",
            "Target choice is not a direct restatement of the object at front.",
        )

    if lift_b3 is not None and lift_b3 >= 0.10 and fidelity >= 0.55:
        return (
            "policy_skill",
            "Adequate fidelity with material lift over the mechanics baseline.",
        )

    return (
        "uncertain",
        "The rule is neither clearly basic nor sufficiently novel over mechanics.",
    )


def reclassify(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for row in rows:
        status = _evidence_status(row)
        kind, reason = _knowledge_kind(row)
        results.append(
            {
                **row,
                "original_label": row.get("label"),
                "status": status,
                "knowledge_kind": kind,
                "knowledge_reason": reason,
                "label": f"{status}_{kind}",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.classified).read_text(encoding="utf-8"))
    results = reclassify(payload.get("results") or [])
    status_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    for row in results:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if row["status"] != "reject":
            kind_counts[row["knowledge_kind"]] = kind_counts.get(row["knowledge_kind"], 0) + 1
    output = {
        "source": args.classified,
        "n": len(results),
        "status_counts": status_counts,
        "retained_kind_counts": kind_counts,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("n", "status_counts", "retained_kind_counts")}, indent=2))


if __name__ == "__main__":
    main()
