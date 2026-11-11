"""B3 mechanics baseline compiled from Crafter data.yaml.

This is an automatic game-rule policy, not teacher knowledge.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from skill_distill.crafter import ACTIONS


MAKE_ACTION = {
    "wood_pickaxe": "Make Wood Pickaxe",
    "stone_pickaxe": "Make Stone Pickaxe",
    "iron_pickaxe": "Make Iron Pickaxe",
    "wood_sword": "Make Wood Sword",
    "stone_sword": "Make Stone Sword",
    "iron_sword": "Make Iron Sword",
}

PLACE_ACTION = {
    "stone": "Place Stone",
    "table": "Place Table",
    "furnace": "Place Furnace",
    "plant": "Place Plant",
}


@lru_cache(maxsize=4)
def load_data_yaml(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def default_data_yaml() -> Path:
    # Prefer the project env copy; fall back to installed package.
    candidates = [
        Path("/home/tiger/rl2lang/envs/ad/lib/python3.11/site-packages/crafter/data.yaml"),
        Path.home() / ".local/lib/python3.11/site-packages/crafter/data.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    import crafter

    return Path(crafter.__file__).resolve().parent / "data.yaml"


def _inv(row: Dict[str, Any], item: str) -> int:
    key = f"inv_{item}"
    if key in row and row[key] is not None:
        return int(row[key])
    # vitals are bare names
    if item in row and row[item] is not None:
        return int(row[item])
    return 0


def _has(row: Dict[str, Any], item: str) -> bool:
    return _inv(row, item) > 0


def _nearby(row: Dict[str, Any], names) -> bool:
    for name in names:
        visible = row.get(f"visible_{name}")
        dist = row.get(f"dist_{name}")
        if visible and dist is not None and int(dist) <= 1:
            return True
        if row.get("front") == name:
            return True
    return False


def _can_make(row: Dict[str, Any], item: str, spec: Dict[str, Any]) -> bool:
    uses = spec.get("uses") or {}
    for need, amount in uses.items():
        if _inv(row, need) < int(amount):
            return False
    nearby = spec.get("nearby") or []
    if nearby and not _nearby(row, nearby):
        return False
    return True


def _can_place(row: Dict[str, Any], item: str, spec: Dict[str, Any]) -> bool:
    uses = spec.get("uses") or {}
    for need, amount in uses.items():
        if _inv(row, need) < int(amount):
            return False
    return True


def _can_collect_front(row: Dict[str, Any], collect: Dict[str, Any]) -> bool:
    front = row.get("front")
    if front not in collect:
        return False
    require = collect[front].get("require") or {}
    for need, amount in require.items():
        if _inv(row, need) < int(amount):
            return False
    return True


def predict_action(row: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> str:
    """Priority mechanics policy over a Decision DB row."""
    data = data or load_data_yaml(str(default_data_yaml()))
    collect = data.get("collect") or {}
    make = data.get("make") or {}
    place = data.get("place") or {}

    # Survive / recover from explicit vitals thresholds in the game loop sense.
    if int(row.get("energy", 9) or 9) <= 1:
        return "Sleep"
    if int(row.get("drink", 9) or 9) <= 1 and row.get("front") == "water":
        return "Do"
    if int(row.get("food", 9) or 9) <= 1 and row.get("front") in {"cow", "plant"}:
        return "Do"

    # Craft progression if locally feasible.
    craft_order = [
        "wood_pickaxe",
        "wood_sword",
        "stone_pickaxe",
        "stone_sword",
        "iron_pickaxe",
        "iron_sword",
    ]
    for item in craft_order:
        if item not in make:
            continue
        if _has(row, item):
            continue
        if _can_make(row, item, make[item]):
            return MAKE_ACTION[item]

    # Place table/furnace if needed for upcoming crafts.
    if not row.get("visible_table") and _can_place(row, "table", place.get("table", {})):
        if _inv(row, "wood") >= 2 and not _has(row, "wood_pickaxe"):
            return "Place Table"
    if (
        row.get("visible_table")
        and not row.get("visible_furnace")
        and _can_place(row, "furnace", place.get("furnace", {}))
    ):
        if _inv(row, "stone") >= 4 and _has(row, "coal") and _has(row, "iron"):
            return "Place Furnace"

    # Collect whatever is in front if requirements are met.
    if _can_collect_front(row, collect):
        return "Do"

    # Face-driven collectibles: if front is nothing useful, move toward nearest visible resource.
    for resource in ("tree", "stone", "water", "coal", "iron", "diamond", "cow"):
        if row.get(f"visible_{resource}") and row.get(f"dist_{resource}") is not None:
            direction = row.get(f"dir_{resource}") or ""
            if "north" in direction:
                return "Move North"
            if "south" in direction:
                return "Move South"
            if "west" in direction:
                return "Move West"
            if "east" in direction:
                return "Move East"

    return "Noop"


def predict_action_id(row: Dict[str, Any], data: Optional[Dict[str, Any]] = None) -> int:
    action = predict_action(row, data=data)
    return ACTIONS.index(action)
