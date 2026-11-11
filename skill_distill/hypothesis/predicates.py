"""Allowed fields for the MVP Hypothesis DSL.

Fields are derived from Decision DB columns. This is a vocabulary for executable
conditions, not a hand-written skill library. LLM-invented concepts must later
compile into these fields (or registered derived concepts).
"""

from __future__ import annotations

from typing import Dict, Set

from skill_distill.crafter import ACTIONS


# Identity / leakage / privileged fields that must never appear in conditions.
FORBIDDEN_FIELDS: Set[str] = {
    "split",
    "episode_id",
    "episode_idx",
    "seed",
    "step",
    "obs_hash",
    "action",
    "action_id",
    "argmax_action",
    "argmax_action_id",
    "selected_is_argmax",
    "sampling",
    "reward",
    "done",
    "entropy",
    "value",
    "log_prob",
    "memory_norm",
    "unlocked",
    "inventory_delta",
    "has_inventory_delta",
    "has_unlock",
    "text_long",
    "text_short",
    "player_pos_x",
    "player_pos_y",
    "next_interaction_type",
    "next_interaction_target",
    "next_interaction_success",
    "steps_to_interaction",
    "next_achievement",
    "steps_to_achievement",
    "achievement_horizon",
    "achievement_horizon_complete",
}


OPS = {"==", "!=", "<", "<=", ">", ">=", "contains"}


def allowed_fields_from_columns(columns) -> Set[str]:
    return {str(col) for col in columns if str(col) not in FORBIDDEN_FIELDS}


def field_kind(field: str) -> str:
    if field in {"front", "dir_tree", "dir_stone", "dir_water", "dir_coal", "dir_iron", "dir_diamond", "dir_cow", "dir_zombie", "dir_skeleton", "dir_plant", "dir_table", "dir_furnace"}:
        return "categorical"
    if field.startswith("visible_") or field.startswith("has_") or field.startswith("ach_") or field in {
        "is_night",
        "sleeping",
        "dead",
    }:
        return "boolean"
    return "numeric"


def action_names() -> Set[str]:
    return set(ACTIONS)


def schema_summary(columns) -> Dict[str, object]:
    allowed = sorted(allowed_fields_from_columns(columns))
    return {
        "n_allowed_fields": len(allowed),
        "allowed_fields_sample": allowed[:40],
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "ops": sorted(OPS),
        "actions": ACTIONS,
        "behavior_types": [
            "action",
            "action_distribution",
            "next_interaction_target",
            "next_achievement",
        ],
    }
