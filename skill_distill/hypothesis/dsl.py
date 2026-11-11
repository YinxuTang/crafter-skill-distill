"""Validate and evaluate MVP Hypothesis DSL objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from skill_distill.crafter import ACTIONS
from skill_distill.hypothesis.predicates import FORBIDDEN_FIELDS, OPS, allowed_fields_from_columns


MAX_PREDICATES = 8
MAX_DEPTH = 3


class DSLError(ValueError):
    pass


def _is_boolish(value: Any) -> bool:
    return isinstance(value, (bool,)) or value in (0, 1, 0.0, 1.0)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1, 0.0, 1.0):
        return bool(value)
    return bool(value)


def _count_predicates(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    if "field" in node and "op" in node:
        return 1
    total = 0
    for key in ("all", "any"):
        if key in node:
            total += sum(_count_predicates(child) for child in node[key])
    if "not" in node:
        total += _count_predicates(node["not"])
    return total


def _depth(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    if "field" in node and "op" in node:
        return 1
    depths = []
    for key in ("all", "any"):
        if key in node:
            depths.extend(_depth(child) for child in node[key])
    if "not" in node:
        depths.append(_depth(node["not"]))
    return 1 + (max(depths) if depths else 0)


def _validate_atom(atom: Dict[str, Any], allowed_fields: Set[str]) -> None:
    if set(atom.keys()) - {"field", "op", "value"}:
        raise DSLError(f"atom has unsupported keys: {sorted(atom.keys())}")
    field = atom.get("field")
    op = atom.get("op")
    if not isinstance(field, str):
        raise DSLError("atom.field must be a string")
    if field in FORBIDDEN_FIELDS:
        raise DSLError(f"forbidden field in condition: {field}")
    if field not in allowed_fields:
        raise DSLError(f"unknown/disallowed field: {field}")
    if op not in OPS:
        raise DSLError(f"unsupported op: {op}")
    if "value" not in atom:
        raise DSLError("atom.value is required")


def validate_condition(condition: Any, allowed_fields: Set[str], depth: int = 1) -> None:
    if not isinstance(condition, dict):
        raise DSLError("condition must be an object")
    if depth > MAX_DEPTH:
        raise DSLError(f"condition nesting exceeds max depth {MAX_DEPTH}")
    if "field" in condition and "op" in condition:
        _validate_atom(condition, allowed_fields)
        return
    keys = set(condition.keys())
    if keys == {"all"}:
        children = condition["all"]
        if not isinstance(children, list) or not children:
            raise DSLError("all[] must be a non-empty list")
        for child in children:
            validate_condition(child, allowed_fields, depth=depth + 1)
        return
    if keys == {"any"}:
        children = condition["any"]
        if not isinstance(children, list) or not children:
            raise DSLError("any[] must be a non-empty list")
        for child in children:
            validate_condition(child, allowed_fields, depth=depth + 1)
        return
    if keys == {"not"}:
        validate_condition(condition["not"], allowed_fields, depth=depth + 1)
        return
    raise DSLError(f"invalid condition shape: keys={sorted(keys)}")


def validate_behavior(behavior: Any) -> None:
    if not isinstance(behavior, dict):
        raise DSLError("behavior must be an object")
    btype = behavior.get("type")
    if btype == "action":
        action = behavior.get("action")
        if action not in ACTIONS:
            raise DSLError(f"unknown action: {action}")
        return
    if btype == "action_distribution":
        actions = behavior.get("actions")
        if not isinstance(actions, list) or not actions:
            raise DSLError("action_distribution.actions must be non-empty")
        for action in actions:
            if action not in ACTIONS:
                raise DSLError(f"unknown action: {action}")
        return
    if btype == "next_interaction_target":
        target = behavior.get("target")
        if not isinstance(target, str) or not target:
            raise DSLError("next_interaction_target.target must be a non-empty string")
        return
    if btype == "next_achievement":
        target = behavior.get("target")
        if not isinstance(target, str) or not target or target == "NONE":
            raise DSLError("next_achievement.target must name a real achievement")
        return
    raise DSLError(f"unsupported behavior.type: {btype}")


def validate_hypothesis(
    hypothesis: Dict[str, Any],
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if not isinstance(hypothesis, dict):
        raise DSLError("hypothesis must be an object")
    required = {"hypothesis_id", "condition", "behavior"}
    missing = required - set(hypothesis.keys())
    if missing:
        raise DSLError(f"missing keys: {sorted(missing)}")
    allowed = allowed_fields_from_columns(columns or [])
    if not allowed and columns is None:
        # Allow structural validation without a frame by only checking forbidden fields later.
        allowed = set()
    if columns is not None:
        validate_condition(hypothesis["condition"], allowed)
    else:
        # Structural-only: use a permissive allowed set minus forbidden, based on referenced fields.
        referenced = _referenced_fields(hypothesis["condition"])
        bad = referenced & FORBIDDEN_FIELDS
        if bad:
            raise DSLError(f"forbidden fields: {sorted(bad)}")
        validate_condition(hypothesis["condition"], referenced)
    validate_behavior(hypothesis["behavior"])
    n_pred = _count_predicates(hypothesis["condition"])
    depth = _depth(hypothesis["condition"])
    if n_pred > MAX_PREDICATES:
        raise DSLError(f"too many predicates: {n_pred} > {MAX_PREDICATES}")
    return {
        "ok": True,
        "n_predicates": n_pred,
        "depth": depth,
        "behavior_type": hypothesis["behavior"]["type"],
    }


def _referenced_fields(node: Any) -> Set[str]:
    if not isinstance(node, dict):
        return set()
    if "field" in node:
        return {str(node["field"])}
    fields: Set[str] = set()
    for key in ("all", "any"):
        if key in node:
            for child in node[key]:
                fields |= _referenced_fields(child)
    if "not" in node:
        fields |= _referenced_fields(node["not"])
    return fields


def _match_atom(atom: Dict[str, Any], row: Dict[str, Any]) -> bool:
    field = atom["field"]
    op = atom["op"]
    expected = atom["value"]
    actual = row.get(field)
    if actual is None:
        return False
    if op == "contains":
        return expected in actual if isinstance(actual, (list, tuple, set, str)) else False
    if op == "==":
        if isinstance(expected, bool) or _is_boolish(expected):
            return _as_bool(actual) == _as_bool(expected)
        return actual == expected
    if op == "!=":
        if isinstance(expected, bool) or _is_boolish(expected):
            return _as_bool(actual) != _as_bool(expected)
        return actual != expected
    try:
        left = float(actual)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    return False


def evaluate_condition(condition: Dict[str, Any], row: Dict[str, Any]) -> bool:
    if "field" in condition and "op" in condition:
        return _match_atom(condition, row)
    if "all" in condition:
        return all(evaluate_condition(child, row) for child in condition["all"])
    if "any" in condition:
        return any(evaluate_condition(child, row) for child in condition["any"])
    if "not" in condition:
        return not evaluate_condition(condition["not"], row)
    raise DSLError(f"invalid condition at eval time: {condition}")


def select_independent_behavior_rows(
    behavior: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    temporal_min_gap: int = 10,
) -> List[Dict[str, Any]]:
    """Reduce overlapping hindsight windows for temporal behavior metrics."""
    if behavior.get("type") not in {"next_interaction_target", "next_achievement"}:
        return list(rows)
    selected: List[Dict[str, Any]] = []
    last_step_by_episode: Dict[Any, int] = {}
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("episode_id", row.get("seed", -1)),
            int(row.get("step", 0)),
        ),
    )
    for row in ordered:
        episode = row.get("episode_id", row.get("seed", -1))
        step = int(row.get("step", 0))
        last_step = last_step_by_episode.get(episode)
        if last_step is not None and step - last_step < temporal_min_gap:
            continue
        selected.append(row)
        last_step_by_episode[episode] = step
    return selected


def evaluate_behavior(behavior: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    btype = behavior["type"]
    if btype == "action":
        predicted = behavior["action"]
        hit = row.get("action") == predicted
        return {"type": btype, "predicted": predicted, "hit": hit, "eligible": True}
    if btype == "action_distribution":
        predicted = list(behavior["actions"])
        hit = row.get("action") in predicted
        return {"type": btype, "predicted": predicted, "hit": hit, "eligible": True}
    if btype == "next_interaction_target":
        predicted = behavior["target"]
        actual = row.get("next_interaction_target")
        hit = actual == predicted
        return {
            "type": btype,
            "predicted": predicted,
            "actual": actual,
            "hit": hit,
            "eligible": True,
        }
    if btype == "next_achievement":
        predicted = behavior["target"]
        actual = row.get("next_achievement")
        raw_complete = row.get("achievement_horizon_complete")
        eligible = raw_complete is True or raw_complete == 1 or str(raw_complete).lower() == "true"
        hit = eligible and actual == predicted
        return {
            "type": btype,
            "predicted": predicted,
            "actual": actual,
            "hit": hit,
            "eligible": eligible,
        }
    raise DSLError(f"unsupported behavior type: {btype}")
