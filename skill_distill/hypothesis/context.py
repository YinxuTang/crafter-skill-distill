"""Build data-conditioned context for the LLM Scientist (no data.yaml)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from skill_distill.crafter import ACTIONS
from skill_distill.hypothesis.predicates import FORBIDDEN_FIELDS, OPS, allowed_fields_from_columns


def _load_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _state_snapshot(row: Mapping[str, Any]) -> Dict[str, Any]:
    state = {
        key: row.get(key)
        for key in (
            "front",
            "health",
            "food",
            "drink",
            "energy",
            "daylight",
            "is_night",
            "sleeping",
        )
        if _is_present(row.get(key))
    }
    state["inventory"] = {
        key[len("inv_") :]: int(value)
        for key, value in row.items()
        if key.startswith("inv_") and _is_present(value) and float(value) > 0
    }
    state["achievements"] = sorted(
        key[len("ach_") :]
        for key, value in row.items()
        if key.startswith("ach_") and _is_present(value) and _as_bool(value)
    )
    nearby: Dict[str, Dict[str, Any]] = {}
    for key, value in row.items():
        if not key.startswith("visible_") or not _is_present(value) or not _as_bool(value):
            continue
        target = key[len("visible_") :]
        nearby[target] = {
            "distance": row.get(f"dist_{target}"),
            "direction": row.get(f"dir_{target}"),
        }
    state["nearby"] = nearby
    return state


def _stable_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _evidence_id(prefix: str, row: Mapping[str, Any]) -> str:
    identity = "|".join(
        str(row.get(key, ""))
        for key in ("episode_idx", "step", "action", "next_achievement")
    )
    digest = hashlib.sha1(f"{prefix}:{identity}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _cyclic_window(items: List[Any], start: int, count: int) -> List[Any]:
    if not items or count <= 0:
        return []
    count = min(count, len(items))
    return [items[(start + offset) % len(items)] for offset in range(count)]


def _rotation_exposure_count(
    pool_size: int,
    window_size: int,
    item_index: int,
    cycle: int,
) -> int:
    if pool_size <= 0 or window_size <= 0:
        return 0
    count = 0
    for prior_cycle in range(cycle + 1):
        indices = {
            (prior_cycle * window_size + offset) % pool_size
            for offset in range(min(window_size, pool_size))
        }
        count += int(item_index in indices)
    return count


def _slim_row(
    row: Dict[str, Any],
    state_lookup: Mapping[Tuple[int, int], Mapping[str, Any]] | None = None,
    selection: str = "rotating",
    exposure_count: int = 1,
) -> Dict[str, Any]:
    key = (int(row.get("episode_idx", -1)), int(row.get("step", -1)))
    decision_row = (state_lookup or {}).get(key, {})
    return {
        "evidence_id": _evidence_id("critical", row),
        "selection": selection,
        "exposure_count": exposure_count,
        "action": row.get("action"),
        "state": _state_snapshot(decision_row or row),
        "teacher_gap_rollout": row.get("teacher_gap_rollout"),
        "best_alt_action": row.get("best_alt_action"),
        "sign_consistency": row.get("sign_consistency"),
        "rollout_horizon": row.get("horizon"),
    }


def _stratified_examples(
    rows: List[Dict[str, Any]],
    *,
    per_action: int,
    max_total: int,
    state_lookup: Mapping[Tuple[int, int], Mapping[str, Any]] | None = None,
    evidence_cycle: int = 0,
    evidence_seed: int = 20261028,
    fixed_per_action: int = 0,
) -> List[Dict[str, Any]]:
    """Mixed pool: up to per_action examples from each action, capped at max_total.

    Avoids action-specialized propose batches while still covering rare actions.
    Rotates examples deterministically without replacement until each action
    bucket wraps. Stable full-split aggregates provide continuity across cycles.
    """
    by_action: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        action = str(row.get("action") or "unknown")
        by_action.setdefault(action, []).append(row)
    for action, items in by_action.items():
        items.sort(key=lambda item: -abs(float(item.get("teacher_gap_rollout") or 0.0)))

    # Round-robin so Do/Move cannot monopolize the prompt.
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for action, items in sorted(by_action.items()):
        fixed_count = min(fixed_per_action, per_action, len(items))
        bucket = [
            _slim_row(
                item,
                state_lookup=state_lookup,
                selection="fixed_anchor",
                exposure_count=evidence_cycle + 1,
            )
            for item in items[:fixed_count]
        ]
        rotating = list(items[fixed_count:])
        random.Random(_stable_seed(evidence_seed, f"critical:{action}")).shuffle(rotating)
        rotating_count = min(per_action - fixed_count, len(rotating))
        start = evidence_cycle * rotating_count
        selected_rotating = _cyclic_window(rotating, start, rotating_count)
        index_by_id = {id(item): index for index, item in enumerate(rotating)}
        bucket.extend(
            _slim_row(
                item,
                state_lookup=state_lookup,
                selection="rotating",
                exposure_count=_rotation_exposure_count(
                    len(rotating),
                    rotating_count,
                    index_by_id[id(item)],
                    evidence_cycle,
                ),
            )
            for item in selected_rotating
        )
        buckets[action] = bucket

    selected: List[Dict[str, Any]] = []
    index = 0
    while len(selected) < max_total and buckets:
        progress = False
        for action in list(buckets.keys()):
            items = buckets[action]
            if index < len(items):
                selected.append(items[index])
                progress = True
                if len(selected) >= max_total:
                    break
            else:
                del buckets[action]
        if not progress:
            break
        index += 1
    return selected


def summarize_critical_bank(
    bank_dir: Path,
    per_class: int = 12,
    *,
    per_action: int = 6,
    max_correct_examples: int = 120,
    include_negatives: bool = False,
    max_negative_examples: int = 24,
    state_lookup: Mapping[Tuple[int, int], Mapping[str, Any]] | None = None,
    evidence_cycle: int = 0,
    evidence_seed: int = 20261028,
) -> Dict[str, Any]:
    correct = _load_jsonl(bank_dir / "critical_correct.jsonl")
    negative = _load_jsonl(bank_dir / "critical_negative.jsonl")

    # Backward-compatible alias: old callers passed per_class as a flat head limit.
    if per_class != 12 and per_action == 6 and max_correct_examples == 120:
        max_correct_examples = max(per_class, 12)

    payload: Dict[str, Any] = {
        "n_critical_correct": len(correct),
        "n_critical_negative": len(negative),
        "correct_action_histogram": dict(Counter(r.get("action") for r in correct)),
        "negative_action_histogram": dict(Counter(r.get("action") for r in negative)),
        "correct_examples": _stratified_examples(
            correct,
            per_action=per_action,
            max_total=max_correct_examples,
            state_lookup=state_lookup,
            evidence_cycle=evidence_cycle,
            evidence_seed=evidence_seed,
        ),
        "sampling": {
            "mode": "action_stratified_mixed",
            "per_action": per_action,
            "max_correct_examples": max_correct_examples,
            "fixed_per_action": 0,
            "evidence_cycle": evidence_cycle,
            "evidence_seed": evidence_seed,
            "include_negatives": include_negatives,
            "note": (
                "One mixed example pool covering all actions. Raw examples rotate "
                "deterministically by cycle with no permanently fixed examples."
            ),
        },
        "note": (
            "These critical states come from live teacher rollouts with repeated "
            "continuation estimates. Use them as hints about important decisions; "
            "final fidelity is measured on Decision DB rows. Prefer diverse skills "
            "across movement, interaction, place, and make — do not collapse to one theme."
        ),
    }
    if include_negatives:
        payload["negative_examples"] = _stratified_examples(
            negative,
            per_action=2,
            max_total=max_negative_examples,
            state_lookup=state_lookup,
            evidence_cycle=evidence_cycle,
            evidence_seed=evidence_seed,
        )
    else:
        payload["negative_examples"] = []
    return payload


def _load_evidence_frame(decisions_csv: Path) -> pd.DataFrame:
    columns = list(pd.read_csv(decisions_csv, nrows=0).columns)
    direct = {
        "split",
        "episode_idx",
        "seed",
        "step",
        "action",
        "front",
        "health",
        "food",
        "drink",
        "energy",
        "daylight",
        "is_night",
        "sleeping",
        "next_achievement",
        "steps_to_achievement",
        "achievement_horizon",
        "achievement_horizon_complete",
    }
    prefixes = ("inv_", "ach_", "visible_", "dist_", "dir_")
    usecols = [
        column
        for column in columns
        if column in direct or column.startswith(prefixes)
    ]
    return pd.read_csv(decisions_csv, usecols=usecols, low_memory=False)


def _decision_lookup(
    frame: pd.DataFrame,
    keys: set[Tuple[int, int]] | None = None,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if "episode_idx" not in frame or "step" not in frame:
        return lookup
    selected = frame
    if keys is not None:
        mask = [
            (int(episode), int(step)) in keys
            for episode, step in zip(frame["episode_idx"], frame["step"])
        ]
        selected = frame.loc[mask]
    for row in selected.to_dict(orient="records"):
        key = (int(row["episode_idx"]), int(row["step"]))
        lookup[key] = row
    return lookup


def _critical_keys(bank_dir: Path) -> set[Tuple[int, int]]:
    keys: set[Tuple[int, int]] = set()
    for name in ("critical_correct.jsonl", "critical_negative.jsonl"):
        for row in _load_jsonl(bank_dir / name):
            keys.add((int(row.get("episode_idx", -1)), int(row.get("step", -1))))
    return keys


def _progress_examples(
    frame: pd.DataFrame,
    *,
    per_achievement: int = 4,
    max_total: int = 64,
    evidence_cycle: int = 0,
    evidence_seed: int = 20261028,
) -> List[Dict[str, Any]]:
    required = {"next_achievement", "achievement_horizon_complete"}
    if not required.issubset(frame.columns):
        return []
    work = frame[frame["split"] == "exploration"] if "split" in frame else frame
    work = work[
        work["next_achievement"].notna()
        & (work["next_achievement"] != "NONE")
        & work["achievement_horizon_complete"].map(_as_bool)
    ]
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    seen_episodes: Dict[str, set[int]] = {}
    for row in work.to_dict(orient="records"):
        target = row.get("next_achievement")
        if not _is_present(target) or target == "NONE":
            continue
        if not _as_bool(row.get("achievement_horizon_complete")):
            continue
        episode = int(row.get("episode_idx", -1))
        target_key = str(target)
        if episode in seen_episodes.setdefault(target_key, set()):
            continue
        seen_episodes[target_key].add(episode)
        by_target.setdefault(target_key, []).append(row)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for target, rows in sorted(by_target.items()):
        rows.sort(
            key=lambda row: (
                (
                    float(row["steps_to_achievement"])
                    if _is_present(row.get("steps_to_achievement"))
                    else float("inf")
                ),
                int(row.get("episode_idx", -1)),
            )
        )
        fixed: List[Dict[str, Any]] = []
        rotating = rows
        random.Random(_stable_seed(evidence_seed, f"progress:{target}")).shuffle(rotating)
        rotating_count = min(max(per_achievement - len(fixed), 0), len(rotating))
        start = evidence_cycle * rotating_count
        chosen = [(row, "fixed_anchor", evidence_cycle + 1) for row in fixed]
        selected_rotating = _cyclic_window(rotating, start, rotating_count)
        index_by_id = {id(row): index for index, row in enumerate(rotating)}
        chosen.extend(
            (
                row,
                "rotating",
                _rotation_exposure_count(
                    len(rotating),
                    rotating_count,
                    index_by_id[id(row)],
                    evidence_cycle,
                ),
            )
            for row in selected_rotating
        )
        buckets[target] = [
            {
                "evidence_id": _evidence_id("progress", row),
                "selection": selection,
                "exposure_count": exposure_count,
                "state": _state_snapshot(row),
                "current_action": row.get("action"),
                "next_achievement": target,
                "steps_to_achievement": (
                    int(row["steps_to_achievement"])
                    if _is_present(row.get("steps_to_achievement"))
                    else None
                ),
                "achievement_horizon": (
                    int(row["achievement_horizon"])
                    if _is_present(row.get("achievement_horizon"))
                    else None
                ),
            }
            for row, selection, exposure_count in chosen
        ]

    selected: List[Dict[str, Any]] = []
    index = 0
    while len(selected) < max_total and buckets:
        progressed = False
        for target in list(buckets):
            if index < len(buckets[target]):
                selected.append(buckets[target][index])
                progressed = True
                if len(selected) >= max_total:
                    break
            else:
                del buckets[target]
        if not progressed:
            break
        index += 1
    return selected


def _policy_aggregates(frame: pd.DataFrame) -> Dict[str, Any]:
    work = frame[frame["split"] == "exploration"] if "split" in frame else frame
    action_summaries = {}
    for action, group in work.groupby("action", dropna=False):
        action_summaries[str(action)] = {
            "rows": int(len(group)),
            "independent_seeds": int(group["seed"].nunique()) if "seed" in group else None,
            "top_fronts": {
                str(key): int(value)
                for key, value in group["front"].value_counts().head(5).items()
            },
            "median_vitals": {
                field: float(group[field].median())
                for field in ("health", "food", "drink", "energy")
                if field in group
            },
        }

    achievement_summaries = {}
    if "next_achievement" in work:
        progress = work[
            work["next_achievement"].notna()
            & (work["next_achievement"] != "NONE")
            & work["achievement_horizon_complete"].map(_as_bool)
        ]
        anchors = progress.drop_duplicates(["episode_idx", "next_achievement"])
        for target, group in anchors.groupby("next_achievement"):
            achievement_summaries[str(target)] = {
                "independent_episode_anchors": int(len(group)),
                "independent_seeds": int(group["seed"].nunique()) if "seed" in group else None,
                "median_steps_to_achievement": float(group["steps_to_achievement"].median()),
                "top_fronts": {
                    str(key): int(value)
                    for key, value in group["front"].value_counts().head(5).items()
                },
            }
    return {
        "source_split": "exploration",
        "n_rows": int(len(work)),
        "action_summaries": action_summaries,
        "next_achievement_summaries": achievement_summaries,
        "interpretation": (
            "Full-split descriptive aggregates are context, not hypotheses. "
            "Use rotating raw examples to infer discriminating conditions."
        ),
    }


def _failure_memory(results: List[Dict[str, Any]], max_groups: int = 16) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in results:
        label = str(row.get("label") or "")
        if not (label == "reject" or label.startswith("reject_")):
            continue
        behavior = row.get("behavior") or {}
        btype = str(behavior.get("type") or "unknown")
        prediction = behavior.get("action") or behavior.get("target") or behavior.get("actions")
        key = (btype, json.dumps(prediction, sort_keys=True), str(row.get("reason") or "unknown"))
        groups.setdefault(key, []).append(row)

    summaries = []
    for (btype, prediction_json, reason), items in groups.items():
        supports = [int(item["support"]) for item in items if item.get("support") is not None]
        fidelities = [
            float(item["fidelity"]) for item in items if item.get("fidelity") is not None
        ]
        representative = items[0]
        summaries.append(
            {
                "behavior_type": btype,
                "prediction": json.loads(prediction_json),
                "reason": reason,
                "repeated_count": len(items),
                "median_support": median(supports) if supports else None,
                "median_fidelity": median(fidelities) if fidelities else None,
                "representative_condition": representative.get("condition"),
                "guidance": (
                    "Do not merely retune numeric thresholds and re-propose this pattern. "
                    "Retry only with a new discriminating observable or new evidence."
                ),
            }
        )
    summaries.sort(key=lambda item: (-item["repeated_count"], item["behavior_type"]))
    return summaries[:max_groups]


def decision_db_field_guide(columns) -> Dict[str, Any]:
    allowed = sorted(allowed_fields_from_columns(columns))
    return {
        "allowed_fields": allowed,
        "forbidden_fields": sorted(FORBIDDEN_FIELDS),
        "ops": sorted(OPS),
        "actions": ACTIONS,
        "behavior_types": [
            {
                "type": "action",
                "example": {"type": "action", "action": "Do"},
            },
            {
                "type": "action_distribution",
                "example": {"type": "action_distribution", "actions": ["Move North", "Move South"]},
            },
            {
                "type": "next_interaction_target",
                "example": {"type": "next_interaction_target", "target": "water"},
            },
            {
                "type": "next_achievement",
                "example": {"type": "next_achievement", "target": "collect_stone"},
                "semantics": (
                    "Predict the next newly unlocked achievement within the fixed "
                    "Decision DB hindsight horizon; this is an observed progress event, "
                    "not privileged teacher intent."
                ),
            },
        ],
        "condition_examples": [
            {"field": "energy", "op": "<=", "value": 2},
            {
                "all": [
                    {"field": "front", "op": "==", "value": "tree"},
                    {"field": "visible_tree", "op": "==", "value": True},
                ]
            },
        ],
    }


def build_proposer_context(
    decisions_csv: Path,
    critical_bank_dir: Path,
    baseline_report: Path | None = None,
    existing_classified: Path | None = None,
    evidence_cycle: int = 0,
    evidence_seed: int = 20261028,
) -> Dict[str, Any]:
    columns = pd.read_csv(decisions_csv, nrows=0).columns
    evidence_frame = _load_evidence_frame(decisions_csv)
    state_lookup = _decision_lookup(evidence_frame, keys=_critical_keys(critical_bank_dir))
    context: Dict[str, Any] = {
        "task": (
            "Propose executable hypotheses that describe the PPO-AD teacher's "
            "decision policy. Do not restate generic game manuals. Prefer rules "
            "that explain critical states and are likely non-trivial versus random "
            "or pure crafting mechanics."
        ),
        "dsl": decision_db_field_guide(columns),
        "critical_bank": summarize_critical_bank(
            critical_bank_dir,
            state_lookup=state_lookup,
            evidence_cycle=evidence_cycle,
            evidence_seed=evidence_seed,
        ),
        "progress_evidence": {
            "semantics": (
                "Observed trajectory transitions. next_achievement is the next newly "
                "unlocked achievement within a fixed horizon, not a hidden intent label."
            ),
            "sampling": {
                "mode": "achievement_stratified_rotating",
                "per_achievement": 4,
                "max_examples": 64,
                "fixed_per_achievement": 0,
                "evidence_cycle": evidence_cycle,
                "evidence_seed": evidence_seed,
            },
            "examples": _progress_examples(
                evidence_frame,
                evidence_cycle=evidence_cycle,
                evidence_seed=evidence_seed,
            ),
        },
        "policy_aggregates": _policy_aggregates(evidence_frame),
    }

    if baseline_report and Path(baseline_report).exists():
        report = json.loads(Path(baseline_report).read_text(encoding="utf-8"))
        baselines = report.get("baselines", {})
        if not isinstance(baselines, dict):
            baselines = {}
        b0 = baselines.get("B0_random_legal", {}) if isinstance(baselines, dict) else {}
        b3 = baselines.get("B3_mechanics_data_yaml", {}) if isinstance(baselines, dict) else {}
        if not isinstance(b0, dict):
            b0 = {}
        if not isinstance(b3, dict):
            b3 = {}
        context["baseline_reference"] = {
            "B0_random_agreement": b0.get("agreement"),
            "B3_mechanics_agreement": b3.get("agreement"),
            "interpretation": (
                "A useful teacher rule should beat B0 clearly. If it merely matches "
                "obvious collect/craft mechanics, the harness will label it "
                "mechanics_like. Prefer teacher timing/priority/combat rules that "
                "mechanics alone do not explain."
            ),
        }

    if existing_classified and Path(existing_classified).exists():
        payload = json.loads(Path(existing_classified).read_text(encoding="utf-8"))
        results = payload.get("results") or []
        context["already_found_rules"] = [
            {
                "hypothesis_id": row.get("hypothesis_id"),
                "label": row.get("label"),
                "fidelity": row.get("fidelity"),
                "lift_vs_b3": row.get("lift_vs_b3"),
                "behavior": row.get("behavior"),
                "condition": row.get("condition"),
            }
            for row in results
            if row.get("label") in {"mechanics_like", "teacher_candidate"}
        ]
        context["promising_undercovered"] = [
            {
                "hypothesis_id": row.get("hypothesis_id"),
                "fidelity": row.get("fidelity"),
                "support": row.get("support"),
                "independent_seeds": row.get("independent_seeds"),
                "behavior": row.get("behavior"),
                "condition": row.get("condition"),
            }
            for row in results
            if row.get("label") == "promising_undercovered"
        ][:20]
        context["failure_memory"] = _failure_memory(results)
        context["avoid_duplicates"] = (
            "Do not merely rewrite already_found_rules or threshold-tune a pattern in "
            "failure_memory. You remain free to discover any evidence-supported rule; "
            "retry a failed pattern only when the supplied observations reveal a new "
            "discriminating condition."
        )
    return context
