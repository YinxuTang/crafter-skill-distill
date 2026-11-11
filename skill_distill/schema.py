"""Canonical schema and predicate evaluation for grounded Crafter skills."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from skill_distill.crafter import ACTIONS


SCHEMA_VERSION = 1
PREDICATE_OPS = {"eq", "ne", "ge", "gt", "le", "lt", "contains", "truthy"}


@dataclass
class EvidenceClaim:
    claim: str
    episode_id: str
    step_start: int
    step_stop: int
    event_id: str


@dataclass
class GroundedSkill:
    skill_id: str
    goal: str
    summary: str
    preconditions: List[Dict[str, Any]]
    decision_rule: str
    procedure: List[str]
    recommended_actions: List[str]
    invariants: List[str]
    success_check: List[Dict[str, Any]]
    failure_recovery: List[str]
    requires: List[str] = field(default_factory=list)
    before: List[str] = field(default_factory=list)
    evidence_claims: List[EvidenceClaim] = field(default_factory=list)
    natural_language_claims: List[Dict[str, Any]] = field(default_factory=list)
    support_count: int = 0
    support_seeds: List[int] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"
    version: int = 1
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "GroundedSkill":
        data = dict(value)
        data["evidence_claims"] = [
            claim if isinstance(claim, EvidenceClaim) else EvidenceClaim(**claim)
            for claim in data.get("evidence_claims", [])
        ]
        return cls(**data)


def _lookup(state: Dict[str, Any], path: str) -> Any:
    value: Any = state
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def evaluate_predicate(predicate: Dict[str, Any], state: Dict[str, Any]) -> bool:
    field_name = predicate.get("field")
    op = predicate.get("op")
    expected = predicate.get("value")
    if op not in PREDICATE_OPS or not isinstance(field_name, str):
        return False
    actual = _lookup(state, field_name)
    if op == "truthy":
        return bool(actual)
    if op == "contains":
        return actual is not None and expected in actual
    if actual is None:
        return False
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "ge":
        return actual >= expected
    if op == "gt":
        return actual > expected
    if op == "le":
        return actual <= expected
    if op == "lt":
        return actual < expected
    return False


def validate_skill(skill: GroundedSkill) -> List[str]:
    errors = []
    if not skill.skill_id or not skill.goal:
        errors.append("skill_id and goal are required")
    if skill.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {skill.schema_version}")
    invalid_actions = [action for action in skill.recommended_actions if action not in ACTIONS]
    if invalid_actions:
        errors.append(f"invalid recommended actions: {invalid_actions}")
    for group_name, predicates in (
        ("preconditions", skill.preconditions),
        ("success_check", skill.success_check),
    ):
        for predicate in predicates:
            if predicate.get("op") not in PREDICATE_OPS or not predicate.get("field"):
                errors.append(f"invalid {group_name} predicate: {predicate}")
    for claim in skill.evidence_claims:
        if claim.step_start < 0 or claim.step_stop <= claim.step_start:
            errors.append(f"invalid evidence span: {claim.event_id}")
    if skill.status not in {"candidate", "validated", "retired"}:
        errors.append(f"invalid status {skill.status}")
    return errors


def render_skill(skill: GroundedSkill) -> str:
    preconditions = "; ".join(
        f"{item['field']} {item['op']} {item.get('value', '')}".strip()
        for item in skill.preconditions
    ) or "none"
    procedure = " → ".join(skill.procedure)
    recovery = "; ".join(skill.failure_recovery) or "replan"
    citations = ", ".join(
        f"{claim.episode_id}[{claim.step_start}:{claim.step_stop}]"
        for claim in skill.evidence_claims
    )
    return (
        f"[{skill.skill_id} v{skill.version}]\n"
        f"Goal: {skill.goal}\n"
        f"When: {preconditions}\n"
        f"Rule: {skill.decision_rule}\n"
        f"Procedure: {procedure}\n"
        f"Stop/Recover: {recovery}\n"
        f"Evidence: {skill.support_count} windows across {len(skill.support_seeds)} seeds.\n"
        f"Citations: {citations or 'none'}"
    )


SKILL_JSON_SCHEMA = {
    "type": "object",
    "required": [
        "skill_id",
        "goal",
        "summary",
        "preconditions",
        "decision_rule",
        "procedure",
        "recommended_actions",
        "success_check",
        "evidence_claims",
    ],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "recommended_actions": {"type": "array", "items": {"enum": ACTIONS}},
        "status": {"enum": ["candidate", "validated", "retired"]},
    },
}


def schema_json() -> str:
    return json.dumps(SKILL_JSON_SCHEMA, indent=2)
