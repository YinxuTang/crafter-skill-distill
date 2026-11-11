"""State-aware retrieval and lightweight action verification for Crafter skills."""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from skill_distill.crafter import ACTIONS, read_jsonl
from skill_distill.schema import GroundedSkill, evaluate_predicate, render_skill


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def parse_observation(text: str) -> Dict[str, Any]:
    inventory: Dict[str, int] = {}
    vitals: Dict[str, int] = {}
    front = None
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"- ([a-z_]+): (\d+)(?:/9)?$", stripped)
        if match:
            name, value = match.group(1), int(match.group(2))
            inventory[name] = value
            if name in ("health", "food", "drink", "energy"):
                vitals[name] = value
        if stripped.startswith("You face ") and stripped.endswith(" at your front."):
            front = stripped[len("You face ") : -len(" at your front.")]
    return {
        "inventory": inventory,
        "vitals": vitals,
        "front": front,
        "visible_text": text.lower(),
        "achievements": {},
    }


def _tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


class SkillRetriever:
    def __init__(
        self,
        path: str,
        top_k: int = 3,
        allow_candidates: bool = False,
    ):
        self.path = str(Path(path).expanduser())
        self.top_k = int(top_k)
        self.skills = [
            GroundedSkill.from_dict(item)
            for item in read_jsonl(self.path)
            if allow_candidates or item.get("status") == "validated"
        ]

    def retrieve(self, observation_text: str, goal_hint: str = "") -> List[GroundedSkill]:
        state = parse_observation(observation_text)
        query_tokens = _tokens(observation_text + " " + goal_hint)
        ranked: List[Tuple[float, GroundedSkill]] = []
        for skill in self.skills:
            if not all(evaluate_predicate(predicate, state) for predicate in skill.preconditions):
                continue
            skill_tokens = _tokens(
                " ".join([skill.goal, skill.summary, skill.decision_rule, *skill.procedure])
            )
            overlap = len(query_tokens & skill_tokens)
            goal_bonus = 4 if skill.goal.lower() in goal_hint.lower() else 0
            dependency_bonus = sum(
                1 for dependency in skill.requires if dependency.lower() in goal_hint.lower()
            )
            ranked.append((float(overlap + goal_bonus + dependency_bonus), skill))
        ranked.sort(key=lambda item: (-item[0], item[1].skill_id))
        return [skill for _, skill in ranked[: self.top_k]]

    @staticmethod
    def render(skills: Iterable[GroundedSkill]) -> str:
        rendered = [render_skill(skill) for skill in skills]
        return "\n\n".join(rendered)


def _inventory_at_least(state: Dict[str, Any], requirements: Dict[str, int]) -> bool:
    inventory = state.get("inventory", {})
    return all(int(inventory.get(item, 0)) >= count for item, count in requirements.items())


ACTION_REQUIREMENTS = {
    "Place Stone": {"stone": 1},
    "Place Table": {"wood": 2},
    "Place Furnace": {"stone": 4},
    "Place Plant": {"sapling": 1},
    "Make Wood Pickaxe": {"wood": 1},
    "Make Stone Pickaxe": {"wood": 1, "stone": 1},
    "Make Iron Pickaxe": {"wood": 1, "coal": 1, "iron": 1},
    "Make Wood Sword": {"wood": 1},
    "Make Stone Sword": {"wood": 1, "stone": 1},
    "Make Iron Sword": {"wood": 1, "coal": 1, "iron": 1},
}


def verify_action(action: str, state: Dict[str, Any]) -> Tuple[bool, str]:
    if action not in ACTIONS:
        return False, "action is not in the Crafter action set"
    if action == "Sleep" and int(state.get("vitals", {}).get("energy", 9)) >= 9:
        return False, "energy is already full"
    if action == "Do" and state.get("front") in (None, "nothing"):
        return False, "nothing is directly in front"
    requirements = ACTION_REQUIREMENTS.get(action)
    if requirements and not _inventory_at_least(state, requirements):
        return False, f"inventory does not satisfy {requirements}"
    if action.startswith("Make ") and "table" not in state.get("visible_text", ""):
        return False, "no visible nearby table"
    if action.startswith("Make Iron ") and "furnace" not in state.get("visible_text", ""):
        return False, "no visible nearby furnace"
    return True, ""


def transition_feedback(action: Optional[str], before: Optional[Dict[str, Any]], after: Dict[str, Any]) -> str:
    if not action or not before:
        return ""
    if action in ("Noop", "Sleep") or action.startswith("Move "):
        return ""
    old_inventory = before.get("inventory", {})
    new_inventory = after.get("inventory", {})
    changed = any(
        int(new_inventory.get(key, 0)) != int(old_inventory.get(key, 0))
        for key in set(old_inventory) | set(new_inventory)
    )
    if not changed:
        return (
            f"The previous action {action} produced no observable inventory delta. "
            "Recheck facing, tools, resources, and workstation requirements before repeating it."
        )
    return ""
