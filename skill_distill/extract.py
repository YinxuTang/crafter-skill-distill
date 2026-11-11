#!/usr/bin/env python3
"""Compile paired trajectory evidence into canonical grounded Crafter skills."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from skill_distill.crafter import read_jsonl, write_jsonl
from skill_distill.schema import EvidenceClaim, GroundedSkill, render_skill


GOAL_TARGET = {
    "collect_wood": "tree",
    "collect_stone": "stone",
    "collect_coal": "coal",
    "collect_iron": "iron",
    "collect_diamond": "diamond",
    "collect_drink": "water",
    "eat_cow": "cow",
    "defeat_zombie": "zombie",
    "defeat_skeleton": "skeleton",
}
GOAL_ACTION = {
    "place_table": "Place Table",
    "place_furnace": "Place Furnace",
    "place_plant": "Place Plant",
    "place_stone": "Place Stone",
    "make_wood_pickaxe": "Make Wood Pickaxe",
    "make_stone_pickaxe": "Make Stone Pickaxe",
    "make_iron_pickaxe": "Make Iron Pickaxe",
    "make_wood_sword": "Make Wood Sword",
    "make_stone_sword": "Make Stone Sword",
    "make_iron_sword": "Make Iron Sword",
}
REQUIREMENTS = {
    "place_table": {"inventory.wood": 2},
    "place_furnace": {"inventory.stone": 4},
    "place_plant": {"inventory.sapling": 1},
    "place_stone": {"inventory.stone": 1},
    "make_wood_pickaxe": {"inventory.wood": 1},
    "make_stone_pickaxe": {"inventory.wood": 1, "inventory.stone": 1},
    "make_iron_pickaxe": {"inventory.wood": 1, "inventory.coal": 1, "inventory.iron": 1},
    "make_wood_sword": {"inventory.wood": 1},
    "make_stone_sword": {"inventory.wood": 1, "inventory.stone": 1},
    "make_iron_sword": {"inventory.wood": 1, "inventory.coal": 1, "inventory.iron": 1},
}
DEPENDENCIES = {
    "place_table": ["collect_wood"],
    "make_wood_pickaxe": ["place_table", "collect_wood"],
    "collect_stone": ["make_wood_pickaxe"],
    "make_stone_pickaxe": ["place_table", "collect_stone"],
    "place_furnace": ["collect_stone"],
    "collect_coal": ["make_wood_pickaxe"],
    "collect_iron": ["make_stone_pickaxe"],
    "make_iron_pickaxe": ["place_table", "place_furnace", "collect_coal", "collect_iron"],
    "collect_diamond": ["make_iron_pickaxe"],
}


def _anchor(event: Dict[str, Any]) -> Dict[str, Any]:
    return next(step for step in event["steps"] if step["step"] == event["anchor_step"])


def _skill_contents(goal: str) -> Tuple[List[Dict[str, Any]], str, List[str], List[str], List[str]]:
    if goal in GOAL_TARGET:
        target = GOAL_TARGET[goal]
        return (
            [{"field": "visible_text", "op": "contains", "value": target}],
            f"When {target} is visible, approach until it is directly in front, then interact once.",
            [f"Move toward visible {target}", f"Face {target}", "Use Do once", "Check the state delta"],
            ["Move West", "Move East", "Move North", "Move South", "Do"],
            ["If distance does not decrease, change movement axis.", "If Do has no effect, recheck facing and tool."],
        )
    if goal in GOAL_ACTION:
        requirements = [
            {"field": field_name, "op": "ge", "value": value}
            for field_name, value in REQUIREMENTS.get(goal, {}).items()
        ]
        action = GOAL_ACTION[goal]
        return (
            requirements,
            f"Execute {action} only after its inventory and nearby-workstation requirements are observable.",
            ["Check inventory requirements", "Check nearby workstation requirements", action, "Check achievement delta"],
            [action],
            ["If no achievement delta occurs, do not repeat blindly; repair the missing requirement."],
        )
    if goal.startswith("recover_"):
        vital = goal.split("recover_", 1)[1]
        action = "Sleep" if vital == "energy" else "Do"
        return (
            [{"field": f"vitals.{vital}", "op": "le", "value": 3}],
            f"Interrupt progression while {vital} is dangerous and prioritize recovery.",
            [f"Locate a recovery opportunity for {vital}", action, f"Check that {vital} increased"],
            [action],
            ["If recovery is unavailable, avoid combat and continue searching."],
        )
    return (
        [],
        f"Apply the demonstrated local policy for {goal}, then check its postcondition.",
        ["Check requirements", "Execute the demonstrated action", "Check postcondition"],
        [_anchor_action_name(goal)],
        ["On no state change, stop repetition and replan."],
    )


def _anchor_action_name(goal: str) -> str:
    return GOAL_ACTION.get(goal, "Do")


def _success_predicates(goal: str) -> List[Dict[str, Any]]:
    if goal.startswith("recover_"):
        vital = goal.split("recover_", 1)[1]
        return [{"field": f"vitals.{vital}", "op": "gt", "value": 3}]
    return [{"field": f"achievements.{goal}", "op": "gt", "value": 0}]


def extract_skills(pairs: List[Dict[str, Any]]) -> List[GroundedSkill]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["goal"]].append(pair)

    skills = []
    for goal, support in sorted(grouped.items()):
        preconditions, rule, procedure, actions, recovery = _skill_contents(goal)
        evidence = []
        seeds = set()
        negative_count = 0
        for pair in support:
            event = pair["positive"]
            seeds.add(int(event["seed"]))
            negative_count += len(pair.get("negatives", []))
            evidence.append(
                EvidenceClaim(
                    claim=f"The cited window achieved {goal}.",
                    episode_id=event["episode_id"],
                    step_start=int(event["step_start"]),
                    step_stop=int(event["step_stop"]),
                    event_id=event["event_id"],
                )
            )
        skill = GroundedSkill(
            skill_id=f"crafter.{goal}",
            goal=goal,
            summary=f"Grounded local policy for {goal}.",
            preconditions=preconditions,
            decision_rule=rule,
            procedure=procedure,
            recommended_actions=actions,
            invariants=[
                "Interrupt the skill if health, food, or drink enters the danger range.",
                "Never treat an attempted action as success without an inventory, vital, or achievement delta.",
            ],
            success_check=_success_predicates(goal),
            failure_recovery=recovery,
            requires=DEPENDENCIES.get(goal, []),
            evidence_claims=evidence,
            support_count=len(support),
            support_seeds=sorted(seeds),
            validation={"matched_negative_count": negative_count},
        )
        skills.append(skill)
    return skills


def _prompt_record(skill: GroundedSkill, pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    evidence = [pair for pair in pairs if pair["goal"] == skill.goal]
    return {
        "skill_id": skill.skill_id,
        "instruction": (
            "Rewrite only the summary, decision_rule, procedure, and failure_recovery for human clarity. "
            "Do not add facts. Every statement must be entailed by the supplied positive/negative evidence. "
            "Return JSON and preserve all citations and machine predicates unchanged."
        ),
        "canonical_skill": skill.to_dict(),
        "evidence_pairs": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts-out")
    parser.add_argument("--human-out")
    args = parser.parse_args()
    pairs = read_jsonl(args.pairs)
    skills = extract_skills(pairs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, [skill.to_dict() for skill in skills])
    if args.prompts_out:
        write_jsonl(args.prompts_out, [_prompt_record(skill, pairs) for skill in skills])
    if args.human_out:
        Path(args.human_out).write_text(
            "\n\n".join(render_skill(skill) for skill in skills) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"skills": len(skills)}, indent=2))


if __name__ == "__main__":
    main()
