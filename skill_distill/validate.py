#!/usr/bin/env python3
"""Validate skill schemas, citations, replay forks, and held-out event behavior."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from skill_distill.crafter import read_jsonl, write_jsonl
from skill_distill.schema import GroundedSkill, evaluate_predicate, render_skill, validate_skill


def _anchor(event: Dict[str, Any]) -> Dict[str, Any]:
    return next(step for step in event["steps"] if step["step"] == event["anchor_step"])


def _online_state(event: Dict[str, Any]) -> Dict[str, Any]:
    anchor = _anchor(event)
    state = dict(anchor["before"])
    long_context = anchor["text_obs"]["long_term_context"]
    front = None
    for line in long_context.splitlines():
        if line.startswith("You face ") and line.endswith(" at your front."):
            front = line[len("You face ") : -len(" at your front.")]
    state["front"] = front
    state["visible_text"] = long_context
    return state


def _replay_success(goal: str, fork: Dict[str, Any]) -> bool:
    teacher = next((branch for branch in fork["branches"] if branch["is_teacher_action"]), None)
    if teacher is None:
        return False
    score = teacher["score"]
    if goal.startswith("recover_"):
        return score["survived"] and score["health_delta"] >= 0
    return goal in score["new_achievements"]


def validate_library(
    skills: List[GroundedSkill],
    events: List[Dict[str, Any]],
    forks: List[Dict[str, Any]],
    validation_events: List[Dict[str, Any]],
    min_seeds: int,
    min_citation_precision: float,
    min_replay_success: float,
    min_validation_count: int,
    min_validation_success: float,
) -> Dict[str, Any]:
    events_by_id = {event["event_id"]: event for event in events}
    forks_by_event = {fork["event_id"]: fork for fork in forks}
    report = {"skills": []}

    for skill in skills:
        schema_errors = validate_skill(skill)
        cited = 0
        correct = 0
        replay_total = 0
        replay_ok = 0
        for claim in skill.evidence_claims:
            event = events_by_id.get(claim.event_id)
            cited += 1
            if (
                event
                and event["episode_id"] == claim.episode_id
                and event["goal"] == skill.goal
                and event["success"]
                and claim.step_start >= event["step_start"]
                and claim.step_stop <= event["step_stop"]
            ):
                correct += 1
            fork = forks_by_event.get(claim.event_id)
            if fork:
                replay_total += 1
                replay_ok += int(_replay_success(skill.goal, fork))

        citation_precision = correct / cited if cited else 0.0
        replay_success = replay_ok / replay_total if replay_total else None
        applicable = [
            event
            for event in validation_events
            if event["goal"] == skill.goal
            and all(evaluate_predicate(predicate, _online_state(event)) for predicate in skill.preconditions)
        ]
        validation_success = (
            sum(bool(event["success"]) for event in applicable) / len(applicable)
            if applicable
            else None
        )
        negative_count = int(skill.validation.get("matched_negative_count", 0))
        passed = (
            not schema_errors
            and len(skill.support_seeds) >= min_seeds
            and citation_precision >= min_citation_precision
            and replay_success is not None
            and replay_success >= min_replay_success
            and negative_count > 0
            and len(applicable) >= min_validation_count
            and validation_success is not None
            and validation_success >= min_validation_success
        )
        skill.status = "validated" if passed else "candidate"
        skill.validation.update(
            {
                "schema_errors": schema_errors,
                "citation_precision": citation_precision,
                "replay_success": replay_success,
                "replay_count": replay_total,
                "validation_applicable_count": len(applicable),
                "validation_success": validation_success,
                "passed": passed,
            }
        )
        report["skills"].append(
            {
                "skill_id": skill.skill_id,
                "status": skill.status,
                **skill.validation,
            }
        )
    report["validated_count"] = sum(skill.status == "validated" for skill in skills)
    report["candidate_count"] = sum(skill.status == "candidate" for skill in skills)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--forks", required=True)
    parser.add_argument("--validation-events")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--book")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-citation-precision", type=float, default=0.95)
    parser.add_argument("--min-replay-success", type=float, default=0.70)
    parser.add_argument("--min-validation-count", type=int, default=1)
    parser.add_argument("--min-validation-success", type=float, default=0.50)
    args = parser.parse_args()

    skills = [GroundedSkill.from_dict(item) for item in read_jsonl(args.skills)]
    events = read_jsonl(args.events)
    forks = read_jsonl(args.forks)
    validation_events = read_jsonl(args.validation_events) if args.validation_events else []
    report = validate_library(
        skills,
        events,
        forks,
        validation_events,
        min_seeds=args.min_seeds,
        min_citation_precision=args.min_citation_precision,
        min_replay_success=args.min_replay_success,
        min_validation_count=args.min_validation_count,
        min_validation_success=args.min_validation_success,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, [skill.to_dict() for skill in skills])
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.book:
        Path(args.book).write_text(
            "\n\n".join(render_skill(skill) for skill in skills) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: report[key] for key in ("validated_count", "candidate_count")}, indent=2))


if __name__ == "__main__":
    main()
