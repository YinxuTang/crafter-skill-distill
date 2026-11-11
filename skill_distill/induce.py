#!/usr/bin/env python3
"""Induce natural-language skills from factual evidence packs via Trae CLI."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from skill_distill import PROJECT_ROOT
from skill_distill.crafter import read_jsonl, write_jsonl
from skill_distill.trae_backend import TraeCLIBackend


SCHEMA_PATH = PROJECT_ROOT / "skill_distill" / "schemas" / "induced_skill.schema.json"


def build_prompt(pack: Dict[str, Any]) -> str:
    return (
        "You are distilling a trained RL policy into a human-readable skill. "
        "You receive a factual evidence pack containing successful, failed, and "
        "counterfactual trajectory windows.\n\n"
        "Rules:\n"
        "1. Infer policy behavior only from the supplied JSON. Do not use prior knowledge "
        "about Crafter, Minecraft, recipes, or game mechanics.\n"
        "2. Contrast successes with failures and counterfactual branches. Correlation is not causation.\n"
        "3. Every precondition, policy step, stop condition, failure mode, exception, dependency, "
        "and claim must cite exact evidence_id values from the pack.\n"
        "4. If evidence does not support a field, leave its array empty and explain the gap in "
        "insufficient_evidence. Never invent a plausible rule.\n"
        "5. Write clear natural language for a human reader. Preserve the observed_goal_label "
        "exactly in the goal field. Use a stable skill_id beginning with 'crafter.induced.'.\n"
        "6. A failed window is evidence only for what failed in that state; do not assume why "
        "unless the observations or counterfactuals distinguish the cause.\n"
        "7. Return only the JSON object required by the output schema. Do not inspect files or use tools.\n\n"
        "EVIDENCE PACK:\n"
        + json.dumps(pack, ensure_ascii=False, separators=(",", ":"))
    )


def cited_ids(skill: Dict[str, Any]) -> Set[str]:
    result: Set[str] = set()
    for key in (
        "preconditions",
        "policy_steps",
        "stop_conditions",
        "failure_modes",
        "exceptions",
        "dependencies",
    ):
        for item in skill.get(key, []):
            result.update(item.get("evidence_ids", []))
    for claim in skill.get("claims", []):
        result.update(claim.get("supporting_evidence_ids", []))
        result.update(claim.get("contradicting_evidence_ids", []))
    return result


def validate_induced_skill(skill: Dict[str, Any], pack: Dict[str, Any]) -> List[str]:
    errors = []
    valid_ids = {item["evidence_id"] for item in pack["evidence"]}
    unknown = cited_ids(skill) - valid_ids
    if unknown:
        errors.append(f"unknown evidence ids: {sorted(unknown)}")
    if skill.get("goal") != pack["observed_goal_label"]:
        errors.append("goal does not match observed_goal_label")
    if not str(skill.get("skill_id", "")).startswith("crafter.induced."):
        errors.append("skill_id must begin with crafter.induced.")
    for claim in skill.get("claims", []):
        if not claim.get("supporting_evidence_ids"):
            errors.append(f"uncited claim: {claim.get('text', '')}")
    return errors


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="GPT-6.1-Sol")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--goal", action="append", default=[])
    parser.add_argument("--log-dir")
    args = parser.parse_args()

    packs = read_jsonl(args.packs)
    if args.goal:
        requested = set(args.goal)
        packs = [pack for pack in packs if pack["observed_goal_label"] in requested]
    if args.limit:
        packs = packs[: args.limit]
    backend = TraeCLIBackend(
        model=args.model,
        workspace=str(PROJECT_ROOT),
        log_dir=args.log_dir,
    )
    results = []
    for index, pack in enumerate(packs):
        label = pack["observed_goal_label"]
        skill = backend.generate(
            build_prompt(pack),
            str(SCHEMA_PATH),
            audit_name=f"{index:03d}_{safe_name(label)}_induce",
        )
        errors = validate_induced_skill(skill, pack)
        results.append(
            {
                "schema_version": 2,
                "pack_id": pack["pack_id"],
                "goal": label,
                "model": args.model,
                "skill": skill,
                "local_validation_errors": errors,
                "status": "critic_pending" if not errors else "invalid",
            }
        )
        print(f"[{index + 1}/{len(packs)}] {label}: {results[-1]['status']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, results)
    print(json.dumps({"results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
