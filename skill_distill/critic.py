#!/usr/bin/env python3
"""Adversarial grounding review for induced trajectory skills."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from skill_distill import PROJECT_ROOT
from skill_distill.crafter import read_jsonl, write_jsonl
from skill_distill.trae_backend import TraeCLIBackend


SCHEMA_PATH = PROJECT_ROOT / "skill_distill" / "schemas" / "critic.schema.json"


def build_prompt(pack: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    return (
        "Act as an adversarial grounding critic for an RL-policy skill. "
        "Check the candidate sentence by sentence against the supplied trajectory evidence.\n\n"
        "Rules:\n"
        "1. Use only the evidence pack; no external Crafter or game knowledge.\n"
        "2. Reject post-hoc explanations, unsupported causality, omitted counterexamples, "
        "over-broad preconditions, and instructions not demonstrated by the policy.\n"
        "3. Check that every cited evidence_id exists and actually supports the associated text.\n"
        "4. A replay succeeding after several actions does not prove the first action caused success.\n"
        "5. Verdict accept only when all material claims are entailed. Use revise when unsupported "
        "details can be removed or narrowed; reject when no useful grounded rule remains.\n"
        "6. Return only the required JSON. Do not inspect files or use tools.\n\n"
        "CANDIDATE SKILL:\n"
        + json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        + "\n\nEVIDENCE PACK:\n"
        + json.dumps(pack, ensure_ascii=False, separators=(",", ":"))
    )


def validate_critic(review: Dict[str, Any], pack: Dict[str, Any]) -> List[str]:
    valid_ids = {item["evidence_id"] for item in pack["evidence"]}
    cited: Set[str] = set()
    for item in review.get("claim_reviews", []):
        cited.update(item.get("checked_evidence_ids", []))
    for item in review.get("issues", []):
        cited.update(item.get("evidence_ids", []))
    unknown = cited - valid_ids
    return [f"critic cited unknown evidence ids: {sorted(unknown)}"] if unknown else []


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packs", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="GPT-6.1-Sol")
    parser.add_argument("--log-dir")
    args = parser.parse_args()

    packs = {pack["pack_id"]: pack for pack in read_jsonl(args.packs)}
    candidates = read_jsonl(args.candidates)
    backend = TraeCLIBackend(
        model=args.model,
        workspace=str(PROJECT_ROOT),
        log_dir=args.log_dir,
    )
    results = []
    for index, candidate in enumerate(candidates):
        pack = packs[candidate["pack_id"]]
        if candidate["local_validation_errors"]:
            results.append(
                {
                    **candidate,
                    "critic": None,
                    "critic_validation_errors": [],
                    "status": "invalid",
                }
            )
            continue
        review = backend.generate(
            build_prompt(pack, candidate["skill"]),
            str(SCHEMA_PATH),
            audit_name=f"{index:03d}_{safe_name(candidate['goal'])}_critic",
        )
        errors = validate_critic(review, pack)
        status = review["verdict"] if not errors else "invalid"
        results.append(
            {
                **candidate,
                "critic": review,
                "critic_validation_errors": errors,
                "status": status,
            }
        )
        print(f"[{index + 1}/{len(candidates)}] {candidate['goal']}: {status}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, results)
    counts: Dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print(json.dumps({"results": len(results), "status": counts}, indent=2))


if __name__ == "__main__":
    main()
