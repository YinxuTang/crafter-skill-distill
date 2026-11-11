#!/usr/bin/env python3
"""Turn canonical grounded skills into a natural-language book using ChatGPT."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from skill_distill.crafter import read_jsonl, write_jsonl
from skill_distill.schema import GroundedSkill, render_skill, validate_skill


ALLOWED_REWRITES = {
    "summary",
    "decision_rule",
    "procedure",
    "failure_recovery",
    "natural_language_claims",
}


def compact_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    positive = pair["positive"]
    anchor = next(
        step for step in positive["steps"] if step["step"] == positive["anchor_step"]
    )
    return {
        "positive": {
            "event_id": positive["event_id"],
            "episode_id": positive["episode_id"],
            "step_span": [positive["step_start"], positive["step_stop"]],
            "observation": anchor["text_obs"],
            "action": anchor["action_name"],
            "before": anchor["before"],
            "after": anchor["after"],
            "unlocked": anchor["unlocked"],
        },
        "negatives": [
            {
                "event_id": item["event"]["event_id"],
                "distance": item["distance"],
                "anchor_action": next(
                    step["action_name"]
                    for step in item["event"]["steps"]
                    if step["step"] == item["event"]["anchor_step"]
                ),
            }
            for item in pair.get("negatives", [])
        ],
    }


def build_prompt(skill: GroundedSkill, pairs: List[Dict[str, Any]]) -> str:
    payload = {
        "canonical_skill": skill.to_dict(),
        "contrastive_evidence": [compact_pair(pair) for pair in pairs[:12]],
    }
    return (
        "You are writing a human-readable Crafter policy skill book distilled from an RL policy.\n"
        "Rewrite the skill into concise natural language. Infer only rules supported by the evidence.\n"
        "Do not change machine predicates, action names, dependencies, status, or evidence spans.\n"
        "Return one JSON object with exactly these keys:\n"
        "summary: string; decision_rule: string; procedure: array of strings; "
        "failure_recovery: array of strings; natural_language_claims: array of "
        "{text: string, event_ids: array of cited event_id strings}.\n"
        "Every factual claim must cite at least one supplied positive event_id. "
        "If evidence is insufficient, say so instead of inventing a rule.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


class OpenAIChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_json(self, prompt: str, retries: int = 4) -> Dict[str, Any]:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Produce grounded policy documentation as strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                return json.loads(content)
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as error:
                if attempt + 1 == retries:
                    raise RuntimeError(f"ChatGPT summarization failed: {error}") from error
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def validate_rewrite(skill: GroundedSkill, rewrite: Dict[str, Any]) -> List[str]:
    errors = []
    unknown = set(rewrite) - ALLOWED_REWRITES
    if unknown:
        errors.append(f"unknown rewrite fields: {sorted(unknown)}")
    event_ids = {claim.event_id for claim in skill.evidence_claims}
    claims = rewrite.get("natural_language_claims", [])
    if not isinstance(claims, list):
        return errors + ["natural_language_claims must be a list"]
    for claim in claims:
        cited = claim.get("event_ids", []) if isinstance(claim, dict) else []
        if not cited:
            errors.append("natural-language claim has no citation")
        invalid = set(cited) - event_ids
        if invalid:
            errors.append(f"claim cites unknown events: {sorted(invalid)}")
    for field_name in ("summary", "decision_rule"):
        if not isinstance(rewrite.get(field_name), str):
            errors.append(f"{field_name} must be a string")
    for field_name in ("procedure", "failure_recovery"):
        if not isinstance(rewrite.get(field_name), list) or not all(
            isinstance(value, str) for value in rewrite[field_name]
        ):
            errors.append(f"{field_name} must be a string array")
    return errors


def apply_rewrite(skill: GroundedSkill, rewrite: Dict[str, Any]) -> GroundedSkill:
    errors = validate_rewrite(skill, rewrite)
    if errors:
        raise ValueError(f"invalid rewrite for {skill.skill_id}: {errors}")
    for field_name in ALLOWED_REWRITES:
        if field_name in rewrite:
            setattr(skill, field_name, rewrite[field_name])
    schema_errors = validate_skill(skill)
    if schema_errors:
        raise ValueError(f"rewrite broke canonical schema: {schema_errors}")
    return skill


def template_rewrite(skill: GroundedSkill) -> Dict[str, Any]:
    return {
        "summary": skill.summary,
        "decision_rule": skill.decision_rule,
        "procedure": skill.procedure,
        "failure_recovery": skill.failure_recovery,
        "natural_language_claims": [
            {
                "text": claim.claim,
                "event_ids": [claim.event_id],
            }
            for claim in skill.evidence_claims
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--backend", choices=("openai", "template"), default="openai")
    parser.add_argument("--model", default=os.environ.get("SKILL_SUMMARIZER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    args = parser.parse_args()

    skills = [GroundedSkill.from_dict(item) for item in read_jsonl(args.skills)]
    pairs_by_goal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pair in read_jsonl(args.pairs):
        pairs_by_goal[pair["goal"]].append(pair)
    client = None
    if args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --backend openai")
        client = OpenAIChatClient(api_key, args.base_url, args.model)

    rewritten = []
    for skill in skills:
        if client:
            rewrite = client.generate_json(build_prompt(skill, pairs_by_goal[skill.goal]))
        else:
            rewrite = template_rewrite(skill)
        rewritten.append(apply_rewrite(skill, rewrite))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, [skill.to_dict() for skill in rewritten])
    Path(args.book).write_text(
        "\n\n".join(render_skill(skill) for skill in rewritten) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"skills": len(rewritten), "backend": args.backend, "model": args.model}, indent=2))


if __name__ == "__main__":
    main()
