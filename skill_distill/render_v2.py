#!/usr/bin/env python3
"""Render induced and critic-reviewed skills without adding policy content."""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from skill_distill.crafter import read_jsonl


def cited(values: Iterable[str]) -> str:
    items = list(values)
    return ", ".join(f"`{value}`" for value in items) if items else "none"


def render_items(items: List[Dict[str, Any]], text_key: str) -> List[str]:
    return [
        f"- {item[text_key]}  \n  Evidence: {cited(item.get('evidence_ids', []))}"
        for item in items
    ]


def render_result(result: Dict[str, Any]) -> str:
    skill = result["skill"]
    lines = [
        f"## {skill['title']}",
        f"Status: **{result['status']}** · Goal: `{skill['goal']}` · ID: `{skill['skill_id']}`",
        "",
        skill["summary"],
        "",
        "### Preconditions",
        *render_items(skill["preconditions"], "condition"),
        "",
        "### Policy",
        *render_items(skill["policy_steps"], "instruction"),
        "",
        "### Stop conditions",
        *render_items(skill["stop_conditions"], "condition"),
        "",
        "### Failure recovery",
    ]
    lines.extend(
        f"- **Symptom:** {item['symptom']}  \n  **Recovery:** {item['recovery']}  "
        f"\n  Evidence: {cited(item.get('evidence_ids', []))}"
        for item in skill["failure_modes"]
    )
    if skill["exceptions"]:
        lines.extend(["", "### Exceptions", *render_items(skill["exceptions"], "description")])
    if skill["insufficient_evidence"]:
        lines.extend(
            [
                "",
                "### Insufficient evidence",
                *[f"- {item}" for item in skill["insufficient_evidence"]],
            ]
        )
    critic = result.get("critic")
    if critic:
        lines.extend(["", "### Critic"])
        lines.append(f"Verdict: **{critic['verdict']}**")
        lines.extend(
            f"- [{item['severity']}] {item['description']} "
            f"(evidence: {cited(item.get('evidence_ids', []))})"
            for item in critic["issues"]
        )
        lines.extend(f"- Revision: {item}" for item in critic["revision_instructions"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = read_jsonl(args.reviewed)
    header = (
        "# Crafter Induced Skill Book v2 Pilot\n\n"
        "Policy content was induced from cited trajectory evidence. Formatting is deterministic; "
        "skills marked revise/reject are not eligible for serving.\n"
    )
    Path(args.out).write_text(
        header + "\n\n" + "\n\n".join(render_result(result) for result in results) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
