#!/usr/bin/env python3
"""Data-conditioned LLM Scientist: propose Hypothesis DSL rules via Trae CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from skill_distill import PROJECT_ROOT
from skill_distill.hypothesis.context import build_proposer_context
from skill_distill.hypothesis.counterexamples import build_counterexample_feedback
from skill_distill.hypothesis.dsl import DSLError, validate_hypothesis
from skill_distill.hypothesis.evaluate import evaluate_hypothesis_on_frame
from skill_distill.trae_backend import TraeCLIBackend


SCHEMA_PATH = PROJECT_ROOT / "skill_distill" / "schemas" / "hypothesis_batch.schema.json"


def build_prompt(context: Dict[str, Any], n_hypotheses: int) -> str:
    return (
        "You are the LLM Scientist in a teacher-knowledge distillation pipeline.\n"
        "Propose executable hypotheses about a trained PPO-AD Crafter teacher.\n\n"
        "Hard rules:\n"
        "1. Output ONLY JSON matching the schema: {\"hypotheses\": [...]}.\n"
        "2. Each hypothesis must use the Hypothesis DSL only (condition + behavior).\n"
        "3. Conditions may only use allowed_fields and ops from CONTEXT.dsl.\n"
        "4. Never use forbidden_fields (seed/episode/absolute coords/future outcomes/etc).\n"
        "5. Do NOT invent Crafter recipe knowledge as if discovered; focus on when/why the "
        "teacher chooses actions, especially around critical examples.\n"
        "6. Freely discover any evidence-supported policy rule. You may predict an immediate "
        "action/action_distribution, a next_interaction_target, or the next observed "
        "achievement. The latter is a fixed-horizon trajectory event, not hidden intent.\n"
        "7. hypothesis_id must be stable snake_case starting with 'hyp_'.\n"
        f"8. Propose exactly {n_hypotheses} hypotheses.\n"
        "9. Do not inspect files or use tools; use only CONTEXT below.\n"
        "10. Ground predicates in the supplied state snapshots. Do not invent precise numeric "
        "thresholds or repeat failure_memory patterns without new discriminating evidence.\n\n"
        "CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )


def build_refine_prompt(
    context: Dict[str, Any],
    evaluated: List[Dict[str, Any]],
    n_hypotheses: int,
    counterexamples: List[Dict[str, Any]] | None = None,
) -> str:
    return (
        "You previously proposed teacher hypotheses. The harness evaluated them.\n"
        "Revise or replace weak hypotheses. Keep strong ones if still valid.\n\n"
        "Hard rules remain the same as before (DSL only, no forbidden fields, JSON schema).\n"
        f"Return exactly {n_hypotheses} hypotheses.\n"
        "Use EVALUATION feedback (low fidelity, zero support, validation errors) to revise.\n"
        "REVISION_COUNTEREXAMPLES are real Decision DB rows from the revision split where "
        "the condition matched but the teacher did NOT follow the predicted behavior. "
        "Use them to tighten/split conditions. Do NOT use acceptance/final_test data.\n"
        "Do not inspect files or use tools.\n\n"
        "CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "\n\nEVALUATION:\n"
        + json.dumps(evaluated, ensure_ascii=False, separators=(",", ":"))
        + "\n\nREVISION_COUNTEREXAMPLES:\n"
        + json.dumps(counterexamples or [], ensure_ascii=False, separators=(",", ":"))
    )


def evaluate_batch(
    hypotheses: List[Dict[str, Any]],
    frame: pd.DataFrame,
    split: str,
) -> List[Dict[str, Any]]:
    results = []
    columns = list(frame.columns)
    for hypo in hypotheses:
        item: Dict[str, Any] = {
            "hypothesis_id": hypo.get("hypothesis_id"),
            "rationale": hypo.get("rationale"),
            "requested_tests": hypo.get("requested_tests") or [],
            "condition": hypo.get("condition"),
            "behavior": hypo.get("behavior"),
        }
        try:
            item["validation"] = validate_hypothesis(hypo, columns=columns)
            item["metrics"] = evaluate_hypothesis_on_frame(hypo, frame, split=split)
            # Compact metrics for refine prompt.
            metrics = item["metrics"]
            item["feedback"] = {
                "support": metrics.get("support"),
                "independent_seeds": metrics.get("independent_seeds"),
                "fidelity": metrics.get("fidelity"),
                "support_action_histogram": metrics.get("support_action_histogram"),
            }
        except DSLError as exc:
            item["error"] = str(exc)
            item["feedback"] = {"error": str(exc)}
        results.append(item)
    return results


def classify_candidate(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return "invalid_dsl"
    metrics = result.get("metrics") or {}
    support = metrics.get("support") or 0
    fidelity = metrics.get("fidelity")
    seeds = metrics.get("independent_seeds") or 0
    if support < 30 or seeds < 5:
        return "insufficient_support"
    if fidelity is None:
        return "unevaluated"
    if fidelity >= 0.75:
        return "high_fidelity_candidate"
    if fidelity >= 0.45:
        return "moderate_fidelity"
    return "low_fidelity"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        default="data/skill_distill/v3/decision_db/decisions.csv",
    )
    parser.add_argument(
        "--critical-bank",
        default="data/skill_distill/v3/critical_states/rollout_bank_wide",
    )
    parser.add_argument(
        "--baseline-report",
        default="data/skill_distill/v3/baselines/mvp_baseline_report.json",
    )
    parser.add_argument(
        "--existing-classified",
        default="",
        help="Optional classified.json from prior rounds to avoid duplicates.",
    )
    parser.add_argument("--out-dir", default="data/skill_distill/v3/hypotheses/round1")
    parser.add_argument("--model", default="GPT-6.1-Sol")
    parser.add_argument("--n-hypotheses", type=int, default=6)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--split", default="exploration")
    parser.add_argument(
        "--evidence-cycle",
        type=int,
        default=0,
        help="Deterministic rotation index for raw evidence examples.",
    )
    parser.add_argument(
        "--evidence-seed",
        type=int,
        default=20261028,
        help="Stable seed used to order rotating evidence reservoirs.",
    )
    parser.add_argument(
        "--counterexample-split",
        default="revision",
        help="Split used to mine counterexamples for refine (not acceptance).",
    )
    parser.add_argument("--log-dir", default="")
    args = parser.parse_args()
    if not 0 <= args.refine_rounds <= 2:
        parser.error("--refine-rounds must be between 0 and 2 to limit revision overfitting")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "audit"

    context = build_proposer_context(
        decisions_csv=Path(args.decisions),
        critical_bank_dir=Path(args.critical_bank),
        baseline_report=Path(args.baseline_report),
        existing_classified=Path(args.existing_classified) if args.existing_classified else None,
        evidence_cycle=args.evidence_cycle,
        evidence_seed=args.evidence_seed,
    )
    (out_dir / "context.json").write_text(json.dumps(context, indent=2) + "\n")

    backend = TraeCLIBackend(
        model=args.model,
        workspace=str(PROJECT_ROOT),
        log_dir=str(log_dir),
    )
    frame = pd.read_csv(args.decisions, low_memory=False)

    prompt = build_prompt(context, n_hypotheses=args.n_hypotheses)
    raw = backend.generate(
        prompt=prompt,
        output_schema=str(SCHEMA_PATH),
        audit_name="propose_round0",
    )
    (out_dir / "raw_round0.json").write_text(json.dumps(raw, indent=2) + "\n")
    hypotheses = raw.get("hypotheses") or []
    evaluated = evaluate_batch(hypotheses, frame, split=args.split)
    for item in evaluated:
        item["class"] = classify_candidate(item)
    (out_dir / "evaluated_round0.json").write_text(json.dumps(evaluated, indent=2) + "\n")

    current = evaluated
    for refine_idx in range(1, args.refine_rounds + 1):
        ce_packs = build_counterexample_feedback(
            current,
            frame,
            split=args.counterexample_split,
            limit_per_hyp=12,
        )
        (out_dir / f"revision_counterexamples_round{refine_idx}.json").write_text(
            json.dumps(ce_packs, indent=2) + "\n"
        )
        refine_prompt = build_refine_prompt(
            context,
            evaluated=[
                {
                    "hypothesis_id": item.get("hypothesis_id"),
                    "class": item.get("class"),
                    "feedback": item.get("feedback"),
                    "condition": item.get("condition"),
                    "behavior": item.get("behavior"),
                    "rationale": item.get("rationale"),
                    "error": item.get("error"),
                }
                for item in current
            ],
            n_hypotheses=args.n_hypotheses,
            counterexamples=ce_packs,
        )
        raw = backend.generate(
            prompt=refine_prompt,
            output_schema=str(SCHEMA_PATH),
            audit_name=f"propose_round{refine_idx}",
        )
        (out_dir / f"raw_round{refine_idx}.json").write_text(json.dumps(raw, indent=2) + "\n")
        hypotheses = raw.get("hypotheses") or []
        current = evaluate_batch(hypotheses, frame, split=args.split)
        for item in current:
            item["class"] = classify_candidate(item)
        (out_dir / f"evaluated_round{refine_idx}.json").write_text(
            json.dumps(current, indent=2) + "\n"
        )

    summary = {
        "model": args.model,
        "n_hypotheses": args.n_hypotheses,
        "refine_rounds": args.refine_rounds,
        "split": args.split,
        "class_counts": {
            key: sum(1 for item in current if item.get("class") == key)
            for key in sorted({item.get("class") for item in current})
        },
        "candidates": [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "class": item.get("class"),
                "fidelity": (item.get("metrics") or {}).get("fidelity"),
                "support": (item.get("metrics") or {}).get("support"),
                "independent_seeds": (item.get("metrics") or {}).get("independent_seeds"),
                "behavior": item.get("behavior"),
                "error": item.get("error"),
            }
            for item in current
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
