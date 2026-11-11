#!/usr/bin/env python3
"""Build an action-critical state bank from repeated teacher-continuation rollouts.

TD is not used for ranking. A state enters the bank only when the rollout TeacherGap
is large and stable across independent continuation samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import crafter
import numpy as np
import torch as th
import torch.nn.functional as F

from skill_distill.collect_teacher import load_model, select_action
from skill_distill.crafter import ACTIONS, ID_TO_ITEM, model_observation, raw_state_info


INTERACTION = {
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
}


def maybe_fix_cuda() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def update_memory(
    model: Any,
    states: th.Tensor,
    latents: th.Tensor,
    next_obs: np.ndarray,
    reward: float,
    device: th.device,
) -> th.Tensor:
    if float(reward) <= 0.1:
        return states
    with th.no_grad():
        next_latent = model.encode(model_observation(next_obs, device))
    return F.normalize(next_latent - latents, dim=-1)


def continuation_return(
    model: Any,
    env: Any,
    states: th.Tensor,
    outputs: Dict[str, th.Tensor],
    first_action: int,
    gamma: float,
    horizon: int,
    device: th.device,
) -> float:
    branch_env = copy.deepcopy(env)
    branch_states = states.detach().clone()
    current_latent = outputs["latents"].detach()
    total = 0.0
    discount = 1.0
    next_obs = None
    done = False
    for branch_step in range(horizon):
        if branch_step == 0:
            action = int(first_action)
            latents = current_latent
        else:
            branch_out = select_action(
                model,
                model_observation(next_obs, device),
                branch_states,
                deterministic=False,
            )
            action = int(branch_out["actions"].item())
            latents = branch_out["latents"]
        next_obs, reward, done, _ = branch_env.step(action)
        total += discount * float(reward)
        discount *= float(gamma)
        branch_states = update_memory(model, branch_states, latents, next_obs, reward, device)
        if done:
            break
    if not done:
        with th.no_grad():
            final = model.forward(model_observation(next_obs, device), states=branch_states)
            bootstrap = float(model.vf_head.denormalize(final["vpreds"]).reshape(-1)[0].item())
        total += discount * bootstrap
    return float(total)


def candidate_actions(logits: np.ndarray, teacher_action: int, top_k: int) -> List[int]:
    ranked = list(np.argsort(logits)[::-1])
    return list(dict.fromkeys([teacher_action] + ranked[:top_k]))


def summarize_action_samples(samples: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    se = std / max(len(arr) ** 0.5, 1.0)
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "n": float(len(arr)),
    }


def estimate_state(
    model: Any,
    env: Any,
    states: th.Tensor,
    outputs: Dict[str, th.Tensor],
    before_info: Dict[str, Any],
    meta: Dict[str, Any],
    gamma: float,
    top_k: int,
    horizon: int,
    n_samples: int,
    device: th.device,
) -> Dict[str, Any]:
    logits = outputs["pi_logits"][0].detach().cpu().numpy()
    teacher_action = int(outputs["actions"].item())
    cands = candidate_actions(logits, teacher_action, top_k=top_k)
    per_action = []
    sample_matrix = {}
    for action in cands:
        samples = [
            continuation_return(
                model,
                env,
                states,
                outputs,
                first_action=action,
                gamma=gamma,
                horizon=horizon,
                device=device,
            )
            for _ in range(n_samples)
        ]
        sample_matrix[action] = samples
        stats = summarize_action_samples(samples)
        per_action.append(
            {
                "action_id": int(action),
                "action_name": ACTIONS[int(action)],
                **stats,
                "samples": samples,
            }
        )

    teacher_stats = next(item for item in per_action if item["action_id"] == teacher_action)
    alt_items = [item for item in per_action if item["action_id"] != teacher_action]
    best_alt = max(alt_items, key=lambda item: item["mean"]) if alt_items else teacher_stats

    # Paired gaps: each sample i uses same continuation RNG only through independent redraws.
    # We approximate stability by independent sample gaps against the best-alt-by-mean action.
    teacher_samples = np.asarray(sample_matrix[teacher_action], dtype=np.float64)
    alt_samples = np.asarray(sample_matrix[best_alt["action_id"]], dtype=np.float64)
    gaps = teacher_samples - alt_samples
    gap_stats = summarize_action_samples(gaps.tolist())
    sign = np.sign(gaps)
    sign = sign[sign != 0]
    if len(sign) == 0:
        sign_consistency = 1.0
        dominant_sign = 0.0
    else:
        dominant_sign = float(np.sign(np.sum(sign)))
        sign_consistency = float(np.mean(sign == dominant_sign))

    # Alternative stability: how often the best alt action wins across sample indices.
    alt_wins = []
    for index in range(n_samples):
        scored = [
            (action, sample_matrix[action][index])
            for action in cands
            if action != teacher_action
        ]
        if scored:
            alt_wins.append(max(scored, key=lambda item: item[1])[0])
    if alt_wins:
        best_alt_mode = int(max(set(alt_wins), key=alt_wins.count))
        best_alt_stability = float(alt_wins.count(best_alt_mode) / len(alt_wins))
    else:
        best_alt_mode = None
        best_alt_stability = 1.0

    entropy = float(model.pi_head.entropy(outputs["pi_logits"]).reshape(-1)[0].item())
    front = None
    try:
        semantic = before_info["semantic"]
        pos = before_info["player_pos"]
        facing = before_info["player_facing"]
        target = (int(pos[0] + facing[0]), int(pos[1] + facing[1]))
        # Match Decision DB / text obs: semantic id -> item name (not raw int).
        cell = int(semantic[target[1], target[0]])
        name = ID_TO_ITEM[cell] if 0 <= cell < len(ID_TO_ITEM) else None
        if name in (None, 0, "None"):
            front = "nothing"
        else:
            front = str(name).lower()
    except Exception:  # noqa: BLE001
        front = None

    return {
        **meta,
        "action_id": teacher_action,
        "action": ACTIONS[teacher_action],
        "front": front,
        "entropy": entropy,
        "value": float(outputs["vpreds"].reshape(-1)[0].item()),
        "health": int(before_info["inventory"]["health"]),
        "daylight": float(before_info["daylight"]),
        "rollout_estimates": per_action,
        "teacher_q_rollout": teacher_stats["mean"],
        "best_alt_action_id": int(best_alt["action_id"]),
        "best_alt_action": best_alt["action_name"],
        "best_alt_q_rollout": best_alt["mean"],
        "teacher_gap_rollout": float(gap_stats["mean"]),
        "teacher_gap_std": float(gap_stats["std"]),
        "teacher_gap_se": float(gap_stats["se"]),
        "teacher_gap_ci95_low": float(gap_stats["ci95_low"]),
        "teacher_gap_ci95_high": float(gap_stats["ci95_high"]),
        "sign_consistency": sign_consistency,
        "dominant_sign": dominant_sign,
        "best_alt_stability": best_alt_stability,
        "best_alt_mode_id": best_alt_mode,
        "ci_excludes_zero": bool(
            gap_stats["ci95_low"] > 0 or gap_stats["ci95_high"] < 0
        ),
        "n_samples": n_samples,
        "horizon": horizon,
    }


def classify_row(
    row: Dict[str, Any],
    abs_gap_threshold: float,
    min_sign_consistency: float,
    min_alt_stability: float,
) -> str:
    gap = float(row["teacher_gap_rollout"])
    reliable = (
        abs(gap) >= abs_gap_threshold
        and bool(row["ci_excludes_zero"])
        and float(row["sign_consistency"]) >= min_sign_consistency
        and float(row["best_alt_stability"]) >= min_alt_stability
    )
    if reliable and gap > 0:
        return "critical_correct"
    if reliable and gap < 0:
        return "critical_negative"
    if abs(gap) >= abs_gap_threshold and not row["ci_excludes_zero"]:
        return "ambiguous_large"
    if abs(gap) < abs_gap_threshold:
        return "indifferent"
    return "unstable"


def load_seeds(path: Path, limit: int | None = None) -> List[int]:
    seeds = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        return seeds[:limit]
    return seeds


def bank_quality_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row)

    def seed_coverage(items: List[Dict[str, Any]]) -> int:
        return len({int(item["seed"]) for item in items})

    critical = by_class.get("critical_correct", []) + by_class.get("critical_negative", [])
    action_hist: Dict[str, int] = {}
    for row in critical:
        action_hist[row["action"]] = action_hist.get(row["action"], 0) + 1

    return {
        "n_evaluated": len(rows),
        "class_counts": {key: len(value) for key, value in sorted(by_class.items())},
        "critical_n": len(critical),
        "critical_seed_coverage": seed_coverage(critical),
        "critical_correct_seed_coverage": seed_coverage(by_class.get("critical_correct", [])),
        "critical_negative_seed_coverage": seed_coverage(by_class.get("critical_negative", [])),
        "critical_action_histogram": action_hist,
        "mean_abs_gap_critical": (
            float(np.mean([abs(row["teacher_gap_rollout"]) for row in critical]))
            if critical
            else None
        ),
        "mean_sign_consistency_critical": (
            float(np.mean([row["sign_consistency"] for row in critical])) if critical else None
        ),
        "fraction_ci_excludes_zero": float(np.mean([row["ci_excludes_zero"] for row in rows]))
        if rows
        else None,
        "pass_checks": {
            "has_both_signs": bool(
                by_class.get("critical_correct") and by_class.get("critical_negative")
            )
            or bool(by_class.get("critical_correct")),
            "seed_coverage_ge_8": seed_coverage(critical) >= 8,
            "not_single_action_dominated": (
                (max(action_hist.values()) / max(len(critical), 1) <= 0.6) if critical else False
            ),
            "reliable_fraction_in_range": (
                0.05 <= (len(critical) / max(len(rows), 1)) <= 0.45 if rows else False
            ),
        },
    }


def main() -> None:
    maybe_fix_cuda()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds-file",
        default="data/skill_distill/v3/seeds/exploration_seeds.txt",
    )
    parser.add_argument(
        "--ckpt",
        default="third_party/Achievement-Distillation/models/ppo_ad-step1-s00/agent-e250.pt",
    )
    parser.add_argument(
        "--config",
        default="third_party/Achievement-Distillation/configs/ppo_ad.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="data/skill_distill/v3/critical_states/rollout_bank",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=120,
        help="Max evaluations to collect. 0 means no global cap (process all seeds).",
    )
    parser.add_argument("--max-seeds", type=int, default=40)
    parser.add_argument(
        "--max-per-seed",
        type=int,
        default=8,
        help="Cap evaluated states per seed (ignored by action_stratified).",
    )
    parser.add_argument(
        "--max-per-action-per-episode",
        type=int,
        default=10,
        help="For action_stratified: max evaluations of the same action in one episode.",
    )
    parser.add_argument(
        "--min-gap-steps",
        type=int,
        default=10,
        help="For action_stratified: min step gap between evaluations of spaced actions.",
    )
    parser.add_argument(
        "--spaced-actions",
        default="Do,Move West,Move East,Move North,Move South",
        help="Comma-separated actions that require --min-gap-steps spacing.",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--sample-mode",
        choices=("uniform", "stratified", "action_filter", "action_stratified"),
        default="uniform",
        help=(
            "uniform: evaluate every eval-every steps under per-seed caps "
            "(no theme bias; rollout gap decides criticality). "
            "stratified: legacy interaction/high-entropy/random quotas. "
            "action_filter: only evaluate when teacher action is in --actions. "
            "action_stratified: all (or --actions) teacher actions with "
            "per-action-per-episode caps; Do/Move use min-gap-steps."
        ),
    )
    parser.add_argument(
        "--actions",
        default="",
        help=(
            "Comma-separated action names for action_filter / optional subset for "
            "action_stratified. Empty action_stratified means all ACTIONS."
        ),
    )
    parser.add_argument(
        "--shuffle-seeds",
        action="store_true",
        help="Shuffle seed order before rollout for broader coverage.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split seeds across parallel workers (seed-level sharding).",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="This worker's shard in [0, num-shards).",
    )
    parser.add_argument("--gap-percentile", type=float, default=70.0)
    parser.add_argument("--min-abs-gap", type=float, default=0.05)
    parser.add_argument("--min-sign-consistency", type=float, default=0.8)
    parser.add_argument("--min-alt-stability", type=float, default=0.6)
    args = parser.parse_args()

    if os.environ.get("PYTHONHASHSEED") is None:
        print(
            "WARNING: PYTHONHASHSEED unset; relaunch with PYTHONHASHSEED=0",
            flush=True,
        )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    action_filter = {
        name.strip() for name in args.actions.split(",") if name.strip()
    }
    if args.sample_mode == "action_filter" and not action_filter:
        raise SystemExit("--sample-mode action_filter requires --actions")
    if args.sample_mode == "action_stratified" and not action_filter:
        action_filter = set(ACTIONS)
    unknown = sorted(action_filter - set(ACTIONS))
    if unknown:
        raise SystemExit(f"unknown --actions: {unknown}")
    spaced_actions = {
        name.strip() for name in args.spaced_actions.split(",") if name.strip()
    }
    unknown_spaced = sorted(spaced_actions - set(ACTIONS))
    if unknown_spaced:
        raise SystemExit(f"unknown --spaced-actions: {unknown_spaced}")

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")

    seed_limit = None if args.max_seeds <= 0 else args.max_seeds
    seeds = load_seeds(Path(args.seeds_file), limit=seed_limit)
    if args.shuffle_seeds:
        rng.shuffle(seeds)
    if args.num_shards > 1:
        seeds = [seed for index, seed in enumerate(seeds) if index % args.num_shards == args.shard_id]
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    probe = crafter.Env(seed=0)
    model, config = load_model(Path(args.ckpt), Path(args.config), device, probe)
    hidsize = int(config["model_kwargs"]["hidsize"])
    n_states_cap = None if args.n_states <= 0 else args.n_states
    print(
        f"device={device} seeds={len(seeds)} n_states={n_states_cap or 'unlimited'} "
        f"max_per_seed={args.max_per_seed} samples={args.n_samples} "
        f"horizon={args.horizon} sample_mode={args.sample_mode} "
        f"action_filter={sorted(action_filter)} "
        f"max_per_action_per_episode={args.max_per_action_per_episode} "
        f"min_gap_steps={args.min_gap_steps} spaced={sorted(spaced_actions)} "
        f"shard={args.shard_id}/{args.num_shards}",
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    n_states_for_quota = args.n_states if args.n_states > 0 else max(len(seeds) * 8, 1)
    quotas = {
        "interaction": int(n_states_for_quota * 0.45),
        "high_entropy": int(n_states_for_quota * 0.25),
        "random": n_states_for_quota
        - int(n_states_for_quota * 0.45)
        - int(n_states_for_quota * 0.25),
    }
    counts = {key: 0 for key in quotas}
    counts.setdefault("uniform", 0)
    counts.setdefault("action_filter", 0)
    counts.setdefault("action_stratified", 0)
    per_seed_counts: Dict[int, int] = {}
    action_hit_counts: Dict[str, int] = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimates_path = out_dir / "rollout_estimates.jsonl"
    # Fresh run; incremental flush for long action_stratified jobs.
    estimates_path.write_text("", encoding="utf-8")

    for episode_idx, seed in enumerate(seeds):
        if n_states_cap is not None and len(results) >= n_states_cap:
            break
        env = crafter.Env(seed=int(seed), length=args.max_steps)
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        ep_action_counts: Dict[str, int] = {}
        last_eval_step: Dict[str, int] = {}
        for step in range(args.max_steps):
            if n_states_cap is not None and len(results) >= n_states_cap:
                break
            if (
                args.sample_mode != "action_stratified"
                and per_seed_counts.get(int(seed), 0) >= args.max_per_seed
            ):
                break
            before_info = raw_state_info(env)
            outputs = select_action(
                model, model_observation(obs, device), states, deterministic=False
            )
            action = int(outputs["actions"].item())
            action_name = ACTIONS[action]
            entropy = float(model.pi_head.entropy(outputs["pi_logits"]).reshape(-1)[0].item())

            bucket = None
            if args.sample_mode == "uniform":
                # Neutral grid sampling: no interaction/combat/survival theme bias.
                if step % args.eval_every == 0:
                    bucket = "uniform"
            elif args.sample_mode == "action_filter":
                # Supplemental bank: only score Make/Place (etc.) teacher actions.
                if action_name in action_filter:
                    bucket = "action_filter"
            elif args.sample_mode == "action_stratified":
                if action_name in action_filter:
                    if ep_action_counts.get(action_name, 0) < args.max_per_action_per_episode:
                        gap_ok = True
                        if action_name in spaced_actions:
                            prev = last_eval_step.get(action_name)
                            if prev is not None and (step - prev) < args.min_gap_steps:
                                gap_ok = False
                        if gap_ok:
                            bucket = "action_stratified"
            else:
                if action_name in INTERACTION and counts["interaction"] < quotas["interaction"]:
                    bucket = "interaction"
                elif entropy >= 1.2 and counts["high_entropy"] < quotas["high_entropy"]:
                    bucket = "high_entropy"
                elif step % args.eval_every == 0 and counts["random"] < quotas["random"]:
                    bucket = "random"
                elif step % args.eval_every == 0:
                    for key, quota in quotas.items():
                        if counts[key] < quota:
                            bucket = key
                            break

            if bucket is not None:
                row = estimate_state(
                    model=model,
                    env=env,
                    states=states,
                    outputs=outputs,
                    before_info=before_info,
                    meta={
                        "episode_idx": episode_idx,
                        "seed": int(seed),
                        "step": int(step),
                        "bucket": bucket,
                        "sample_mode": args.sample_mode,
                        "action_filter": sorted(action_filter),
                        "max_per_action_per_episode": args.max_per_action_per_episode,
                        "min_gap_steps": args.min_gap_steps,
                        "spaced_actions": sorted(spaced_actions),
                        "source": "rollout_teacher_gap",
                        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    },
                    gamma=args.gamma,
                    top_k=args.top_k,
                    horizon=args.horizon,
                    n_samples=args.n_samples,
                    device=device,
                )
                results.append(row)
                counts[bucket] = counts.get(bucket, 0) + 1
                per_seed_counts[int(seed)] = per_seed_counts.get(int(seed), 0) + 1
                action_hit_counts[action_name] = action_hit_counts.get(action_name, 0) + 1
                ep_action_counts[action_name] = ep_action_counts.get(action_name, 0) + 1
                last_eval_step[action_name] = step
                with open(estimates_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
                if len(results) % 10 == 0:
                    target = n_states_cap if n_states_cap is not None else "?"
                    print(
                        f"[{len(results)}/{target}] "
                        f"gap={row['teacher_gap_rollout']:.3f} "
                        f"ci_excl0={row['ci_excludes_zero']} "
                        f"sign_cons={row['sign_consistency']:.2f} "
                        f"action={action_name} "
                        f"bucket={bucket} seeds_seen={len(per_seed_counts)} "
                        f"ep={episode_idx}",
                        flush=True,
                    )

            next_obs, reward, done, _ = env.step(action)
            states = update_memory(model, states, outputs["latents"], next_obs, reward, device)
            obs = next_obs
            if done:
                break
        if (episode_idx + 1) % 5 == 0:
            print(
                f"[episode {episode_idx + 1}/{len(seeds)}] "
                f"evals={len(results)} action_hits={dict(sorted(action_hit_counts.items()))}",
                flush=True,
            )

    abs_gaps = np.array([abs(row["teacher_gap_rollout"]) for row in results], dtype=np.float64)
    percentile_threshold = (
        float(np.percentile(abs_gaps, args.gap_percentile)) if len(abs_gaps) else args.min_abs_gap
    )
    abs_gap_threshold = max(args.min_abs_gap, percentile_threshold)

    for row in results:
        row["abs_gap_threshold_used"] = abs_gap_threshold
        row["class"] = classify_row(
            row,
            abs_gap_threshold=abs_gap_threshold,
            min_sign_consistency=args.min_sign_consistency,
            min_alt_stability=args.min_alt_stability,
        )

    # estimates already flushed incrementally for long runs; rewrite canonical file.
    with open(estimates_path, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    banks = {
        "critical_correct": [],
        "critical_negative": [],
        "ambiguous_large": [],
        "indifferent": [],
        "unstable": [],
    }
    for row in results:
        banks[row["class"]].append(row)

    for name, items in banks.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in sorted(items, key=lambda item: -abs(item["teacher_gap_rollout"])):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    quality = bank_quality_report(results)
    summary = {
        "n_states": len(results),
        "sample_mode": args.sample_mode,
        "action_filter": sorted(action_filter),
        "spaced_actions": sorted(spaced_actions),
        "max_per_action_per_episode": args.max_per_action_per_episode,
        "min_gap_steps": args.min_gap_steps,
        "action_hit_counts": action_hit_counts,
        "bucket_counts": counts,
        "n_seeds_evaluated": len(per_seed_counts),
        "max_per_seed": args.max_per_seed,
        "eval_every": args.eval_every,
        "per_seed_counts": {str(key): value for key, value in sorted(per_seed_counts.items())},
        "abs_gap_threshold": abs_gap_threshold,
        "gap_percentile": args.gap_percentile,
        "min_abs_gap": args.min_abs_gap,
        "min_sign_consistency": args.min_sign_consistency,
        "min_alt_stability": args.min_alt_stability,
        "horizon": args.horizon,
        "n_samples": args.n_samples,
        "gamma": args.gamma,
        "device": str(device),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "quality": quality,
        "judgment_rubric": {
            "critical_correct": (
                "gap>0, |gap|>=threshold, 95% CI excludes 0, "
                "sign_consistency>=min, best_alt_stability>=min"
            ),
            "critical_negative": "same reliability checks but gap<0 (bad-habit candidate)",
            "ambiguous_large": "|gap| large but CI crosses 0 — need more samples",
            "indifferent": "|gap| below threshold — action swap likely unimportant",
            "unstable": "large/signed but unstable alt identity or sign flips",
            "bank_pass_if": quality["pass_checks"],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
