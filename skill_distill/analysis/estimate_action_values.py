#!/usr/bin/env python3
"""Estimate action values, calibrate one-step TD, and rank critical states.

Important: Crafter object-spawn/despawn iterates Python sets, so trajectories are
only reproducible under a fixed PYTHONHASHSEED. Logged exploration episodes were
collected without a fixed hash seed, so prefix replay does not restore the same
states. This script therefore evaluates TeacherGap on live teacher rollouts.
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
import pandas as pd
import torch as th
import torch.nn.functional as F
from scipy.stats import spearmanr

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


def one_step_td(
    model: Any,
    env: Any,
    states: th.Tensor,
    outputs: Dict[str, th.Tensor],
    action: int,
    gamma: float,
    device: th.device,
) -> Dict[str, Any]:
    branch_env = copy.deepcopy(env)
    branch_states = states.detach().clone()
    next_obs, reward, done, info = branch_env.step(int(action))
    next_states = update_memory(
        model, branch_states, outputs["latents"], next_obs, reward, device
    )
    with th.no_grad():
        if done:
            next_value = 0.0
        else:
            next_out = model.forward(model_observation(next_obs, device), states=next_states)
            next_value = float(model.vf_head.denormalize(next_out["vpreds"]).reshape(-1)[0].item())
    return {
        "action_id": int(action),
        "action_name": ACTIONS[int(action)],
        "reward": float(reward),
        "done": bool(done),
        "next_value": next_value,
        "q_td": float(reward) + float(gamma) * next_value,
        "health": int(info["inventory"]["health"]),
    }


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
        next_obs, reward, done, info = branch_env.step(action)
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


def evaluate_live_state(
    model: Any,
    env: Any,
    obs: np.ndarray,
    states: th.Tensor,
    outputs: Dict[str, th.Tensor],
    before_info: Dict[str, Any],
    meta: Dict[str, Any],
    gamma: float,
    top_k: int,
    calibrate: bool,
    rollout_horizon: int,
    rollout_samples: int,
    device: th.device,
) -> Dict[str, Any]:
    logits = outputs["pi_logits"][0].detach().cpu().numpy()
    teacher_action = int(outputs["actions"].item())
    cands = candidate_actions(logits, teacher_action, top_k=top_k)
    td_rows = [
        one_step_td(model, env, states, outputs, action, gamma, device) for action in cands
    ]
    td_by_action = {item["action_id"]: item for item in td_rows}
    teacher_q = td_by_action[teacher_action]["q_td"]
    alt_q = max(
        (item["q_td"] for action, item in td_by_action.items() if action != teacher_action),
        default=teacher_q,
    )
    result: Dict[str, Any] = {
        **meta,
        "action_id": teacher_action,
        "action": ACTIONS[teacher_action],
        "front": meta.get("front"),
        "entropy": float(model.pi_head.entropy(outputs["pi_logits"]).reshape(-1)[0].item()),
        "value": float(outputs["vpreds"].reshape(-1)[0].item()),
        "teacher_q_td": teacher_q,
        "best_alt_q_td": alt_q,
        "teacher_gap_td": teacher_q - alt_q,
        "td_estimates": td_rows,
        "calibrated": False,
        "health": int(before_info["inventory"]["health"]),
        "daylight": float(before_info["daylight"]),
    }
    if calibrate:
        rollout_stats = []
        for action in cands:
            samples = [
                continuation_return(
                    model,
                    env,
                    states,
                    outputs,
                    first_action=action,
                    gamma=gamma,
                    horizon=rollout_horizon,
                    device=device,
                )
                for _ in range(rollout_samples)
            ]
            rollout_stats.append(
                {
                    "action_id": int(action),
                    "action_name": ACTIONS[int(action)],
                    "q_rollout_mean": float(np.mean(samples)),
                    "q_rollout_std": float(np.std(samples)),
                    "n_samples": int(rollout_samples),
                }
            )
        roll_by_action = {item["action_id"]: item for item in rollout_stats}
        teacher_qr = roll_by_action[teacher_action]["q_rollout_mean"]
        alt_qr = max(
            (
                item["q_rollout_mean"]
                for action, item in roll_by_action.items()
                if action != teacher_action
            ),
            default=teacher_qr,
        )
        result.update(
            {
                "calibrated": True,
                "rollout_estimates": rollout_stats,
                "teacher_q_rollout": teacher_qr,
                "best_alt_q_rollout": alt_qr,
                "teacher_gap_rollout": teacher_qr - alt_qr,
            }
        )
    return result


def load_seeds(path: Path, limit: int | None = None) -> List[int]:
    seeds = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if limit is not None:
        return seeds[:limit]
    return seeds


def summarize_calibration(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    calibrated = [row for row in rows if row.get("calibrated")]
    if len(calibrated) < 3:
        return {"n": len(calibrated), "error": "too few calibrated states"}
    td_gap = np.array([row["teacher_gap_td"] for row in calibrated], dtype=np.float64)
    roll_gap = np.array([row["teacher_gap_rollout"] for row in calibrated], dtype=np.float64)
    td_best = np.array(
        [max(row["td_estimates"], key=lambda item: item["q_td"])["action_id"] for row in calibrated]
    )
    roll_best = np.array(
        [
            max(row["rollout_estimates"], key=lambda item: item["q_rollout_mean"])["action_id"]
            for row in calibrated
        ]
    )
    corr = spearmanr(td_gap, roll_gap)
    return {
        "n": int(len(calibrated)),
        "spearman_gap": float(corr.correlation) if corr.correlation is not None else None,
        "spearman_pvalue": float(corr.pvalue) if corr.pvalue is not None else None,
        "sign_agreement": float(np.mean(np.sign(td_gap) == np.sign(roll_gap))),
        "best_action_agreement": float(np.mean(td_best == roll_best)),
        "mean_td_gap": float(td_gap.mean()),
        "mean_rollout_gap": float(roll_gap.mean()),
        "td_negative_rate": float(np.mean(td_gap < 0)),
        "rollout_negative_rate": float(np.mean(roll_gap < 0)),
    }


def main() -> None:
    maybe_fix_cuda()
    if os.environ.get("PYTHONHASHSEED") is None:
        print(
            "WARNING: PYTHONHASHSEED is unset; Crafter spawn/despawn is not "
            "stable across processes. Relaunch with PYTHONHASHSEED=0.",
            flush=True,
        )
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
    parser.add_argument("--out-dir", default="data/skill_distill/v3/critical_states")
    parser.add_argument("--n-states", type=int, default=300)
    parser.add_argument("--n-calibrate", type=int, default=80)
    parser.add_argument("--max-seeds", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--rollout-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critical-percentile", type=float, default=80.0)
    parser.add_argument("--eval-every", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    seeds = load_seeds(Path(args.seeds_file), limit=args.max_seeds)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    probe = crafter.Env(seed=0)
    model, config = load_model(Path(args.ckpt), Path(args.config), device, probe)
    hidsize = int(config["model_kwargs"]["hidsize"])
    print(
        f"device={device} seeds={len(seeds)} target_states={args.n_states} "
        f"calibrate={args.n_calibrate} hash_seed={os.environ.get('PYTHONHASHSEED')}",
        flush=True,
    )

    results: List[Dict[str, Any]] = []
    calibrate_budget = args.n_calibrate
    quotas = {
        "interaction": int(args.n_states * 0.4),
        "high_entropy": int(args.n_states * 0.3),
        "random": args.n_states - int(args.n_states * 0.4) - int(args.n_states * 0.3),
    }
    counts = {key: 0 for key in quotas}

    for episode_idx, seed in enumerate(seeds):
        if len(results) >= args.n_states:
            break
        env = crafter.Env(seed=int(seed), length=args.max_steps)
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        for step in range(args.max_steps):
            if len(results) >= args.n_states:
                break
            before_info = raw_state_info(env)
            outputs = select_action(
                model, model_observation(obs, device), states, deterministic=False
            )
            action = int(outputs["actions"].item())
            action_name = ACTIONS[action]
            entropy = float(model.pi_head.entropy(outputs["pi_logits"]).reshape(-1)[0].item())

            bucket = None
            if action_name in INTERACTION and counts["interaction"] < quotas["interaction"]:
                bucket = "interaction"
            elif entropy >= 1.2 and counts["high_entropy"] < quotas["high_entropy"]:
                bucket = "high_entropy"
            elif step % args.eval_every == 0 and counts["random"] < quotas["random"]:
                bucket = "random"
            elif (
                len(results) < args.n_states
                and step % args.eval_every == 0
                and sum(counts.values()) < args.n_states
            ):
                # Fill whatever quota remains.
                for key, quota in quotas.items():
                    if counts[key] < quota:
                        bucket = key
                        break

            if bucket is not None:
                calibrate = calibrate_budget > 0 and (
                    bucket == "interaction" or rng.random() < 0.35
                )
                if calibrate:
                    calibrate_budget -= 1
                front = None
                # Lightweight front object from semantic local cell if available.
                try:
                    semantic = before_info["semantic"]
                    pos = before_info["player_pos"]
                    facing = before_info["player_facing"]
                    target = (int(pos[0] + facing[0]), int(pos[1] + facing[1]))
                    cell = int(semantic[target[1], target[0]])
                    name = ID_TO_ITEM[cell] if 0 <= cell < len(ID_TO_ITEM) else None
                    if name in (None, 0, "None"):
                        front = "nothing"
                    else:
                        front = str(name).lower()
                except Exception:  # noqa: BLE001
                    front = None
                result = evaluate_live_state(
                    model=model,
                    env=env,
                    obs=obs,
                    states=states,
                    outputs=outputs,
                    before_info=before_info,
                    meta={
                        "episode_idx": episode_idx,
                        "seed": int(seed),
                        "step": int(step),
                        "bucket": bucket,
                        "front": front,
                        "source": "live_teacher_rollout",
                        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    },
                    gamma=args.gamma,
                    top_k=args.top_k,
                    calibrate=calibrate,
                    rollout_horizon=args.rollout_horizon,
                    rollout_samples=args.rollout_samples,
                    device=device,
                )
                results.append(result)
                counts[bucket] += 1
                if len(results) % 10 == 0:
                    print(
                        f"[{len(results)}/{args.n_states}] "
                        f"gap_td={result['teacher_gap_td']:.4f} "
                        f"bucket={bucket} cal={calibrate}",
                        flush=True,
                    )

            next_obs, reward, done, _ = env.step(action)
            states = update_memory(model, states, outputs["latents"], next_obs, reward, device)
            obs = next_obs
            if done:
                break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "action_value_estimates.jsonl", "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    valid = [row for row in results if "teacher_gap_td" in row]
    calibration = summarize_calibration(results)
    if valid:
        gaps = np.array([abs(row["teacher_gap_td"]) for row in valid], dtype=np.float64)
        threshold = float(np.percentile(gaps, args.critical_percentile))
        critical = [row for row in valid if abs(row["teacher_gap_td"]) >= threshold]
    else:
        threshold = None
        critical = []

    summary = {
        "n_requested": args.n_states,
        "n_evaluated": len(results),
        "n_valid": len(valid),
        "bucket_counts": counts,
        "device": str(device),
        "gamma": args.gamma,
        "top_k": args.top_k,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "source": "live_teacher_rollout",
        "note": (
            "Logged exploration trajectories are not prefix-reproducible without a fixed "
            "PYTHONHASHSEED; critical states are estimated on live rollouts instead."
        ),
        "calibration": calibration,
        "critical_percentile": args.critical_percentile,
        "critical_abs_gap_threshold": threshold,
        "n_critical": len(critical),
        "gap_td_mean": float(np.mean([row["teacher_gap_td"] for row in valid])) if valid else None,
        "gap_td_negative_rate": (
            float(np.mean([row["teacher_gap_td"] < 0 for row in valid])) if valid else None
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out_dir / "critical_states.jsonl", "w", encoding="utf-8") as handle:
        for row in sorted(critical, key=lambda item: -abs(item["teacher_gap_td"])):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
