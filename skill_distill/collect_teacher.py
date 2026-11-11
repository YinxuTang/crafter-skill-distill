#!/usr/bin/env python3
"""Collect full, text-aligned trajectories from a trained PPO-AD Crafter policy."""

import argparse
import copy
import json
import os
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import crafter
import gym
import numpy as np
import torch as th
import torch.nn.functional as F
import yaml

from achievement_distillation.model import PPOADModel

from skill_distill import AD_ROOT, PROJECT_ROOT
from skill_distill.crafter import (
    ACTIONS,
    augment_step_info,
    compact_state,
    jsonable,
    model_observation,
    observation_hash,
    raw_state_info,
    text_observation,
    write_jsonl,
)


def load_seeds(path: str) -> List[int]:
    with open(path, encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def load_model(
    checkpoint: Path, config_path: Path, device: th.device, probe_env: Any
) -> tuple:
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    obs_shape = probe_env.observation_space.shape
    observation_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
        dtype=np.float32,
    )
    model = PPOADModel(
        observation_space=observation_space,
        action_space=probe_env.action_space,
        **config["model_kwargs"],
    ).to(device)
    try:
        state_dict = th.load(str(checkpoint), map_location=device, weights_only=True)
    except TypeError:
        state_dict = th.load(str(checkpoint), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def select_action(model: Any, obs: th.Tensor, states: th.Tensor, deterministic: bool) -> Dict[str, th.Tensor]:
    with th.no_grad():
        outputs = model.forward(obs, states=states)
        actions = model.pi_head.sample(outputs["pi_logits"], deterministic=deterministic)
        log_probs = model.pi_head.log_prob(outputs["pi_logits"], actions)
        outputs["actions"] = actions
        outputs["log_probs"] = log_probs
        outputs["vpreds"] = model.vf_head.denormalize(outputs["vpreds"])
    return outputs


def counterfactual_forks(
    model: Any,
    env_snapshot: Any,
    before_info: Dict[str, Any],
    outputs: Dict[str, th.Tensor],
    states: th.Tensor,
    device: th.device,
    horizon: int,
    top_k: int = 3,
    bottom_k: int = 1,
) -> List[Dict[str, Any]]:
    logits = outputs["pi_logits"][0].detach().cpu().numpy()
    ranked = list(np.argsort(logits)[::-1])
    actual = int(outputs["actions"].item())
    candidates = list(dict.fromkeys(ranked[:top_k] + ranked[-bottom_k:] + [actual]))
    before_achievements = before_info["achievements"]
    before_health = int(before_info["inventory"]["health"])
    branches = []
    for candidate in candidates:
        branch_env = copy.deepcopy(env_snapshot)
        branch_states = states.detach().clone()
        current_latent = outputs["latents"].detach()
        actions = []
        total_reward = 0.0
        done = False
        final_info = before_info
        next_obs = None
        for branch_step in range(horizon):
            if branch_step == 0:
                action = int(candidate)
            else:
                branch_outputs = select_action(
                    model,
                    model_observation(next_obs, device),
                    branch_states,
                    deterministic=True,
                )
                action = int(branch_outputs["actions"].item())
                current_latent = branch_outputs["latents"].detach()
            next_obs, reward, done, final_info = branch_env.step(action)
            actions.append(action)
            total_reward += float(reward)
            if float(reward) > 0.1:
                with th.no_grad():
                    next_latent = model.encode(model_observation(next_obs, device))
                branch_states = F.normalize(next_latent - current_latent, dim=-1)
            if done:
                break
        new_achievements = sorted(
            key
            for key, value in final_info["achievements"].items()
            if int(value) > int(before_achievements.get(key, 0))
        )
        branches.append(
            {
                "first_action_id": int(candidate),
                "first_action_name": ACTIONS[int(candidate)],
                "is_teacher_action": int(candidate) == actual,
                "teacher_logit": float(logits[int(candidate)]),
                "actions": [int(action) for action in actions],
                "action_names": [ACTIONS[int(action)] for action in actions],
                "score": {
                    "reward": total_reward,
                    "new_achievements": new_achievements,
                    "achievement_gain": len(new_achievements),
                    "health_delta": int(final_info["inventory"]["health"]) - before_health,
                    "survived": not bool(done and int(final_info["inventory"]["health"]) <= 0),
                },
            }
        )
    return branches


def collect_episode(
    model: Any,
    seed: int,
    episode_idx: int,
    device: th.device,
    hidsize: int,
    max_steps: int,
    deterministic: bool,
    include_daynight: bool,
    save_pixels: bool,
    fork_horizon: int,
) -> tuple:
    env = crafter.Env(seed=int(seed), length=max_steps)
    obs = env.reset()
    states = th.zeros(1, hidsize, device=device)
    records: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    action_counts: Counter = Counter()
    episode_return = 0.0

    for step in range(max_steps):
        before_info = raw_state_info(env)
        env_snapshot = copy.deepcopy(env) if fork_horizon > 0 else None
        before_state = compact_state(before_info)
        before_text = text_observation(before_info, include_daynight=include_daynight)
        obs_tensor = model_observation(obs, device)
        outputs = select_action(model, obs_tensor, states, deterministic)
        action = int(outputs["actions"].item())
        pi_logits_tensor = outputs["pi_logits"][0].detach()
        argmax_action = int(th.argmax(pi_logits_tensor).item())
        pi_logits = pi_logits_tensor.cpu().tolist()
        value = float(outputs["vpreds"].reshape(-1)[0].item())
        log_prob = float(outputs["log_probs"].reshape(-1)[0].item())
        entropy = float(model.pi_head.entropy(outputs["pi_logits"]).reshape(-1)[0].item())
        latent = outputs["latents"]

        next_obs, reward, done, step_info = env.step(action)
        after_info = augment_step_info(env, step_info, before_info)
        after_state = compact_state(after_info)
        after_text = text_observation(after_info, include_daynight=include_daynight)
        episode_return += float(reward)
        action_counts[ACTIONS[action]] += 1
        should_fork = bool(
            after_info["unlocked"]
            or after_info["inventory_delta"]
            or ACTIONS[action].startswith(("Make ", "Place "))
            or ACTIONS[action] in ("Do", "Sleep")
        )
        forks = (
            counterfactual_forks(
                model,
                env_snapshot,
                before_info,
                outputs,
                states,
                device,
                horizon=fork_horizon,
            )
            if fork_horizon > 0 and should_fork
            else []
        )

        records.append(
            {
                "schema_version": 3,
                "episode_id": f"ep_{episode_idx:05d}_seed_{seed}",
                "episode_idx": episode_idx,
                "seed": int(seed),
                "step": step,
                "obs_hash": observation_hash(obs),
                "action_id": action,
                "action_name": ACTIONS[action],
                "argmax_action_id": argmax_action,
                "argmax_action_name": ACTIONS[argmax_action],
                "reward": float(reward),
                "done": bool(done),
                "teacher": {
                    "sampling": "deterministic" if deterministic else "stochastic",
                    "pi_logits": pi_logits,
                    "log_prob": log_prob,
                    "value": value,
                    "entropy": entropy,
                    "memory_norm": float(th.linalg.vector_norm(states).item()),
                    "selected_is_argmax": action == argmax_action,
                },
                "before": before_state,
                "after": after_state,
                "text_obs": before_text,
                "text_obs_after": after_text,
                "unlocked": sorted(after_info["unlocked"]),
                "inventory_delta": jsonable(after_info["inventory_delta"]),
                "counterfactuals": forks,
            }
        )
        if save_pixels:
            frames.append(np.asarray(obs, dtype=np.uint8))

        if float(reward) > 0.1:
            with th.no_grad():
                next_latent = model.encode(model_observation(next_obs, device))
            states = F.normalize(next_latent - latent, dim=-1)
        obs = next_obs
        if done:
            break

    summary = {
        "episode_id": f"ep_{episode_idx:05d}_seed_{seed}",
        "episode_idx": episode_idx,
        "seed": int(seed),
        "num_steps": len(records),
        "episode_return": episode_return,
        "action_frequency": dict(action_counts),
        "achievements": records[-1]["after"]["achievements"] if records else {},
        "progression": (
            sum(value > 0 for value in records[-1]["after"]["achievements"].values()) / 22.0
            if records
            else 0.0
        ),
        "done": bool(records and records[-1]["done"]),
    }
    pixel_array = np.stack(frames) if frames else None
    return records, pixel_array, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=str(AD_ROOT / "models" / "ppo_ad-step1-s00" / "agent-e250.pt"),
    )
    parser.add_argument("--config", default=str(AD_ROOT / "configs" / "ppo_ad.yaml"))
    parser.add_argument("--seeds-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--include-daynight", action="store_true")
    parser.add_argument("--no-pixels", action="store_true")
    parser.add_argument(
        "--fork-horizon",
        type=int,
        default=0,
        help="Same-process deterministic branch rollout horizon; 0 disables forks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(args.seeds_file)
    if args.limit:
        seeds = seeds[: args.limit]

    if os.environ.get("PYTHONHASHSEED") is None:
        print(
            "WARNING: PYTHONHASHSEED is unset. Crafter set-iteration spawn/despawn "
            "makes trajectories non-reproducible across processes. "
            "Relaunch with PYTHONHASHSEED=0 for prefix-replayable logs.",
            flush=True,
        )

    random.seed(args.sampling_seed)
    np.random.seed(args.sampling_seed)
    th.manual_seed(args.sampling_seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.sampling_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    probe_env = crafter.Env(seed=seeds[0], length=args.max_steps)
    model, config = load_model(Path(args.ckpt), Path(args.config), device, probe_env)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "seeds_file": str(Path(args.seeds_file).expanduser().resolve()),
        "seeds": seeds,
        "sampling_seed": args.sampling_seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "deterministic": args.deterministic,
        "sampling": "deterministic" if args.deterministic else "stochastic",
        "max_steps": args.max_steps,
        "save_pixels": not args.no_pixels,
        "fork_horizon": args.fork_horizon,
        "include_daynight": args.include_daynight,
        "text_observation": "BALROG obs-A bearing, unique_items=True, world_view=9x7",
        "schema_version": 3,
        "actions": ACTIONS,
        "device": str(device),
        "python": platform.python_version(),
        "torch": th.__version__,
        "project_root": str(PROJECT_ROOT),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    summaries = []
    for episode_idx, seed in enumerate(seeds):
        records, pixels, summary = collect_episode(
            model=model,
            seed=seed,
            episode_idx=episode_idx,
            device=device,
            hidsize=int(config["model_kwargs"]["hidsize"]),
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            include_daynight=args.include_daynight,
            save_pixels=not args.no_pixels,
            fork_horizon=args.fork_horizon,
        )
        stem = summary["episode_id"]
        write_jsonl(str(episodes_dir / f"{stem}.jsonl"), records)
        if pixels is not None:
            np.savez_compressed(episodes_dir / f"{stem}.npz", obs=pixels)
        with open(episodes_dir / f"{stem}.summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        summaries.append(summary)
        print(
            f"[{episode_idx + 1}/{len(seeds)}] seed={seed} "
            f"steps={summary['num_steps']} progression={summary['progression']:.3f}",
            flush=True,
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump({"episodes": summaries}, handle, indent=2)


if __name__ == "__main__":
    main()
