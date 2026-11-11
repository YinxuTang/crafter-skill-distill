#!/usr/bin/env python3
"""Formal Crafter eval for AD teacher checkpoints.

Stock eval.py only rolls out 1 episode for a video. This script runs N episodes
and reports per-achievement success rates + official Crafter geometric score.

Also supports --seeds_file for matched comparisons against BALROG LLM runs
(one integer seed per line; must match crafter.Env constructor seed).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch as th
import torch.nn.functional as F
import yaml
from crafter.env import Env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from achievement_distillation.algorithm import *  # noqa: F401,F403
from achievement_distillation.constant import TASKS
from achievement_distillation.model import *  # noqa: F401,F403
from achievement_distillation.wrapper import VecPyTorch


def crafter_score(success_rate_pct: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(1.0 + success_rate_pct))) - 1.0)


def balrog_progression(success_vec: np.ndarray) -> float:
    """BALROG crafter progression = fraction of 22 achievements unlocked."""
    return float(success_vec.sum() / len(TASKS))


def run_episode(model, venv, hidsize: int, device: th.device, max_steps: int = 10000):
    """Returns success vector shape [22] (0/1) for one episode."""
    model.eval()
    obs = venv.reset()
    states = th.zeros(1, hidsize, device=device)
    last_successes = np.zeros(len(TASKS), dtype=np.int32)
    for _ in range(max_steps):
        with th.no_grad():
            outputs = model.act(obs, states=states)
        latents = outputs["latents"]
        actions = outputs["actions"]
        obs, rewards, dones, infos = venv.step(actions)
        last_successes = infos["successes"][0].detach().cpu().numpy().astype(np.int32)
        if dones.any():
            break
        if (rewards > 0.1).any():
            with th.no_grad():
                next_latents = model.encode(obs)
            states = F.normalize(next_latents - latents, dim=-1)
    return last_successes


def load_seeds(path: str):
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(int(line))
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo_ad")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to agent-eXXX.pt")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Base seed if --seeds_file not set")
    parser.add_argument("--seeds_file", type=str, default="", help="One integer seed per line")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    with open(f"configs/{args.exp_name}.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.seeds_file:
        ep_seeds = load_seeds(args.seeds_file)
        args.num_episodes = len(ep_seeds)
    else:
        ep_seeds = [int(args.seed + ep + 1) for ep in range(args.num_episodes)]

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    probe_env = Env(seed=ep_seeds[0])
    probe_venv = VecPyTorch(DummyVecEnv([lambda: probe_env]), device=device)

    model_cls = getattr(sys.modules[__name__], config["model_cls"])
    model = model_cls(
        observation_space=probe_venv.observation_space,
        action_space=probe_venv.action_space,
        **config["model_kwargs"],
    ).to(device)
    state_dict = th.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded {args.ckpt} on {device}", flush=True)

    successes = np.zeros((args.num_episodes, len(TASKS)), dtype=np.int32)
    progressions = []

    for ep, ep_seed in enumerate(ep_seeds):
        env = Env(seed=ep_seed)
        venv = VecPyTorch(DummyVecEnv([lambda e=env: e]), device=device)
        successes[ep] = run_episode(model, venv, config["model_kwargs"]["hidsize"], device)
        unlocked = int(successes[ep].sum())
        prog = balrog_progression(successes[ep])
        progressions.append(prog)
        print(
            f"ep {ep+1:03d}/{args.num_episodes}: seed={ep_seed} unlocked={unlocked}/22 "
            f"progression={100*prog:.2f}%",
            flush=True,
        )

    rates = 100.0 * successes.mean(axis=0)
    score = crafter_score(rates)
    avg_prog = 100.0 * float(np.mean(progressions))
    se_prog = (
        100.0 * float(np.std(progressions, ddof=1) / np.sqrt(len(progressions)))
        if len(progressions) > 1
        else 0.0
    )
    result = {
        "ckpt": os.path.abspath(args.ckpt),
        "num_episodes": args.num_episodes,
        "seeds": ep_seeds,
        "seeds_file": args.seeds_file or None,
        "score": score,
        "balrog_progression_pct": avg_prog,
        "balrog_progression_se": se_prog,
        "per_episode_progression": [100.0 * p for p in progressions],
        "success_rate": {t: float(r) for t, r in zip(TASKS, rates)},
        "mean_unlocked": float(successes.sum(axis=1).mean()),
    }
    print(json.dumps(result, indent=2))

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        f"eval_e{os.path.basename(args.ckpt).replace('agent-', '').replace('.pt', '')}_{args.num_episodes}ep.json",
    )
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
