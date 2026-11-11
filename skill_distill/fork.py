#!/usr/bin/env python3
"""Replay event prefixes and evaluate short counterfactual action forks."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import crafter
import numpy as np

from skill_distill.crafter import ACTIONS, observation_hash, read_jsonl, write_jsonl


def _branch_score(before: Dict[str, Any], after: Dict[str, Any], reward: float, done: bool) -> Dict[str, Any]:
    new_achievements = [
        key
        for key, value in after["achievements"].items()
        if int(value) > int(before["achievements"].get(key, 0))
    ]
    return {
        "reward": float(reward),
        "new_achievements": sorted(new_achievements),
        "achievement_gain": len(new_achievements),
        "health_delta": int(after["inventory"]["health"]) - int(before["inventory"]["health"]),
        "survived": not bool(done and int(after["inventory"]["health"]) <= 0),
    }


def _reconstruct(event: Dict[str, Any]):
    env = crafter.Env(seed=int(event["seed"]), length=10000)
    obs = env.reset()
    for action in event["action_prefix"]:
        obs, _, done, _ = env.step(int(action))
        if done:
            raise RuntimeError(f"episode ended before anchor for {event['event_id']}")
    anchor = next(step for step in event["steps"] if step["step"] == event["anchor_step"])
    if observation_hash(obs) != anchor["obs_hash"]:
        raise RuntimeError(f"observation hash mismatch for {event['event_id']}")
    return env, anchor


def fork_event(event: Dict[str, Any], top_k: int = 3, bottom_k: int = 1, horizon: int = 9) -> Dict[str, Any]:
    anchor_record = next(
        step for step in event["steps"] if step["step"] == event["anchor_step"]
    )
    if anchor_record.get("counterfactuals"):
        return {
            "schema_version": 1,
            "event_id": event["event_id"],
            "goal": event["goal"],
            "episode_id": event["episode_id"],
            "seed": event["seed"],
            "anchor_step": event["anchor_step"],
            "source": "same_process_collector",
            "branches": anchor_record["counterfactuals"],
        }
    env, anchor = _reconstruct(event)
    logits = np.asarray(anchor["teacher"]["pi_logits"])
    ranked = list(np.argsort(logits)[::-1])
    candidates = ranked[:top_k] + ranked[-bottom_k:] + [int(anchor["action_id"])]
    candidates = list(dict.fromkeys(int(value) for value in candidates))
    continuation = [
        int(step["action_id"])
        for step in event["steps"]
        if int(step["step"]) > int(event["anchor_step"])
    ][: max(0, horizon - 1)]

    before = {
        "inventory": env._player.inventory.copy(),
        "achievements": env._player.achievements.copy(),
    }
    branches = []
    for candidate in candidates:
        branch_env = copy.deepcopy(env)
        actions = [candidate] + continuation
        total_reward = 0.0
        done = False
        info = before
        steps = 0
        for action in actions:
            _, reward, done, info = branch_env.step(int(action))
            total_reward += float(reward)
            steps += 1
            if done:
                break
        branches.append(
            {
                "first_action_id": candidate,
                "first_action_name": ACTIONS[candidate],
                "is_teacher_action": candidate == int(anchor["action_id"]),
                "teacher_logit": float(logits[candidate]),
                "steps": steps,
                "score": _branch_score(before, info, total_reward, done),
            }
        )
    return {
        "schema_version": 1,
        "event_id": event["event_id"],
        "goal": event["goal"],
        "episode_id": event["episode_id"],
        "seed": event["seed"],
        "anchor_step": event["anchor_step"],
        "branches": branches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--bottom-k", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-unreplayable",
        action="store_true",
        help="Skip legacy cross-process replays whose states diverge.",
    )
    args = parser.parse_args()
    events = read_jsonl(args.events)
    if args.limit:
        events = events[: args.limit]
    forks = []
    skipped = []
    for event in events:
        try:
            forks.append(
                fork_event(event, top_k=args.top_k, bottom_k=args.bottom_k, horizon=args.horizon)
            )
        except RuntimeError as error:
            if not args.skip_unreplayable:
                raise
            skipped.append({"event_id": event["event_id"], "error": str(error)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, forks)
    print(json.dumps({"forks": len(forks), "skipped": skipped}, indent=2))


if __name__ == "__main__":
    main()
