"""Shared Crafter state, text-projection, and replay helpers."""

import hashlib
import itertools
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import crafter
import numpy as np

VITALS = ("health", "food", "drink", "energy")
ACTIONS = [
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
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
]


def _semantic_names() -> Tuple[List[Any], int]:
    names: List[Any] = [0] * 19
    env = crafter.Env()
    for name, index in itertools.chain(
        env._world._mat_ids.items(), env._sem_view._obj_ids.items()
    ):
        text = str(name)
        if "objects." in text:
            text = text[text.find("objects.") + len("objects.") : -2].lower()
        names[index] = text
    return names, names.index("player")


ID_TO_ITEM, PLAYER_INDEX = _semantic_names()


def model_observation(obs: np.ndarray, device: Any) -> Any:
    """Convert one Crafter HWC uint8 frame to the teacher's BCHW float input."""
    import torch as th

    batch = np.asarray(obs)[None]
    batch = np.transpose(batch, (0, 3, 1, 2))
    return th.from_numpy(batch.copy()).float().to(device) / 255.0


def observation_hash(obs: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()


def _int_dict(values: Dict[str, Any]) -> Dict[str, int]:
    return {str(key): int(value) for key, value in values.items()}


def raw_state_info(env: Any) -> Dict[str, Any]:
    """Build the raw/augmented info dictionary without advancing the environment."""
    player = env._player
    return {
        "inventory": player.inventory.copy(),
        "achievements": player.achievements.copy(),
        "semantic": env._sem_view(),
        "player_pos": np.asarray(player.pos).copy(),
        "player_facing": np.asarray(player.facing).copy(),
        "sleeping": bool(player.sleeping),
        "dead": bool(player.health <= 0),
        "daylight": float(env._world.daylight),
        "view": np.asarray(env._view).copy(),
        "unlocked": set(),
    }


def augment_step_info(env: Any, info: Dict[str, Any], before: Dict[str, Any]) -> Dict[str, Any]:
    """Add fields used by BALROG and derive per-step deltas."""
    result = dict(info)
    player = env._player
    result.update(
        {
            "player_facing": np.asarray(player.facing).copy(),
            "sleeping": bool(player.sleeping),
            "dead": bool(player.health <= 0),
            "daylight": float(env._world.daylight),
            "view": np.asarray(env._view).copy(),
        }
    )
    old_achievements = before["achievements"]
    old_inventory = before["inventory"]
    result["unlocked"] = {
        key
        for key, value in result["achievements"].items()
        if int(value) > int(old_achievements.get(key, 0))
    }
    result["inventory_delta"] = {
        key: int(value) - int(old_inventory.get(key, 0))
        for key, value in result["inventory"].items()
        if int(value) != int(old_inventory.get(key, 0))
    }
    return result


def text_observation(info: Dict[str, Any], include_daynight: bool = False) -> Dict[str, str]:
    long_context = _describe_frame(info, include_daynight)
    short_context = _describe_inventory(info)
    return {
        "long_term_context": long_context,
        "short_term_context": short_context,
    }


def _bearing(delta: np.ndarray) -> str:
    parts = []
    if delta[1] < 0:
        parts.append("north")
    elif delta[1] > 0:
        parts.append("south")
    if delta[0] < 0:
        parts.append("west")
    elif delta[0] > 0:
        parts.append("east")
    distance = int(abs(delta[0]) + abs(delta[1]))
    return f"{distance} step{'s' if distance > 1 else ''} to your {'-'.join(parts)}"


def _describe_inventory(info: Dict[str, Any]) -> str:
    inventory = info["inventory"]
    status = "Your status:\n{}".format(
        "\n".join(f"- {name}: {int(inventory[name])}/9" for name in VITALS)
    )
    items = "\n".join(
        f"- {name}: {int(count)}"
        for name, count in inventory.items()
        if name not in VITALS and int(count) != 0
    )
    return status + "\n\n" + (
        f"Your inventory:\n{items}" if items else "You have nothing in your inventory."
    )


def _describe_frame(info: Dict[str, Any], include_daynight: bool) -> str:
    prefix = ""
    if info["sleeping"]:
        prefix += "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif info["dead"]:
        prefix += "You died.\n\n"
    if include_daynight:
        prefix += "It is nighttime.\n\n" if info["daylight"] < 0.5 else "It is daytime.\n\n"

    semantic = info["semantic"]
    player_pos = np.asarray(info["player_pos"])
    view = np.asarray(info["view"])
    semantic = semantic[
        player_pos[0] - view[0] // 2 : player_pos[0] + view[0] // 2 + 1,
        player_pos[1] - view[1] // 2 + 1 : player_pos[1] + view[1] // 2,
    ]
    center = np.array([view[0] // 2, view[1] // 2 - 1])
    closest: Dict[str, Tuple[str, int]] = {}
    for i in range(semantic.shape[0]):
        for j in range(semantic.shape[1]):
            index = int(semantic[i, j])
            if index == PLAYER_INDEX:
                continue
            delta = np.array([i, j]) - center
            distance = int(abs(delta[0]) + abs(delta[1]))
            name = str(ID_TO_ITEM[index])
            if name not in closest or distance < closest[name][1]:
                closest[name] = (_bearing(delta), distance)
    if closest:
        visible = "You see:\n" + "\n".join(
            f"- {name} {location}" for name, (location, _) in closest.items()
        )
    else:
        visible = "You see nothing away from you."

    facing = np.asarray(info["player_facing"])
    target = center + facing
    if 0 <= target[0] < semantic.shape[0] and 0 <= target[1] < semantic.shape[1]:
        front = str(ID_TO_ITEM[int(semantic[int(target[0]), int(target[1])])])
    else:
        front = "nothing"
    return f"{prefix}\n\n{visible}\n\nYou face {front} at your front.".strip()


def compact_state(info: Dict[str, Any]) -> Dict[str, Any]:
    """Keep structured fields needed for grounding; omit the full semantic array."""
    inventory = _int_dict(info["inventory"])
    achievements = _int_dict(info["achievements"])
    return {
        "inventory": inventory,
        "achievements": achievements,
        "vitals": {key: inventory[key] for key in VITALS},
        "player_pos": [int(x) for x in np.asarray(info["player_pos"]).tolist()],
        "player_facing": [int(x) for x in np.asarray(info["player_facing"]).tolist()],
        "sleeping": bool(info["sleeping"]),
        "dead": bool(info["dead"]),
        "daylight": float(info["daylight"]),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(jsonable(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def replay_prefix(env: Any, actions: Sequence[int]) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    """Replay an action prefix and return the state immediately after it."""
    obs = env.reset()
    done = False
    info = raw_state_info(env)
    for action in actions:
        obs, _, done, step_info = env.step(int(action))
        info = augment_step_info(env, step_info, info)
        if done:
            break
    return obs, info, done


def action_id(name: str) -> Optional[int]:
    try:
        return ACTIONS.index(name)
    except ValueError:
        return None
