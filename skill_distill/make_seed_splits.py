#!/usr/bin/env python3
"""Create reproducible, disjoint Crafter seed files for teacher distillation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


SPLITS = ("exploration", "revision", "acceptance", "final_test")


def read_seeds(path: Path) -> Set[int]:
    if not path.exists():
        return set()
    return {int(line.strip()) for line in path.read_text().splitlines() if line.strip()}


def write_seeds(path: Path, seeds: List[int]) -> None:
    path.write_text("".join(f"{seed}\n" for seed in seeds))


def sample_unique(
    rng: np.random.RandomState,
    count: int,
    excluded: Set[int],
) -> List[int]:
    selected: List[int] = []
    used = set(excluded)
    while len(selected) < count:
        seed = int(rng.randint(1, 2**32 - 1, dtype=np.uint32))
        if seed in used:
            continue
        used.add(seed)
        selected.append(seed)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exploration", type=int, default=120)
    parser.add_argument("--revision", type=int, default=40)
    parser.add_argument("--acceptance", type=int, default=40)
    parser.add_argument("--final-test", type=int, default=40)
    parser.add_argument("--rng-seed", type=int, default=20261027)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded: Set[int] = set()
    for path in args.exclude:
        excluded |= read_seeds(Path(path).expanduser())

    counts = {
        "exploration": args.exploration,
        "revision": args.revision,
        "acceptance": args.acceptance,
        "final_test": args.final_test,
    }
    total = sum(counts.values())
    rng = np.random.RandomState(args.rng_seed)
    pool = sample_unique(rng, total, excluded)

    splits: Dict[str, List[int]] = {}
    offset = 0
    for name in SPLITS:
        size = counts[name]
        splits[name] = pool[offset : offset + size]
        offset += size
        write_seeds(out_dir / f"{name}_seeds.txt", splits[name])

    # Backward-compatible aliases used by older scripts.
    write_seeds(out_dir / "train_seeds.txt", splits["exploration"])
    write_seeds(out_dir / "validation_seeds.txt", splits["revision"])

    all_selected = [seed for name in SPLITS for seed in splits[name]]
    assert len(all_selected) == len(set(all_selected))
    assert not set(all_selected) & excluded

    manifest = {
        "schema_version": 3,
        "rng_seed": args.rng_seed,
        "excluded_count": len(excluded),
        "exclude_files": args.exclude,
        "counts": {name: len(splits[name]) for name in SPLITS},
        "roles": {
            "exploration": "LLM may query repeatedly and propose hypotheses",
            "revision": "Counterexample search and hypothesis revision",
            "acceptance": "Fresh batches for one-shot acceptance gates",
            "final_test": "Frozen rule-set evaluation; used once",
        },
        "aliases": {
            "train_seeds.txt": "exploration_seeds.txt",
            "validation_seeds.txt": "revision_seeds.txt",
        },
    }
    (out_dir / "seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
