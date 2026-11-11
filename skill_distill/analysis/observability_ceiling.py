#!/usr/bin/env python3
"""Measure how well text-visible features can predict teacher actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


VITALS = ("health", "food", "drink", "energy")
VISIBLE_OBJECTS = (
    "tree",
    "stone",
    "water",
    "coal",
    "iron",
    "diamond",
    "cow",
    "zombie",
    "skeleton",
    "table",
    "furnace",
    "plant",
    "lava",
)


def load_decisions(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported decisions file: {path}")


def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for obj in VISIBLE_OBJECTS:
        vis = f"visible_{obj}"
        dist = f"dist_{obj}"
        direction = f"dir_{obj}"
        if vis not in out.columns:
            out[vis] = False
        if dist not in out.columns:
            out[dist] = np.nan
        if direction not in out.columns:
            out[direction] = None
        out[vis] = out[vis].fillna(False).astype(bool)
        out[f"dist_bin_{obj}"] = pd.cut(
            out[dist],
            bins=[-0.5, 0.5, 1.5, 2.5, 4.5, 100],
            labels=["0", "1", "2", "3-4", "5+"],
        ).astype(str)
    for vital in VITALS:
        if vital not in out.columns:
            out[vital] = -1
        out[f"{vital}_bin"] = pd.cut(
            out[vital],
            bins=[-1.5, -0.5, 2.5, 5.5, 9.5],
            labels=["missing", "low", "mid", "high"],
        ).astype(str)
    if "steps_since_achievement" not in out.columns:
        out["steps_since_achievement"] = np.nan
    out["steps_since_achievement_bin"] = pd.cut(
        out["steps_since_achievement"].fillna(-1),
        bins=[-1.5, -0.5, 0.5, 5.5, 20.5, 1e9],
        labels=["none", "0", "1-5", "6-20", "21+"],
    ).astype(str)
    if "last_unlocked_achievement" not in out.columns:
        out["last_unlocked_achievement"] = None
    out["last_unlocked_achievement"] = out["last_unlocked_achievement"].fillna("none").astype(str)
    for col in ("front", "is_night", "sleeping"):
        if col not in out.columns:
            out[col] = None
    out["front"] = out["front"].fillna("unknown").astype(str)
    out["is_night"] = out["is_night"].fillna(False).astype(bool)
    out["sleeping"] = out["sleeping"].fillna(False).astype(bool)
    ach_cols = [c for c in out.columns if c.startswith("ach_")]
    for col in ach_cols:
        out[col] = out[col].fillna(False).astype(bool)
    inv_flag_cols = [c for c in out.columns if c.startswith("has_") and c != "has_inventory_delta" and c != "has_unlock"]
    for col in inv_flag_cols:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def feature_sets(frame: pd.DataFrame) -> Dict[str, List[str]]:
    text_cols = [
        "front",
        "is_night",
        "sleeping",
        *[f"visible_{obj}" for obj in VISIBLE_OBJECTS],
        *[f"dist_bin_{obj}" for obj in VISIBLE_OBJECTS],
        *[f"{vital}_bin" for vital in VITALS],
        *[c for c in frame.columns if c.startswith("has_") and c not in ("has_inventory_delta", "has_unlock")],
    ]
    ach_cols = [c for c in frame.columns if c.startswith("ach_")]
    history_cols = ["last_unlocked_achievement", "steps_since_achievement_bin"]
    return {
        "text_only": text_cols,
        "text_plus_achievements": text_cols + ach_cols,
        "text_plus_ach_plus_history": text_cols + ach_cols + history_cols,
    }


def majority_accuracy(y: pd.Series) -> float:
    if len(y) == 0:
        return float("nan")
    return float((y == y.mode().iloc[0]).mean())


def aliasing_ceiling(frame: pd.DataFrame, feature_cols: Sequence[str], target: str) -> Dict[str, Any]:
    work = frame[list(feature_cols) + [target]].copy()
    # Hash feature tuples for grouping.
    keys = pd.util.hash_pandas_object(work[list(feature_cols)], index=False).astype(str)
    work = work.assign(_key=keys)
    grouped = work.groupby("_key", sort=False)[target]
    # Within-bucket modal accuracy.
    correct = 0
    total = 0
    multi_action_steps = 0
    n_buckets = 0
    n_conflict_buckets = 0
    for _, actions in grouped:
        n_buckets += 1
        total += len(actions)
        mode = actions.mode().iloc[0]
        correct += int((actions == mode).sum())
        if actions.nunique() > 1:
            n_conflict_buckets += 1
            multi_action_steps += len(actions)
    return {
        "steps": int(total),
        "buckets": int(n_buckets),
        "conflict_buckets": int(n_conflict_buckets),
        "conflict_bucket_rate": float(n_conflict_buckets / n_buckets) if n_buckets else None,
        "steps_in_conflict_buckets": int(multi_action_steps),
        "steps_in_conflict_rate": float(multi_action_steps / total) if total else None,
        "aliasing_ceiling_accuracy": float(correct / total) if total else None,
        "majority_baseline_accuracy": majority_accuracy(frame[target]),
    }


def prepare_xy(frame: pd.DataFrame, feature_cols: Sequence[str], target: str):
    data = frame[list(feature_cols) + [target]].copy()
    bool_cols = [c for c in feature_cols if data[c].dtype == bool]
    cat_cols = [c for c in feature_cols if c not in bool_cols]
    for col in bool_cols:
        data[col] = data[col].astype(int)
    for col in cat_cols:
        data[col] = data[col].astype(str).fillna("missing")
    x = data[list(feature_cols)]
    y = data[target].astype(str)
    return x, y, bool_cols, cat_cols


def train_tree(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    max_depth: int = 8,
) -> Dict[str, Any]:
    x_train, y_train, bool_cols, cat_cols = prepare_xy(train, feature_cols, target)
    x_test, y_test, _, _ = prepare_xy(test, feature_cols, target)
    # Keep only labels seen in train for log_loss stability.
    labels = sorted(y_train.unique().tolist())
    mask = y_test.isin(labels)
    x_test = x_test.loc[mask]
    y_test = y_test.loc[mask]
    if len(y_test) == 0:
        return {"error": "no overlapping labels in test"}

    preprocessor = ColumnTransformer(
        transformers=[
            ("bool", "passthrough", bool_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=20,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)
    return {
        "test_steps": int(len(y_test)),
        "n_classes_train": int(len(labels)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "log_loss": float(log_loss(y_test, proba, labels=labels)),
        "majority_test_accuracy": majority_accuracy(y_test),
        "tree_depth": int(clf.named_steps["tree"].get_depth()),
        "tree_leaves": int(clf.named_steps["tree"].get_n_leaves()),
    }


def evaluate_target(
    frame: pd.DataFrame,
    train_split: str,
    test_split: str,
    target: str,
    max_depth: int,
) -> Dict[str, Any]:
    sets = feature_sets(frame)
    train = frame[frame["split"] == train_split]
    test = frame[frame["split"] == test_split]
    report: Dict[str, Any] = {
        "target": target,
        "train_split": train_split,
        "test_split": test_split,
        "train_steps": int(len(train)),
        "test_steps": int(len(test)),
        "feature_sets": {},
    }
    for name, cols in sets.items():
        present = [c for c in cols if c in frame.columns]
        alias_train = aliasing_ceiling(train, present, target)
        alias_test = aliasing_ceiling(test, present, target)
        tree = train_tree(train, test, present, target, max_depth=max_depth)
        report["feature_sets"][name] = {
            "n_features": len(present),
            "aliasing_train": alias_train,
            "aliasing_test": alias_test,
            "decision_tree": tree,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decisions",
        default="data/skill_distill/v3/decision_db/decisions.csv",
    )
    parser.add_argument("--train-split", default="exploration")
    parser.add_argument("--test-split", default="revision")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument(
        "--out",
        default="data/skill_distill/v3/decision_db/observability_ceiling.json",
    )
    args = parser.parse_args()

    frame = ensure_columns(load_decisions(Path(args.decisions)))
    # Interaction subset for target-level predictability.
    interaction = frame[frame["next_interaction_type"].isin(["Do", "Place", "Make", "Sleep"])].copy()

    report = {
        "n_rows": int(len(frame)),
        "splits": frame["split"].value_counts().to_dict(),
        "argmax_match_rate": {
            split: float(sub["selected_is_argmax"].mean())
            for split, sub in frame.groupby("split")
            if "selected_is_argmax" in frame.columns
        },
        "predict_action": evaluate_target(
            frame, args.train_split, args.test_split, "action", args.max_depth
        ),
        "predict_next_interaction_target": evaluate_target(
            interaction,
            args.train_split,
            args.test_split,
            "next_interaction_target",
            args.max_depth,
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
