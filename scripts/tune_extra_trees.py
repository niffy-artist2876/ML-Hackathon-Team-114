"""Bayesian optimisation and final fit for the PathSnap ExtraTrees model.

The search uses only train trips. Validation is grouped by trip, and the
student_test partition is never used for tuning. To keep the search practical
on the large candidate table, each GPS point keeps every positive candidate
and a deterministic sample of up to four negative candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import list_trips, read_arcs
from src.evaluation import evaluate_candidates
from src.matching.features import FEATURE_COLUMNS, build_candidate_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--radius-m", type=float, default=75.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-per-point", type=int, default=4)
    return parser.parse_args()


def sample_training_rows(table: pd.DataFrame, negatives_per_point: int, seed: int) -> pd.DataFrame:
    """Keep all positives and deterministic negatives for each GPS point."""
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for _, group in table.groupby(["trip_id", "point_idx"], sort=False):
        positions = group.index.to_numpy()
        positives = positions[group["label"].to_numpy(dtype=int) == 1]
        negatives = positions[group["label"].to_numpy(dtype=int) == 0]
        if len(negatives) > negatives_per_point:
            negatives = rng.choice(negatives, size=negatives_per_point, replace=False)
        selected.append(np.concatenate([positives, np.asarray(negatives, dtype=int)]))
    return table.loc[np.concatenate(selected)].sort_index().reset_index(drop=True)


def candidate_route(table: pd.DataFrame) -> np.ndarray:
    return (
        table.sort_values("prob")
        .groupby("point_idx")
        .tail(1)
        .sort_values("point_idx")["arc_id"]
        .to_numpy(dtype=int)
    )


def score_validation(table: pd.DataFrame, root: str) -> dict[str, float]:
    rows = []
    for trip_id, group in table.groupby("trip_id", sort=True):
        arcs = read_arcs("train", str(trip_id), root)
        rows.append(evaluate_candidates(group, candidate_route(group), arcs))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def model_from_trial(trial: optuna.Trial, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 150, step=50),
        max_depth=trial.suggest_categorical("max_depth", [None, 20, 40, 60]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
        max_features=trial.suggest_float("max_features", 0.5, 1.0, step=0.25),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.negative_per_point < 1:
        raise ValueError("trials and negative-per-point must be positive")

    output = args.output_dir
    for folder in (output / "tables", output / "logs", output / "models"):
        folder.mkdir(parents=True, exist_ok=True)

    trips = list_trips("train", args.data_root)
    tables = [
        build_candidate_table("train", trip, args.data_root, args.radius_m, args.max_candidates)
        for trip in trips
    ]
    full = pd.concat(tables, ignore_index=True)
    sampled = sample_training_rows(full, args.negative_per_point, args.seed)
    cv = GroupKFold(n_splits=min(5, len(trips)))
    X = sampled[FEATURE_COLUMNS]
    y = sampled["label"]
    groups = sampled["trip_id"]

    def objective(trial: optuna.Trial) -> float:
        model = model_from_trial(trial, args.seed)
        fold_scores = []
        for fit_idx, valid_idx in cv.split(X, y, groups):
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            valid = full[full["trip_id"].isin(sampled.iloc[valid_idx]["trip_id"].unique())].copy()
            valid["prob"] = model.predict_proba(valid[FEATURE_COLUMNS])[:, 1]
            valid["predicted_label"] = (valid["prob"] >= 0.5).astype(int)
            fold_scores.append(score_validation(valid, args.data_root)["top1_accuracy"])
        trial.set_user_attr("fold_top1", fold_scores)
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials)
    trials_path = output / "tables" / "extra_trees_bayesian_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    best_params = dict(study.best_params)
    best_model = ExtraTreesClassifier(
        **best_params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    started = time.perf_counter()
    best_model.fit(X, y)
    fit_seconds = time.perf_counter() - started
    artifact = output / "models" / "final_extra_trees.joblib"
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": FEATURE_COLUMNS,
            "config": vars(args),
            "tuning": {"best_value": float(study.best_value), "best_params": best_params},
        },
        artifact,
    )

    payload = {
        "model": "extra_trees",
        "objective": "mean_grouped_validation_top1_accuracy",
        "trials": args.trials,
        "training_trips": len(trips),
        "full_candidate_rows": int(len(full)),
        "sampled_training_rows": int(len(sampled)),
        "negative_per_point": args.negative_per_point,
        "best_value": float(study.best_value),
        "best_params": best_params,
        "final_fit_seconds": fit_seconds,
        "artifact": str(artifact),
    }
    (output / "logs" / "extra_trees_bayesian_best.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
