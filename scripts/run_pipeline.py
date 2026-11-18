"""Run candidate features, five models, tuning, HMM decoding, and evaluation.

The script deliberately keeps train/test groups intact. It never reads an
``evaluator`` partition; only ``train`` and ``student_test`` are accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import joblib
import pandas as pd
import optuna
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

from src.data.loader import list_trips, read_arcs
from src.evaluation import evaluate_candidates, route_continuity
from src.hmm.viterbi import build_transition_graph, decode_viterbi
from src.matching.features import FEATURE_COLUMNS, build_candidate_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--radius-m", type=float, default=75.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--limit-trips", type=int)
    parser.add_argument(
        "--final-model",
        choices=["logistic_regression", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"],
        default="extra_trees",
        help="Model exported for API inference; all five models are still compared.",
    )
    return parser.parse_args()


def models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=300, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, n_jobs=-1, random_state=seed, class_weight="balanced"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=150, random_state=seed),
    }


def attach_predictions(table: pd.DataFrame, model: object) -> pd.DataFrame:
    result = table.copy()
    result["prob"] = model.predict_proba(result[FEATURE_COLUMNS])[:, 1]
    result["predicted_label"] = (result["prob"] >= 0.5).astype(int)
    return result


def classifier_route(table: pd.DataFrame) -> np.ndarray:
    keys = ["trip_id", "point_idx"] if "trip_id" in table.columns else ["point_idx"]
    return table.sort_values("prob").groupby(keys).tail(1).sort_values(keys)["arc_id"].to_numpy(dtype=int)


def evaluate_by_trip(table: pd.DataFrame, root: str) -> dict[str, float]:
    """Evaluate each trip with its own network, then average the metrics."""
    metrics = []
    for trip_id, group in table.groupby("trip_id", sort=True):
        arcs = read_arcs("train", str(trip_id), root)
        metrics.append(evaluate_candidates(group, classifier_route(group), arcs))
    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def run_bayesian_tuning(train: pd.DataFrame, trips: list[str], root: str, output: Path) -> dict[str, float]:
    """Tune logistic-regression regularisation with Optuna's TPE sampler."""
    cv = GroupKFold(n_splits=min(5, len(trips)))
    X, y, groups = train[FEATURE_COLUMNS], train["label"], train["trip_id"]

    def objective(trial: optuna.Trial) -> float:
        model = LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 1e3, log=True),
            class_weight="balanced",
            max_iter=300,
            random_state=42,
        )
        scores = []
        for fit_idx, valid_idx in cv.split(X, y, groups):
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            valid = attach_predictions(train.iloc[valid_idx], model)
            scores.append(evaluate_by_trip(valid, root)["top1_accuracy"])
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)
    trials = study.trials_dataframe()
    trials.to_csv(output / "tables" / "bayesian_trials.csv", index=False)
    return {"best_value": float(study.best_value), **{key: float(value) for key, value in study.best_params.items()}}


def main() -> None:
    args = parse_args()
    if args.data_root.endswith("evaluator"):
        raise ValueError("The evaluator partition is forbidden")
    seed = 42
    train_trips = list_trips("train", args.data_root)
    test_trips = list_trips("student_test", args.data_root)
    if args.limit_trips:
        train_trips, test_trips = train_trips[: args.limit_trips], test_trips[: args.limit_trips]
    tables = [build_candidate_table("train", trip, args.data_root, args.radius_m, args.max_candidates) for trip in train_trips]
    train = pd.concat(tables, ignore_index=True)
    X, y, groups = train[FEATURE_COLUMNS], train["label"], train["trip_id"]
    output = args.output_dir
    for folder in [output / "tables", output / "logs", output / "models"]:
        folder.mkdir(parents=True, exist_ok=True)
    cv = GroupKFold(n_splits=min(5, len(train_trips)))
    rows: list[dict[str, object]] = []
    for name, model in models(seed).items():
        fold_predictions = []
        started = time.perf_counter()
        for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y, groups)):
            model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            valid = attach_predictions(train.iloc[valid_idx], model)
            fold_predictions.append({"model": name, "fold": fold, **evaluate_by_trip(valid, args.data_root)})
        model.fit(X, y)
        rows.extend(fold_predictions)
        rows.append({"model": name, "fold": "full", "fit_seconds": time.perf_counter() - started})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "tables" / "model_comparison.csv", index=False)
    fold_rows = comparison[comparison["fold"] != "full"]
    metric_columns = [
        "top1_accuracy",
        "candidate_miss_rate",
        "route_continuity",
        "macro_f1_candidate",
        "log_loss_candidate",
    ]
    summary = fold_rows.groupby("model", as_index=False)[metric_columns].mean()
    for column in metric_columns:
        summary[f"{column}_percent"] = summary[column] * 100.0
    summary.to_csv(output / "tables" / "model_summary_percentages.csv", index=False)

    tuning = run_bayesian_tuning(train, train_trips, args.data_root, output)
    (output / "logs" / "tuning_best.json").write_text(json.dumps(tuning, indent=2), encoding="utf-8")

    final_model = models(seed)[args.final_model]
    if args.final_model == "extra_trees":
        final_model = ExtraTreesClassifier(
            n_estimators=250, n_jobs=-1, random_state=seed, class_weight="balanced"
        )
    final_model.fit(X, y)
    joblib.dump(
        {"model": final_model, "feature_columns": FEATURE_COLUMNS, "config": vars(args)},
        output / "models" / "final_extra_trees.joblib",
    )
    test_rows = []
    for trip in test_trips:
        table = build_candidate_table("student_test", trip, args.data_root, args.radius_m, args.max_candidates)
        predicted = attach_predictions(table, final_model)
        arcs = read_arcs("student_test", trip, args.data_root)
        classifier_route_pred = classifier_route(predicted)
        graph = build_transition_graph(arcs)
        hmm_pred = decode_viterbi(predicted, graph)
        truth = np.array(
            [int(group.loc[group["label"].eq(1), "arc_id"].iloc[0]) for _, group in table.groupby("point_idx", sort=True)]
        )
        for decoder, route in [("classifier_only", classifier_route_pred), ("hmm", hmm_pred)]:
            test_rows.append(
                {
                    "trip_id": trip,
                    "decoder": decoder,
                    "top1_accuracy": float(np.mean(route == truth)),
                    "sequence_exact_match": float(np.all(route == truth)),
                    "route_continuity": route_continuity(route, arcs),
                    "n_points": len(route),
                }
            )
    pd.DataFrame(test_rows).to_csv(output / "tables" / "decoder_comparison.csv", index=False)
    decoder = pd.DataFrame(test_rows)
    extra_trees_percentages = {
        "model": args.final_model,
        "student_test_classifier_top1_percent": float(
            decoder.loc[decoder["decoder"].eq("classifier_only"), "top1_accuracy"].mean() * 100
        ),
        "student_test_hmm_top1_percent": float(
            decoder.loc[decoder["decoder"].eq("hmm"), "top1_accuracy"].mean() * 100
        ),
        "student_test_classifier_continuity_percent": float(
            decoder.loc[decoder["decoder"].eq("classifier_only"), "route_continuity"].mean() * 100
        ),
        "student_test_hmm_continuity_percent": float(
            decoder.loc[decoder["decoder"].eq("hmm"), "route_continuity"].mean() * 100
        ),
    }
    (output / "logs" / "extra_trees_percentages.json").write_text(
        json.dumps(extra_trees_percentages, indent=2), encoding="utf-8"
    )
    print(json.dumps(extra_trees_percentages, indent=2))
    (output / "logs" / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
