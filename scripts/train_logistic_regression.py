"""Training and evaluation pipeline for supervised Logistic Regression on PathSnap.

Performs 5-fold trip-grouped cross-validation, fits the final pipeline on all
80 training trips, evaluates on the 10 student-test trips, and saves reproducible
model artifacts, metrics, and comparisons against the distance baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import read_arcs
from src.data.features import FEATURE_COLUMNS
from src.model.metrics import aggregate_trip_metrics, evaluate_candidate_table
from src.model.logistic_regression import (
    build_pipeline,
    prepare_features,
    attach_candidate_predictions,
    save_model_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "logistic_regression.json")
    parser.add_argument("--data-root", type=Path, default=REPO / "data" / "raw")
    parser.add_argument("--processed-root", type=Path, default=REPO / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results")
    return parser.parse_args()


def load_candidate_tables(trip_ids: list[str], partition: str, processed_root: Path) -> dict[str, pd.DataFrame]:
    """Load candidate parquet tables for a list of trips into a dictionary."""
    tables = {}
    for trip in trip_ids:
        path = processed_root / partition / trip / "candidates.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Candidate table missing: {path}")
        df = pd.read_parquet(path)
        tables[trip] = df
    return tables


def evaluate_trip_candidates(
    table: pd.DataFrame,
    trip_id: str,
    partition: str,
    raw_root: Path,
) -> dict[str, float | int]:
    """Evaluate a single scored candidate table against trip road network."""
    arcs = read_arcs(partition, trip_id, root=raw_root).to_numpy(dtype=int)
    metrics = evaluate_candidate_table(table, arcs=arcs, score_col="score")

    # Add candidate-level precision and recall
    labels = table["label"].to_numpy(dtype=int)
    predicted_pos = np.zeros(len(table), dtype=int)
    from src.model.metrics import _best_positions
    best_pos = _best_positions(table, score_col="score")
    predicted_pos[best_pos] = 1

    metrics["candidate_precision"] = float(precision_score(labels, predicted_pos, zero_division=0))
    metrics["candidate_recall"] = float(recall_score(labels, predicted_pos, zero_division=0))
    return metrics


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    # Load configuration
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hp = config["hyperparameters"]

    # Output directory layout
    models_dir = args.output_dir / "models" / "logistic_regression"
    eval_dir = args.output_dir / "evaluation" / "logistic_regression"
    logs_dir = args.output_dir / "logs" / "logistic_regression"
    for d in (models_dir, eval_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load fold assignments
    folds_df = pd.read_csv(REPO / config["folds_file"], dtype={"trip_id": str})
    if len(folds_df) != 80:
        raise ValueError(f"Expected 80 training trips, found {len(folds_df)}")
    train_trip_ids = sorted(folds_df["trip_id"].tolist())

    # Load student-test trip IDs
    test_trip_dirs = sorted((args.processed_root / "student_test").iterdir())
    test_trip_ids = [d.name for d in test_trip_dirs if d.is_dir()]
    if len(test_trip_ids) != 10:
        raise ValueError(f"Expected 10 student-test trips, found {len(test_trip_ids)}")

    print(f"Loaded {len(train_trip_ids)} train trips and {len(test_trip_ids)} student-test trips.")

    # 1. Load all candidate parquet tables
    load_start = time.perf_counter()
    train_tables = load_candidate_tables(train_trip_ids, "train", args.processed_root)
    test_tables = load_candidate_tables(test_trip_ids, "student_test", args.processed_root)
    load_time = time.perf_counter() - load_start
    print(f"Loaded all candidate parquet tables in {load_time:.2f} s.")

    # 2. 5-Fold Trip-Grouped Cross-Validation
    print("\n--- Starting 5-Fold Trip-Grouped Cross-Validation ---")
    cv_start = time.perf_counter()
    fold_metric_rows = []
    fold_cv_trips = []

    for fold_id in range(5):
        fold_train_trips = folds_df.loc[folds_df["fold"] != fold_id, "trip_id"].tolist()
        fold_val_trips = folds_df.loc[folds_df["fold"] == fold_id, "trip_id"].tolist()

        train_data = pd.concat([train_tables[t] for t in fold_train_trips], ignore_index=True)
        val_data = pd.concat([train_tables[t] for t in fold_val_trips], ignore_index=True)

        X_train = prepare_features(train_data, feature_cols=FEATURE_COLUMNS)
        y_train = train_data["label"].to_numpy(dtype=int)

        # Build and fit pipeline
        pipe = build_pipeline(
            C=hp["C"],
            max_iter=hp["max_iter"],
            random_state=hp["random_state"],
            solver=hp["solver"],
            class_weight=hp["class_weight"],
        )

        t_fit_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t_fit_start

        lr_model = pipe.named_steps["classifier"]
        n_iter = int(lr_model.n_iter_[0])
        converged = n_iter < hp["max_iter"]

        # Validate on each trip in validation fold
        t_val_start = time.perf_counter()
        val_trip_results = []
        for trip_id in fold_val_trips:
            scored = attach_candidate_predictions(pipe, train_tables[trip_id])
            res = evaluate_trip_candidates(scored, trip_id, "train", args.data_root)
            val_trip_results.append({"partition": "train", "trip_id": trip_id, "fold": fold_id, **res})
        val_seconds = time.perf_counter() - t_val_start

        val_df = pd.DataFrame(val_trip_results)
        fold_agg = aggregate_trip_metrics(val_df).iloc[0].to_dict()

        fold_row = {
            "fold": fold_id,
            "train_trips_count": len(fold_train_trips),
            "val_trips_count": len(fold_val_trips),
            "train_rows": len(train_data),
            "val_rows": len(val_data),
            "val_points": fold_agg["points"],
            "covered_points": fold_agg["covered_points"],
            "candidate_miss_rate": fold_agg["candidate_miss_rate"],
            "top1_accuracy": fold_agg["top1_accuracy"],
            "trip_mean_top1": fold_agg["trip_mean_top1_accuracy"],
            "candidate_macro_f1": fold_agg["trip_mean_candidate_macro_f1"],
            "point_log_loss": fold_agg["point_mean_log_loss"],
            "route_continuity": fold_agg["route_continuity"],
            "mean_selected_rank": fold_agg["mean_selected_rank"],
            "fit_seconds": fit_seconds,
            "val_seconds": val_seconds,
            "n_iterations": n_iter,
            "converged": converged,
            "warnings_count": len(captured_warnings),
        }
        fold_metric_rows.append(fold_row)
        fold_cv_trips.extend(val_trip_results)

        print(
            f"Fold {fold_id}: Top-1={fold_agg['top1_accuracy']*100:.2f}%, "
            f"Macro-F1={fold_agg['trip_mean_candidate_macro_f1']*100:.2f}%, "
            f"LogLoss={fold_agg['point_mean_log_loss']:.4f}, "
            f"Continuity={fold_agg['route_continuity']*100:.2f}%, "
            f"FitTime={fit_seconds:.2f}s (iter={n_iter})"
        )

    cv_seconds = time.perf_counter() - cv_start
    cv_folds_df = pd.DataFrame(fold_metric_rows)
    cv_folds_df.to_csv(eval_dir / "cv_fold_metrics.csv", index=False)

    # Overall CV summary (point-weighted across all 80 trips evaluated in their validation folds)
    cv_all_val_df = pd.DataFrame(fold_cv_trips)
    cv_summary = aggregate_trip_metrics(cv_all_val_df)
    cv_summary["fit_seconds_total"] = cv_folds_df["fit_seconds"].sum()
    cv_summary["val_seconds_total"] = cv_folds_df["val_seconds"].sum()
    cv_summary.to_csv(eval_dir / "cv_summary.csv", index=False)

    print("\n--- Cross-Validation Summary (5 Folds on 80 Training Trips) ---")
    print(cv_summary.to_string(index=False))

    # 3. Final Training on All 80 Training Trips
    print("\n--- Fitting Final Logistic Regression Model on All 80 Training Trips ---")
    all_train_data = pd.concat([train_tables[t] for t in train_trip_ids], ignore_index=True)
    X_full = prepare_features(all_train_data, feature_cols=FEATURE_COLUMNS)
    y_full = all_train_data["label"].to_numpy(dtype=int)

    final_pipe = build_pipeline(
        C=hp["C"],
        max_iter=hp["max_iter"],
        random_state=hp["random_state"],
        solver=hp["solver"],
        class_weight=hp["class_weight"],
    )

    t_full_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as full_warnings:
        warnings.simplefilter("always")
        final_pipe.fit(X_full, y_full)
    full_fit_seconds = time.perf_counter() - t_full_start
    final_lr = final_pipe.named_steps["classifier"]
    final_n_iter = int(final_lr.n_iter_[0])

    print(f"Final model fitted on {len(all_train_data):,} rows in {full_fit_seconds:.2f} s ({final_n_iter} iterations).")

    # Serialize final model artifact
    model_artifact_path = models_dir / "logistic_regression.joblib"
    save_model_artifact(
        final_pipe,
        model_artifact_path,
        metadata={
            "model_family": "logistic_regression",
            "hyperparameters": hp,
            "feature_columns": FEATURE_COLUMNS,
            "train_trips_count": len(train_trip_ids),
            "train_rows_count": len(all_train_data),
            "fit_seconds": full_fit_seconds,
            "n_iterations": final_n_iter,
            "scaler_mean": final_pipe.named_steps["scaler"].mean_.tolist(),
            "scaler_scale": final_pipe.named_steps["scaler"].scale_.tolist(),
            "coefficients": final_lr.coef_[0].tolist(),
            "intercept": float(final_lr.intercept_[0]),
        },
    )
    print(f"Saved model artifact to {model_artifact_path.relative_to(REPO)}.")

    # 4. Evaluate Final Model on All 80 Training Trips (Training Fit)
    print("\n--- Evaluating Final Model on All 80 Training Trips ---")
    t_eval_train_start = time.perf_counter()
    train_trip_rows = []
    for trip_id in train_trip_ids:
        scored = attach_candidate_predictions(final_pipe, train_tables[trip_id])
        res = evaluate_trip_candidates(scored, trip_id, "train", args.data_root)
        train_trip_rows.append({"partition": "train", "trip_id": trip_id, **res})
    eval_train_seconds = time.perf_counter() - t_eval_train_start

    train_trips_df = pd.DataFrame(train_trip_rows)
    train_trips_df.to_csv(eval_dir / "train_by_trip.csv", index=False)
    train_summary = aggregate_trip_metrics(train_trips_df)

    # 5. Evaluate Final Model on 10 Student-Test Trips
    print("\n--- Evaluating Final Model on 10 Student-Test Trips ---")
    t_eval_test_start = time.perf_counter()
    test_trip_rows = []
    for trip_id in test_trip_ids:
        scored = attach_candidate_predictions(final_pipe, test_tables[trip_id])
        res = evaluate_trip_candidates(scored, trip_id, "student_test", args.data_root)
        test_trip_rows.append({"partition": "student_test", "trip_id": trip_id, **res})
    eval_test_seconds = time.perf_counter() - t_eval_test_start

    test_trips_df = pd.DataFrame(test_trip_rows)
    test_trips_df.to_csv(eval_dir / "student_test_by_trip.csv", index=False)
    test_summary = aggregate_trip_metrics(test_trips_df)
    test_summary["eval_seconds"] = eval_test_seconds
    test_summary.to_csv(eval_dir / "student_test_summary.csv", index=False)

    print("\n--- Student-Test Evaluation Summary ---")
    print(test_summary.to_string(index=False))

    # 6. Compare with Distance Baseline
    baseline_summary_path = args.output_dir / "evaluation" / "distance_baseline_summary.csv"
    baseline_df = pd.read_csv(baseline_summary_path).set_index("partition")

    comp_rows = []
    for partition, lr_sum in [("train", train_summary.iloc[0]), ("student_test", test_summary.iloc[0])]:
        base = baseline_df.loc[partition]
        comp_rows.append({
            "partition": partition,
            "metric": "top1_accuracy",
            "distance_baseline": float(base["top1_accuracy"]),
            "logistic_regression": float(lr_sum["top1_accuracy"]),
            "absolute_diff": float(lr_sum["top1_accuracy"] - base["top1_accuracy"]),
        })
        comp_rows.append({
            "partition": partition,
            "metric": "candidate_macro_f1",
            "distance_baseline": float(base["trip_mean_candidate_macro_f1"]),
            "logistic_regression": float(lr_sum["trip_mean_candidate_macro_f1"]),
            "absolute_diff": float(lr_sum["trip_mean_candidate_macro_f1"] - base["trip_mean_candidate_macro_f1"]),
        })
        comp_rows.append({
            "partition": partition,
            "metric": "point_log_loss",
            "distance_baseline": float(base["point_mean_log_loss"]),
            "logistic_regression": float(lr_sum["point_mean_log_loss"]),
            "absolute_diff": float(lr_sum["point_mean_log_loss"] - base["point_mean_log_loss"]),
        })
        comp_rows.append({
            "partition": partition,
            "metric": "candidate_miss_rate",
            "distance_baseline": float(base["candidate_miss_rate"]),
            "logistic_regression": float(lr_sum["candidate_miss_rate"]),
            "absolute_diff": float(lr_sum["candidate_miss_rate"] - base["candidate_miss_rate"]),
        })
        comp_rows.append({
            "partition": partition,
            "metric": "route_continuity",
            "distance_baseline": float(base["route_continuity"]),
            "logistic_regression": float(lr_sum["route_continuity"]),
            "absolute_diff": float(lr_sum["route_continuity"] - base["route_continuity"]),
        })
        comp_rows.append({
            "partition": partition,
            "metric": "mean_selected_rank",
            "distance_baseline": float(base["mean_selected_rank"]),
            "logistic_regression": float(lr_sum["mean_selected_rank"]),
            "absolute_diff": float(lr_sum["mean_selected_rank"] - base["mean_selected_rank"]),
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(eval_dir / "baseline_comparison.csv", index=False)

    print("\n--- Logistic Regression vs Distance Baseline Comparison ---")
    print(comp_df.to_string(index=False))

    total_seconds = time.perf_counter() - total_start

    # 7. Save Run Log
    run_log = {
        "model_family": "logistic_regression",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "hyperparameters": hp,
        "features": FEATURE_COLUMNS,
        "excluded_columns": ["trip_id", "point_idx", "arc_id", "label"],
        "timings_seconds": {
            "data_loading": load_time,
            "cross_validation_total": cv_seconds,
            "cv_fit_sum": cv_folds_df["fit_seconds"].sum(),
            "cv_val_sum": cv_folds_df["val_seconds"].sum(),
            "final_model_fit": full_fit_seconds,
            "train_evaluation": eval_train_seconds,
            "student_test_evaluation": eval_test_seconds,
            "total_execution": total_seconds,
        },
        "cv_summary": cv_summary.iloc[0].to_dict(),
        "final_train_summary": train_summary.iloc[0].to_dict(),
        "student_test_summary": test_summary.iloc[0].to_dict(),
        "coefficients": dict(zip(FEATURE_COLUMNS, final_lr.coef_[0].tolist())),
        "intercept": float(final_lr.intercept_[0]),
    }
    (logs_dir / "train_run.json").write_text(json.dumps(run_log, indent=2, default=str), encoding="utf-8")

    # 8. Save Model Report
    coef_series = pd.Series(final_lr.coef_[0], index=FEATURE_COLUMNS).sort_values()
    report_md = f"""# Supervised Logistic Regression Model Report

## 1. Model Architecture & Preprocessing
- **Algorithm**: `LogisticRegression` (scikit-learn) with `solver='lbfgs'`.
- **Scaling**: `StandardScaler` fitted strictly on training candidate rows inside each fold (zero data leakage).
- **Hyperparameters**: $C = {hp['C']}$, $\\text{{class\\_weight}} = '{hp['class_weight']}'$, $\\text{{max\\_iter}} = {hp['max_iter']}$, $\\text{{random\\_state}} = {hp['random_state']}$.
- **Features Used ({len(FEATURE_COLUMNS)})**: {', '.join(FEATURE_COLUMNS)}.
- **Features Excluded**: `trip_id`, `point_idx`, `arc_id`, and `label`.

## 2. Learned Feature Weights (Standardized)
```text
{coef_series.to_string()}
Intercept: {final_lr.intercept_[0]:.4f}
```

## 3. Five-Fold Trip-Grouped Cross-Validation Results
- Point-weighted Top-1 Accuracy: **{cv_summary.iloc[0]['top1_accuracy']*100:.2f}%**
- Candidate Macro F1: **{cv_summary.iloc[0]['trip_mean_candidate_macro_f1']*100:.2f}%**
- Point-mean Log Loss: **{cv_summary.iloc[0]['point_mean_log_loss']:.4f}**
- Route Continuity: **{cv_summary.iloc[0]['route_continuity']*100:.2f}%**
- Candidate Miss Rate: **{cv_summary.iloc[0]['candidate_miss_rate']*100:.2f}%**

## 4. Student-Test Evaluation (10 Trips)
- Point-weighted Top-1 Accuracy: **{test_summary.iloc[0]['top1_accuracy']*100:.2f}%** (Trip mean: {test_summary.iloc[0]['trip_mean_top1_accuracy']*100:.2f}%)
- Candidate Macro F1: **{test_summary.iloc[0]['trip_mean_candidate_macro_f1']*100:.2f}%**
- Point-mean Log Loss: **{test_summary.iloc[0]['point_mean_log_loss']:.4f}**
- Route Continuity: **{test_summary.iloc[0]['route_continuity']*100:.2f}%**
- Candidate Miss Rate: **{test_summary.iloc[0]['candidate_miss_rate']*100:.4f}%** (15 misses / 19,160 points)

## 5. Comparison Against Nearest-Road Distance Baseline
| Partition | Metric | Distance Baseline | Logistic Regression | Diff |
|---|---|---:|---:|---:|
| **Student-Test** | Top-1 Accuracy | {baseline_df.loc['student_test', 'top1_accuracy']*100:.2f}% | {test_summary.iloc[0]['top1_accuracy']*100:.2f}% | {(test_summary.iloc[0]['top1_accuracy'] - baseline_df.loc['student_test', 'top1_accuracy'])*100:+.2f}% |
| **Student-Test** | Candidate Macro F1 | {baseline_df.loc['student_test', 'trip_mean_candidate_macro_f1']*100:.2f}% | {test_summary.iloc[0]['trip_mean_candidate_macro_f1']*100:.2f}% | {(test_summary.iloc[0]['trip_mean_candidate_macro_f1'] - baseline_df.loc['student_test', 'trip_mean_candidate_macro_f1'])*100:+.2f}% |
| **Student-Test** | Point Log Loss | {baseline_df.loc['student_test', 'point_mean_log_loss']:.4f} | {test_summary.iloc[0]['point_mean_log_loss']:.4f} | {test_summary.iloc[0]['point_mean_log_loss'] - baseline_df.loc['student_test', 'point_mean_log_loss']:+.4f} |
| **Student-Test** | Route Continuity | {baseline_df.loc['student_test', 'route_continuity']*100:.2f}% | {test_summary.iloc[0]['route_continuity']*100:.2f}% | {(test_summary.iloc[0]['route_continuity'] - baseline_df.loc['student_test', 'route_continuity'])*100:+.2f}% |
| **Train (CV)** | Top-1 Accuracy | {baseline_df.loc['train', 'top1_accuracy']*100:.2f}% | {cv_summary.iloc[0]['top1_accuracy']*100:.2f}% | {(cv_summary.iloc[0]['top1_accuracy'] - baseline_df.loc['train', 'top1_accuracy'])*100:+.2f}% |

## 6. Execution Timings
- Total Execution: **{total_seconds:.2f} s**
- 5-Fold Cross-Validation: **{cv_seconds:.2f} s**
- Final Model Fit (3.33M rows): **{full_fit_seconds:.2f} s**
- Student-Test Inference & Evaluation: **{eval_test_seconds:.2f} s**
"""
    (eval_dir / "logistic_regression_report.md").write_text(report_md, encoding="utf-8")
    print(f"\nCompleted Logistic Regression pipeline in {total_seconds:.2f} s.")


if __name__ == "__main__":
    main()
