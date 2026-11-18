"""Shared Phase 4 train/validate/fit/test runner for candidate classifiers.

The steps match ``scripts/train_random_forest.py`` on the Random Forest branch:
fixed trip-level folds, training-only negative sampling, full-candidate
evaluation with the Phase 3 metrics, a final fit on all labelled training
trips, and a single scoring pass over student test.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.features import FEATURE_COLUMNS
from src.data.loader import read_arcs
from src.model.candidate_classifier import (
    add_predictions,
    feature_matrix,
    positive_probability,
    sample_training_candidates,
)
from src.model.metrics import aggregate_trip_metrics, evaluate_candidate_table

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    probability_column: str
    fit: Callable[..., Any]
    comparison_parameters: Sequence[str]


def resolve_path(path_value: str | Path, *, base: Path = REPO) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base / path


def load_candidate_tables(partition: str, model_name: str) -> dict[str, pd.DataFrame]:
    root = REPO / "data" / "processed" / partition
    if not root.is_dir():
        raise FileNotFoundError(f"Processed partition not found: {root}. Run the data stage first.")
    tables: dict[str, pd.DataFrame] = {}
    for trip_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        candidate_path = trip_dir / "candidates.parquet"
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Candidate table not found: {candidate_path}")
        table = pd.read_parquet(candidate_path)
        if "label" not in table.columns:
            raise ValueError(f"{model_name} evaluation requires labels in {candidate_path}")
        tables[trip_dir.name] = table
    if not tables:
        raise ValueError(f"No candidate tables found in {root}")
    return tables


def load_folds(path: Path, train_trip_ids: set[str]) -> dict[str, int]:
    folds = pd.read_csv(path, dtype={"trip_id": str})
    required = {"trip_id", "fold"}
    if not required.issubset(folds.columns):
        raise ValueError(f"Fold file must contain {sorted(required)}")
    folds["trip_id"] = folds["trip_id"].astype(str).str.zfill(8)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)
    if folds["trip_id"].duplicated().any():
        raise ValueError("Fold file contains duplicate trip IDs")
    mapping = dict(zip(folds["trip_id"], folds["fold"], strict=True))
    missing = sorted(train_trip_ids - set(mapping))
    extra = sorted(set(mapping) - train_trip_ids)
    if missing or extra:
        raise ValueError(f"Fold coverage mismatch; missing={missing}, extra={extra}")
    if folds["fold"].min() < 0:
        raise ValueError("Fold IDs must be non-negative")
    return {trip_id: mapping[trip_id] for trip_id in train_trip_ids}


def concat_tables(tables: dict[str, pd.DataFrame], trip_ids: list[str]) -> pd.DataFrame:
    if not trip_ids:
        raise ValueError("Cannot concatenate an empty trip list")
    return pd.concat([tables[trip_id] for trip_id in trip_ids], ignore_index=True)


def score_trip(
    table: pd.DataFrame,
    *,
    spec: ModelSpec,
    partition: str,
    trip_id: str,
    model: Any,
    feature_columns: list[str],
    fold: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = time.perf_counter()
    probabilities = positive_probability(model, table, feature_columns=feature_columns)
    inference_seconds = time.perf_counter() - start
    predictions = add_predictions(table, probabilities, probability_col=spec.probability_column)
    arcs = read_arcs(partition, trip_id, root=REPO / "data" / "raw").to_numpy(dtype=int)
    metrics = evaluate_candidate_table(predictions, arcs=arcs, score_col="score")
    metrics.update({
        "partition": partition,
        "trip_id": trip_id,
        "fold": fold if fold is not None else -1,
        "inference_seconds": inference_seconds,
    })
    return predictions, metrics


def prediction_columns(table: pd.DataFrame, probability_column: str) -> list[str]:
    columns = [
        "trip_id", "point_idx", "arc_id", "label", "candidate_rank",
        "distance_m", probability_column, "score", "selected",
    ]
    return [column for column in columns if column in table.columns]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def feature_importance(
    model: Any,
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    sample_rows: int,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Permutation importance on a fixed sample of final training rows.

    MLP and SVM pipelines have no impurity importances, so each feature is
    shuffled in turn and the increase in candidate-row log loss is recorded.
    """
    sample = table.sample(n=min(sample_rows, len(table)), random_state=random_state)
    result = permutation_importance(
        model,
        feature_matrix(sample, feature_columns),
        pd.to_numeric(sample["label"], errors="raise").to_numpy(dtype=int),
        scoring="neg_log_loss",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame({
        "feature": feature_columns,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance", ascending=False)


def run_training(
    spec: ModelSpec,
    *,
    config_path: Path,
    output_dir: Path | None = None,
    model_path: Path | None = None,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_columns = list(config.get("feature_columns", FEATURE_COLUMNS))
    output_dir = output_dir or resolve_path(config["outputs"]["directory"])
    model_path = model_path or resolve_path(config["outputs"]["model_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_tables = load_candidate_tables("train", spec.display_name)
    student_tables = load_candidate_tables("student_test", spec.display_name)
    folds = load_folds(resolve_path(config["folds_path"]), set(train_tables))
    fold_ids = sorted(set(folds.values()))
    model_parameters = dict(config["model_parameters"])
    max_negatives = config.get("max_negative_candidates_per_point", 4)
    random_state = int(model_parameters["random_state"])

    cv_trip_metrics: list[dict[str, Any]] = []
    cv_prediction_tables: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold in fold_ids:
        validation_ids = sorted(trip_id for trip_id, assigned_fold in folds.items() if assigned_fold == fold)
        training_ids = sorted(trip_id for trip_id, assigned_fold in folds.items() if assigned_fold != fold)
        full_training_table = concat_tables(train_tables, training_ids)
        training_table = sample_training_candidates(
            full_training_table,
            max_negative_candidates_per_point=max_negatives,
            random_state=random_state,
        )
        print(
            f"[fold {fold}] fitting on {len(training_table):,} rows "
            f"({len(full_training_table):,} before negative sampling); "
            f"validating {len(validation_ids)} trips",
            flush=True,
        )
        started = time.perf_counter()
        model = spec.fit(training_table, config=model_parameters, feature_columns=feature_columns)
        training_seconds = time.perf_counter() - started
        fold_inference_seconds = 0.0
        for trip_id in validation_ids:
            predictions, metrics = score_trip(
                train_tables[trip_id],
                spec=spec,
                partition="train",
                trip_id=trip_id,
                model=model,
                feature_columns=feature_columns,
                fold=fold,
            )
            fold_inference_seconds += float(metrics["inference_seconds"])
            cv_trip_metrics.append(metrics)
            predictions["fold"] = fold
            cv_prediction_tables.append(predictions.loc[:, prediction_columns(predictions, spec.probability_column)])
        print(
            f"[fold {fold}] completed in {training_seconds:.2f}s training "
            f"and {fold_inference_seconds:.2f}s inference",
            flush=True,
        )
        fold_metrics.append({
            "fold": fold,
            "training_trips": len(training_ids),
            "validation_trips": len(validation_ids),
            "training_rows_before_sampling": len(full_training_table),
            "training_rows": len(training_table),
            "training_seconds": training_seconds,
            "validation_inference_seconds": fold_inference_seconds,
        })

    cv_by_trip = pd.DataFrame(cv_trip_metrics).sort_values(["fold", "trip_id"])
    cv_summary_input = cv_by_trip.copy()
    cv_summary_input["partition"] = "cross_validation"
    cv_summary = aggregate_trip_metrics(cv_summary_input)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "cv_by_fold.csv", index=False)
    cv_by_trip.to_csv(output_dir / "cv_by_trip.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    pd.concat(cv_prediction_tables, ignore_index=True).to_parquet(
        output_dir / "cv_predictions.parquet", index=False
    )

    full_final_training_table = concat_tables(train_tables, sorted(train_tables))
    final_training_table = sample_training_candidates(
        full_final_training_table,
        max_negative_candidates_per_point=max_negatives,
        random_state=random_state,
    )
    print(
        f"[final] fitting on {len(final_training_table):,} rows "
        f"({len(full_final_training_table):,} before negative sampling)",
        flush=True,
    )
    final_started = time.perf_counter()
    final_model = spec.fit(final_training_table, config=model_parameters, feature_columns=feature_columns)
    final_training_seconds = time.perf_counter() - final_started
    joblib.dump(final_model, model_path, compress=int(config["outputs"].get("model_compression", 9)))

    fit_trip_metrics: list[dict[str, Any]] = []
    fit_prediction_tables: list[pd.DataFrame] = []
    for trip_id in sorted(train_tables):
        predictions, metrics = score_trip(
            train_tables[trip_id],
            spec=spec,
            partition="train",
            trip_id=trip_id,
            model=final_model,
            feature_columns=feature_columns,
            fold=folds[trip_id],
        )
        metrics["partition"] = "train_fit"
        fit_trip_metrics.append(metrics)
        fit_prediction_tables.append(predictions.loc[:, prediction_columns(predictions, spec.probability_column)])
    fit_by_trip = pd.DataFrame(fit_trip_metrics).sort_values("trip_id")
    fit_summary = aggregate_trip_metrics(fit_by_trip)
    fit_by_trip.to_csv(output_dir / "train_fit_by_trip.csv", index=False)
    fit_summary.to_csv(output_dir / "train_fit_summary.csv", index=False)
    pd.concat(fit_prediction_tables, ignore_index=True).to_parquet(
        output_dir / "train_fit_predictions.parquet", index=False
    )

    student_trip_metrics: list[dict[str, Any]] = []
    student_prediction_tables: list[pd.DataFrame] = []
    student_inference_seconds = 0.0
    for trip_id in sorted(student_tables):
        predictions, metrics = score_trip(
            student_tables[trip_id],
            spec=spec,
            partition="student_test",
            trip_id=trip_id,
            model=final_model,
            feature_columns=feature_columns,
            fold=None,
        )
        student_inference_seconds += float(metrics["inference_seconds"])
        student_trip_metrics.append(metrics)
        student_prediction_tables.append(predictions.loc[:, prediction_columns(predictions, spec.probability_column)])
    student_by_trip = pd.DataFrame(student_trip_metrics).sort_values("trip_id")
    student_summary = aggregate_trip_metrics(student_by_trip)
    student_by_trip.to_csv(output_dir / "student_test_by_trip.csv", index=False)
    student_summary.to_csv(output_dir / "student_test_summary.csv", index=False)
    pd.concat(student_prediction_tables, ignore_index=True).to_parquet(
        output_dir / "student_test_predictions.parquet", index=False
    )

    importance_settings = dict(config.get("feature_importance", {}))
    importance = feature_importance(
        final_model,
        final_training_table,
        feature_columns=feature_columns,
        sample_rows=int(importance_settings.get("sample_rows", 50000)),
        n_repeats=int(importance_settings.get("n_repeats", 3)),
        random_state=random_state,
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    cv_row = cv_summary.iloc[0].to_dict()
    student_row = student_summary.iloc[0].to_dict()
    comparison = pd.DataFrame([{
        "model": spec.display_name,
        "algorithm_family": config.get("algorithm_family"),
        "features": ",".join(feature_columns),
        "cv_top1_accuracy": cv_row.get("top1_accuracy"),
        "cv_trip_mean_top1_accuracy": cv_row.get("trip_mean_top1_accuracy"),
        "cv_candidate_macro_f1": cv_row.get("trip_mean_candidate_macro_f1"),
        "cv_point_mean_log_loss": cv_row.get("point_mean_log_loss"),
        "cv_route_continuity": cv_row.get("route_continuity"),
        "student_test_top1_accuracy": student_row.get("top1_accuracy"),
        "student_test_trip_mean_top1_accuracy": student_row.get("trip_mean_top1_accuracy"),
        "student_test_candidate_macro_f1": student_row.get("trip_mean_candidate_macro_f1"),
        "student_test_point_mean_log_loss": student_row.get("point_mean_log_loss"),
        "student_test_route_continuity": student_row.get("route_continuity"),
        "cross_validation_training_seconds": sum(row["training_seconds"] for row in fold_metrics),
        "final_training_seconds": final_training_seconds,
        "student_test_inference_seconds": student_inference_seconds,
        **{name: model_parameters.get(name) for name in spec.comparison_parameters},
    }])
    comparison.to_csv(output_dir / "comparison_row.csv", index=False)

    run_metadata = {
        "config": config,
        "resolved_feature_columns": feature_columns,
        "train_trips": len(train_tables),
        "student_test_trips": len(student_tables),
        "train_candidate_rows_before_sampling": len(full_final_training_table),
        "train_candidate_rows_used": len(final_training_table),
        "max_negative_candidates_per_point": max_negatives,
        "fold_ids": fold_ids,
        "feature_importance_method": "permutation importance, candidate-row log loss",
        "model_path": model_path.relative_to(REPO).as_posix() if model_path.is_relative_to(REPO) else str(model_path),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    print("Cross-validation summary:")
    print(cv_summary.to_string(index=False))
    print("\nFinal train-fit summary:")
    print(fit_summary.to_string(index=False))
    print("\nStudent-test summary:")
    print(student_summary.to_string(index=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved results: {output_dir}")
