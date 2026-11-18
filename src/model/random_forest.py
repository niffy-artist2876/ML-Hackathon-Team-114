"""Reusable Random Forest candidate scorer for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.features import FEATURE_COLUMNS


DEFAULT_RANDOM_FOREST_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


def feature_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return the numeric feature matrix used by every model fold.

    Boolean feature columns are explicitly converted to floats and any values
    that cannot be represented numerically become NaN for the fold-fitted
    median imputer. This function never reads labels or route columns.
    """
    columns = list(feature_columns)
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Candidate table is missing feature columns: {missing}")
    matrix = table.loc[:, columns].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def build_random_forest(
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build an imputer + Random Forest pipeline from a config mapping."""
    settings = dict(DEFAULT_RANDOM_FOREST_CONFIG)
    if config:
        settings.update(config)
    classifier = RandomForestClassifier(**settings)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def fit_random_forest(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Fit a candidate-level binary classifier on one training fold."""
    if "label" not in table.columns:
        raise ValueError("Training candidate table must contain label")
    labels = pd.to_numeric(table["label"], errors="raise").to_numpy(dtype=int)
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(f"Random Forest training requires both labels 0 and 1; found {unique_labels}")
    model = build_random_forest(config)
    model.fit(feature_matrix(table, feature_columns), labels)
    return model


def sample_training_candidates(
    table: pd.DataFrame,
    *,
    max_negative_candidates_per_point: int | None = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep all positives and a deterministic number of negatives per point.

    Candidate generation creates a highly imbalanced table: normally one
    candidate is positive and the remaining candidates are negative. Sampling
    is applied only to fitting rows, never to validation or test rows. Groups
    without a positive candidate retain the configured number of negatives so
    the model still sees hard candidate-miss examples.
    """
    if max_negative_candidates_per_point is None:
        return table.copy()
    if max_negative_candidates_per_point < 1:
        raise ValueError("max_negative_candidates_per_point must be positive or None")
    if "label" not in table.columns:
        raise ValueError("Training candidate table must contain label")

    labels = pd.to_numeric(table["label"], errors="raise").to_numpy(dtype=int)
    positives = table.loc[labels == 1].copy()
    negatives = table.loc[labels == 0].copy()
    if negatives.empty and positives.empty:
        raise ValueError("Cannot sample an empty candidate table")
    if not negatives.empty:
        # pandas' stable row hash avoids Python's process-randomized hash and
        # lets us sample all candidate groups without a Python loop.
        hash_columns = negatives[["trip_id", "point_idx", "arc_id"]]
        order = pd.util.hash_pandas_object(hash_columns, index=False).to_numpy(dtype="uint64")
        negatives["_sample_order"] = order ^ np.uint64(int(random_state))
        negatives = negatives.sort_values(
            ["trip_id", "point_idx", "_sample_order", "arc_id"],
            kind="mergesort",
        )
        negatives = negatives.groupby(
            ["trip_id", "point_idx"], sort=False, group_keys=False
        ).head(max_negative_candidates_per_point)
        negatives = negatives.drop(columns=["_sample_order"])
    return pd.concat([positives, negatives], ignore_index=True)


def positive_probability(
    model: Pipeline,
    table: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Return the model probability for the positive candidate class."""
    probabilities = model.predict_proba(feature_matrix(table, feature_columns))
    classifier = model.named_steps["classifier"]
    classes = classifier.classes_.tolist()
    if 1 not in classes:
        raise ValueError(f"Fitted classifier has no positive class; classes={classes}")
    values = probabilities[:, classes.index(1)]
    return np.clip(values.astype(float), 1e-12, 1.0 - 1e-12)


def add_predictions(
    table: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Attach candidate scores and deterministic per-point selections."""
    if len(table) != len(probabilities):
        raise ValueError("Probability length does not match candidate table")
    result = table.copy()
    result["rf_probability"] = np.asarray(probabilities, dtype=float)
    # The shared evaluator applies a softmax to score. Log probability gives
    # the evaluator a candidate-normalized distribution proportional to the RF
    # probabilities while preserving the same argmax prediction.
    result["score"] = np.log(result["rf_probability"])
    result["selected"] = False
    ordered = result.sort_values(
        ["trip_id", "point_idx", "rf_probability", "candidate_rank", "arc_id"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    selected_indices = ordered.groupby(["trip_id", "point_idx"], sort=False).head(1).index
    result.loc[selected_indices, "selected"] = True
    return result
