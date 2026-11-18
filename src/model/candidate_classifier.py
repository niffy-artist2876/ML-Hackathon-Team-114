"""Shared training and scoring helpers for Phase 4 candidate classifiers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.features import FEATURE_COLUMNS


def feature_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return the numeric feature matrix used by every model fold.

    Non-numeric values become NaN for the fold-fitted median imputer. Only
    the named features are read.
    """
    columns = list(feature_columns)
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Candidate table is missing feature columns: {missing}")
    matrix = table.loc[:, columns].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def build_classifier_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Fit the same median imputation separately for each model fit."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def fit_candidate_classifier(
    model: Pipeline,
    table: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Fit a candidate-level binary classifier on one training fold."""
    if "label" not in table.columns:
        raise ValueError("Training candidate table must contain label")
    labels = pd.to_numeric(table["label"], errors="raise").to_numpy(dtype=int)
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(f"Training requires both labels 0 and 1; found {unique_labels}")
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
    *,
    probability_col: str,
) -> pd.DataFrame:
    """Attach candidate scores and deterministic per-point selections."""
    if len(table) != len(probabilities):
        raise ValueError("Probability length does not match candidate table")
    result = table.copy()
    result[probability_col] = np.asarray(probabilities, dtype=float)
    # The evaluator softmax-normalizes these log scores within each point.
    result["score"] = np.log(result[probability_col])
    result["selected"] = False
    ordered = result.sort_values(
        ["trip_id", "point_idx", probability_col, "candidate_rank", "arc_id"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    selected_indices = ordered.groupby(["trip_id", "point_idx"], sort=False).head(1).index
    result.loc[selected_indices, "selected"] = True
    return result
