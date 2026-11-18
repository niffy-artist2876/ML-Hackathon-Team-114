"""Shared helpers for Phase 4 candidate-level classifiers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.features import FEATURE_COLUMNS


def feature_matrix(
    table: pd.DataFrame,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return the numeric feature matrix used by every model fold.

    Boolean feature columns are converted to numbers and any values that
    cannot be represented numerically become NaN for the fold-fitted median
    imputer. This function never reads labels or route columns.
    """
    columns = list(feature_columns)
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Candidate table is missing feature columns: {missing}")
    matrix = table.loc[:, columns].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix.astype(float)


def training_labels(table: pd.DataFrame, model_name: str) -> np.ndarray:
    """Return binary labels after checking both classes are present."""
    if "label" not in table.columns:
        raise ValueError("Training candidate table must contain label")
    labels = pd.to_numeric(table["label"], errors="raise").to_numpy(dtype=int)
    unique_labels = set(np.unique(labels).tolist())
    if unique_labels != {0, 1}:
        raise ValueError(f"{model_name} training requires both labels 0 and 1; found {unique_labels}")
    return labels


def sample_training_candidates(
    table: pd.DataFrame,
    *,
    max_negative_candidates_per_point: int | None = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep all positives and a deterministic number of negatives per point.

    Sampling is applied only to fitting rows, never to validation or test
    rows. Groups without a positive candidate retain the configured number of
    negatives so the model still sees hard candidate-miss examples.
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
        # pandas' stable row hash avoids Python's process-randomized hash.
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
    model: Any,
    table: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Return the model probability for the positive candidate class."""
    probabilities = model.predict_proba(feature_matrix(table, feature_columns))
    classes = model.named_steps["classifier"].classes_.tolist()
    if 1 not in classes:
        raise ValueError(f"Fitted classifier has no positive class; classes={classes}")
    values = probabilities[:, classes.index(1)]
    return np.clip(values.astype(float), 1e-12, 1.0 - 1e-12)


def add_predictions(
    table: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    probability_column: str,
) -> pd.DataFrame:
    """Attach candidate scores and deterministic per-point selections."""
    if len(table) != len(probabilities):
        raise ValueError("Probability length does not match candidate table")
    result = table.copy()
    result[probability_column] = np.asarray(probabilities, dtype=float)
    # The shared evaluator applies a softmax to score. Log probability gives a
    # candidate-normalized distribution proportional to the model probabilities
    # while preserving the same argmax prediction.
    result["score"] = np.log(result[probability_column])
    result["selected"] = False
    ordered = result.sort_values(
        ["trip_id", "point_idx", probability_column, "candidate_rank", "arc_id"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    selected_indices = ordered.groupby(["trip_id", "point_idx"], sort=False).head(1).index
    result.loc[selected_indices, "selected"] = True
    return result
