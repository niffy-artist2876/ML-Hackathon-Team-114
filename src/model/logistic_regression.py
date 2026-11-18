"""Supervised Logistic Regression pipeline for PathSnap candidate classification.

This module trains a binary classifier to score road segment candidates for each
GPS observation using the standard engineered candidate feature schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLUMNS
from src.model.metrics import group_softmax

IDENTIFIER_COLUMNS = ("trip_id", "point_idx", "arc_id")
TARGET_COLUMN = "label"


def build_pipeline(
    *,
    C: float = 1.0,
    max_iter: int = 300,
    random_state: int = 42,
    solver: str = "lbfgs",
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Build a scikit-learn Pipeline with StandardScaler and LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver=solver,
            class_weight=class_weight,
        )),
    ])


def prepare_features(
    table: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Extract and validate feature matrix, strictly excluding identifiers and labels."""
    for col in IDENTIFIER_COLUMNS:
        if col in feature_cols:
            raise ValueError(f"Identifier column '{col}' cannot be used as a model feature")
    if TARGET_COLUMN in feature_cols:
        raise ValueError(f"Target column '{TARGET_COLUMN}' cannot be used as a model feature")

    missing = set(feature_cols) - set(table.columns)
    if missing:
        raise ValueError(f"Table is missing required feature columns: {sorted(missing)}")

    X = table[list(feature_cols)].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains non-finite values (NaN or Inf)")
    return X


def attach_candidate_predictions(
    pipeline: Pipeline,
    candidate_table: pd.DataFrame,
    *,
    score_col: str = "score",
    prob_col: str = "prob",
    group_prob_col: str = "group_prob",
    feature_cols: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Score candidate rows with model decision function and normalized probabilities.

    - score: raw decision function logit (w^T x + b), monotonic with candidate ranking.
    - prob: binary model probability P(label=1 | x) via sigmoid.
    - group_prob: group-softmax normalized probability within each (trip_id, point_idx) group.
    """
    result = candidate_table.copy()
    X = prepare_features(result, feature_cols=feature_cols)

    logits = pipeline.decision_function(X)
    probs = pipeline.predict_proba(X)[:, 1]

    result[score_col] = logits
    result[prob_col] = probs
    result[group_prob_col] = group_softmax(result, score_col=score_col)
    return result


def save_model_artifact(
    pipeline: Pipeline,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Serialize the trained pipeline and metadata safely with joblib."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "feature_columns": list(FEATURE_COLUMNS),
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "target_column": TARGET_COLUMN,
        "metadata": metadata or {},
    }
    joblib.dump(payload, dest)


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Load a serialized model artifact and verify payload structure."""
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Model artifact not found: {src}")
    payload = joblib.load(src)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Invalid model artifact structure in {src}")
    return payload
