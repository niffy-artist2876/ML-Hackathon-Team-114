"""Reusable MLP candidate scorer for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FEATURE_COLUMNS
from src.model.candidate_classifier import (
    add_predictions,
    feature_matrix,
    positive_probability,
    sample_training_candidates,
    training_labels,
)

__all__ = [
    "DEFAULT_MLP_CONFIG",
    "add_predictions",
    "build_mlp",
    "feature_matrix",
    "fit_mlp",
    "positive_probability",
    "sample_training_candidates",
]


DEFAULT_MLP_CONFIG: dict[str, Any] = {
    "hidden_layer_sizes": [64, 32],
    "activation": "relu",
    "alpha": 0.0001,
    "batch_size": 512,
    "learning_rate_init": 0.001,
    "max_iter": 50,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 5,
    "random_state": 42,
}


def build_mlp(
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build an imputer + scaler + MLP pipeline from a config mapping.

    The MLP is sensitive to feature scale, so the scaler is fitted inside the
    pipeline on each training fold only.
    """
    settings = dict(DEFAULT_MLP_CONFIG)
    if config:
        settings.update(config)
    settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(**settings)),
    ])


def fit_mlp(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Fit a candidate-level binary MLP classifier on one training fold."""
    labels = training_labels(table, "MLP")
    model = build_mlp(config)
    model.fit(feature_matrix(table, feature_columns), labels)
    return model
