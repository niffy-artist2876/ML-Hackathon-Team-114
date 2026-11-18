"""Random Forest candidate scorer for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.data.features import FEATURE_COLUMNS
from src.model import candidate_classifier
from src.model.candidate_classifier import (
    feature_matrix,
    positive_probability,
    sample_training_candidates,
)


DEFAULT_RANDOM_FOREST_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


def build_random_forest(config: Mapping[str, Any] | None = None) -> Pipeline:
    settings = DEFAULT_RANDOM_FOREST_CONFIG | dict(config or {})
    return candidate_classifier.build_classifier_pipeline(RandomForestClassifier(**settings))


def fit_random_forest(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    return candidate_classifier.fit_candidate_classifier(
        build_random_forest(config), table, feature_columns=feature_columns,
    )


def add_predictions(table: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    return candidate_classifier.add_predictions(table, probabilities, probability_col="rf_probability")
