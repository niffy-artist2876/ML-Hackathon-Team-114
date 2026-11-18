"""Histogram Gradient Boosting candidate scorer for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.data.features import FEATURE_COLUMNS
from src.model.candidate_classifier import build_classifier_pipeline, fit_candidate_classifier


def fit_hist_gradient_boosting(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Fit boosting on complete training trips using a fixed round budget."""
    settings = dict(config)
    if settings.pop("early_stopping", False) is not False:
        raise ValueError("Use early_stopping=false to keep validation at the trip level")
    classifier = HistGradientBoostingClassifier(early_stopping=False, **settings)
    return fit_candidate_classifier(
        build_classifier_pipeline(classifier), table, feature_columns=feature_columns,
    )
