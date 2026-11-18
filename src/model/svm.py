"""Reusable SVM candidate scorer for Phase 4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.data.features import FEATURE_COLUMNS
from src.model.candidate_classifier import (
    add_predictions,
    feature_matrix,
    positive_probability,
    sample_training_candidates,
    fit_candidate_classifier,
)

__all__ = [
    "DEFAULT_SVM_CONFIG",
    "add_predictions",
    "build_svm",
    "feature_matrix",
    "fit_svm",
    "positive_probability",
    "sample_training_candidates",
]


DEFAULT_SVM_CONFIG: dict[str, Any] = {
    "kernel": "rbf",
    "gamma": 0.1,
    "n_components": 100,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
    "calibration_method": "sigmoid",
    "calibration_cv": 3,
    "random_state": 42,
}


def build_svm(
    config: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Build an imputer + scaler + approximate-RBF SVM pipeline.

    An exact kernel SVC scales roughly quadratically with training rows and
    is impractical for about a million candidate rows. A Nystroem feature map
    approximates the RBF kernel and a linear SVM is fitted in that space.
    LinearSVC has no probabilities, so it is wrapped in sigmoid (Platt)
    calibration to give the evaluator a usable log loss.
    """
    settings = dict(DEFAULT_SVM_CONFIG)
    if config:
        settings.update(config)
    kernel_map = Nystroem(
        kernel=settings["kernel"],
        gamma=settings["gamma"],
        n_components=settings["n_components"],
        random_state=settings["random_state"],
    )
    svm = LinearSVC(
        C=settings["C"],
        class_weight=settings["class_weight"],
        max_iter=settings["max_iter"],
        dual=False,
        random_state=settings["random_state"],
    )
    classifier = CalibratedClassifierCV(
        svm,
        method=settings["calibration_method"],
        cv=settings["calibration_cv"],
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("kernel_map", kernel_map),
        ("classifier", classifier),
    ])


def fit_svm(
    table: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None = None,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> Pipeline:
    """Fit a candidate-level binary SVM classifier on one training fold."""
    return fit_candidate_classifier(build_svm(config), table, feature_columns=feature_columns)
