"""Cache the fixed train-only preprocessing and validation data for each fold."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.candidate_classifier import feature_matrix, sample_training_candidates
from src.model.random_forest import build_random_forest


@dataclass
class PreparedFold:
    fold: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    group_starts: np.ndarray

    @property
    def point_count(self) -> int:
        return len(self.group_starts)

    def correct_count(self, probabilities: np.ndarray) -> int:
        """Pick the max score in each point group, with rank/arc-order ties."""
        if len(probabilities) != len(self.y_validation) or not np.isfinite(probabilities).all():
            raise ValueError("Expected one finite probability per validation candidate")
        maxima = np.maximum.reduceat(probabilities, self.group_starts)
        widths = np.diff(self.group_starts, append=len(probabilities))
        rows = np.arange(len(probabilities))
        first_winner = np.minimum.reduceat(
            np.where(probabilities == np.repeat(maxima, widths), rows, len(rows)), self.group_starts
        )
        return int(self.y_validation[first_winner].sum())


def prepare_folds(
    tables: dict[str, pd.DataFrame],
    folds: dict[str, int],
    config: dict,
) -> list[PreparedFold]:
    """Fit each fold's imputer on its sampled training trips exactly once."""
    prepared = []
    features = config["feature_columns"]
    sample_settings = {
        "max_negative_candidates_per_point": config.get("max_negative_candidates_per_point", 4),
        "random_state": config["model_parameters"]["random_state"],
    }
    for fold in sorted(set(folds.values())):
        fit = pd.concat([tables[trip] for trip in sorted(tables) if folds[trip] != fold], ignore_index=True)
        sample = sample_training_candidates(fit, **sample_settings)
        del fit
        validation = pd.concat(
            [tables[trip] for trip in sorted(tables) if folds[trip] == fold], ignore_index=True,
        ).sort_values(["trip_id", "point_idx", "candidate_rank", "arc_id"])
        imputer = build_random_forest(config["model_parameters"]).named_steps["imputer"]
        x_train = imputer.fit_transform(feature_matrix(sample, features)).astype(np.float32)
        y_train = sample["label"].to_numpy(dtype=np.int8)
        x_validation = imputer.transform(feature_matrix(validation, features)).astype(np.float32)
        y_validation = validation["label"].to_numpy(dtype=np.int8)
        if set(np.unique(y_train)) != {0, 1} or not np.isin(y_validation, [0, 1]).all():
            raise ValueError(f"Invalid candidate labels in fold {fold}")
        starts = np.flatnonzero(~validation.duplicated(["trip_id", "point_idx"]).to_numpy())
        prepared.append(PreparedFold(fold, x_train, y_train, x_validation, y_validation, starts))
        print(f"[data] fold {fold}: {len(y_train):,} fitting rows, {len(starts):,} validation points", flush=True)
    return prepared
