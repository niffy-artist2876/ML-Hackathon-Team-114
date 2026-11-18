import unittest

import numpy as np
import pandas as pd

from src.model.candidate_classifier import add_predictions, positive_probability
from src.model.hist_gradient_boosting import fit_hist_gradient_boosting
from src.model.metrics import evaluate_candidate_table


class HistGradientBoostingTests(unittest.TestCase):
    def test_scores_unlabelled_features_and_matches_point_evaluator(self):
        table = pd.DataFrame({
            "trip_id": ["trip"] * 6,
            "point_idx": [0, 0, 1, 1, 2, 2],
            "arc_id": [11, 12, 12, 11, 12, 11],
            "candidate_rank": [1, 2, 1, 2, 1, 2],
            "distance_m": [0.1, 10.0, 0.2, 12.0, 0.3, 14.0],
            "label": [1, 0, 1, 0, 1, 0],
        })
        features = ["distance_m"]
        model = fit_hist_gradient_boosting(
            table,
            feature_columns=features,
            config={"max_iter": 3, "min_samples_leaf": 1, "random_state": 42},
        )
        probabilities = positive_probability(
            model, table.drop(columns="label"), feature_columns=features,
        )
        scored = add_predictions(table, probabilities, probability_col="hgb_probability")
        self.assertEqual(model.named_steps["classifier"].n_iter_, 3)
        self.assertTrue(np.isfinite(scored["score"]).all())
        self.assertEqual(scored.groupby(["trip_id", "point_idx"])["selected"].sum().tolist(), [1, 1, 1])
        self.assertEqual(evaluate_candidate_table(scored)["top1_correct"], 3)

    def test_rejects_internal_row_level_early_stopping(self):
        with self.assertRaisesRegex(ValueError, "early_stopping=false"):
            fit_hist_gradient_boosting(pd.DataFrame(), config={"early_stopping": "auto"})
