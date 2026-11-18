import unittest

import numpy as np
import pandas as pd

from src.model.random_forest import (
    add_predictions,
    fit_random_forest,
    positive_probability,
    sample_training_candidates,
)


class RandomForestTests(unittest.TestCase):
    def setUp(self):
        self.table = pd.DataFrame({
            "trip_id": ["t", "t", "t", "t"],
            "point_idx": [0, 0, 1, 1],
            "arc_id": [10, 20, 20, 30],
            "candidate_rank": [1, 2, 1, 2],
            "distance_m": [1.0, 8.0, 1.2, 7.0],
            "distance_gap_m": [0.0, 7.0, 0.0, 5.8],
            "arc_length_m": [10.0, 20.0, 10.0, 20.0],
            "projection_fraction": [0.2, 0.5, 0.3, 0.6],
            "heading_cos": [1.0, 0.0, 1.0, 0.0],
            "heading_valid": [True, True, True, True],
            "speed_mps": [5.0, 5.0, 6.0, 6.0],
            "speed_valid": [True, True, True, True],
            "time_gap_s": [0.0, 0.0, 1.0, 1.0],
            "large_gap": [False, False, False, False],
            "from_out_degree": [1, 2, 1, 2],
            "to_out_degree": [2, 1, 2, 1],
            "label": [1, 0, 1, 0],
        })

    def test_fit_predict_and_select_one_candidate_per_point(self):
        model = fit_random_forest(
            self.table,
            config={"n_estimators": 10, "n_jobs": 1, "random_state": 7},
        )
        probabilities = positive_probability(model, self.table)
        self.assertEqual(len(probabilities), len(self.table))
        self.assertTrue(np.isfinite(probabilities).all())
        predictions = add_predictions(self.table, probabilities)
        selected = predictions[predictions["selected"]]
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected.groupby(["trip_id", "point_idx"]).size().max(), 1)

    def test_missing_numeric_value_is_imputed_inside_pipeline(self):
        table = self.table.copy()
        table.loc[0, "speed_mps"] = np.nan
        model = fit_random_forest(
            table,
            config={"n_estimators": 10, "n_jobs": 1, "random_state": 7},
        )
        probabilities = positive_probability(model, table)
        self.assertEqual(len(probabilities), len(table))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_negative_sampling_keeps_positive_and_limits_negatives(self):
        table = pd.concat([self.table, self.table.assign(
            point_idx=0, arc_id=[40, 50, 60, 70], label=[0, 0, 0, 0]
        )], ignore_index=True)
        sampled = sample_training_candidates(table, max_negative_candidates_per_point=2, random_state=3)
        for _, group in sampled.groupby(["trip_id", "point_idx"]):
            self.assertLessEqual(int((group["label"] == 0).sum()), 2)
        self.assertEqual(int((sampled["label"] == 1).sum()), int((table["label"] == 1).sum()))


if __name__ == "__main__":
    unittest.main()
