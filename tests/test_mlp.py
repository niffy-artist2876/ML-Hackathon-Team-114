import unittest

import numpy as np
import pandas as pd

from src.model.mlp import (
    add_predictions,
    fit_mlp,
    positive_probability,
    sample_training_candidates,
)


def candidate_table(points=30, candidates=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for point in range(points):
        distances = np.sort(rng.uniform(0.5, 25.0, size=candidates))
        for rank in range(candidates):
            rows.append({
                "trip_id": "t", "point_idx": point, "arc_id": point * candidates + rank,
                "candidate_rank": rank + 1, "distance_m": distances[rank],
                "distance_gap_m": distances[rank] - distances[0], "arc_length_m": 20.0,
                "projection_fraction": rng.random(),
                "heading_cos": 1.0 if rank == 0 else rng.uniform(-1.0, 0.2),
                "heading_valid": True, "speed_mps": 5.0, "speed_valid": True,
                "time_gap_s": 1.0, "large_gap": False,
                "from_out_degree": 2, "to_out_degree": 2, "label": int(rank == 0),
            })
    return pd.DataFrame(rows)


SMALL_MLP = {
    "hidden_layer_sizes": [8], "batch_size": 32, "learning_rate_init": 0.01,
    "max_iter": 300, "early_stopping": False, "random_state": 7,
}


class MlpTests(unittest.TestCase):
    def setUp(self):
        self.table = candidate_table()

    def test_fit_predict_and_select_one_candidate_per_point(self):
        model = fit_mlp(self.table, config=SMALL_MLP)
        probabilities = positive_probability(model, self.table)
        self.assertEqual(len(probabilities), len(self.table))
        self.assertTrue(np.isfinite(probabilities).all())
        predictions = add_predictions(self.table, probabilities, probability_column="mlp_probability")
        selected = predictions[predictions["selected"]]
        self.assertEqual(len(selected), self.table["point_idx"].nunique())
        self.assertEqual(selected.groupby(["trip_id", "point_idx"]).size().max(), 1)
        self.assertGreater(float(selected["label"].mean()), 0.9)

    def test_missing_numeric_value_is_imputed_inside_pipeline(self):
        table = self.table.copy()
        table.loc[0, "speed_mps"] = np.nan
        model = fit_mlp(table, config=SMALL_MLP)
        probabilities = positive_probability(model, table)
        self.assertEqual(len(probabilities), len(table))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_scaler_is_fitted_inside_pipeline(self):
        model = fit_mlp(self.table, config=SMALL_MLP)
        self.assertEqual(list(model.named_steps), ["imputer", "scaler", "classifier"])

    def test_negative_sampling_keeps_positive_and_limits_negatives(self):
        sampled = sample_training_candidates(self.table, max_negative_candidates_per_point=2, random_state=3)
        for _, group in sampled.groupby(["trip_id", "point_idx"]):
            self.assertLessEqual(int((group["label"] == 0).sum()), 2)
        self.assertEqual(int((sampled["label"] == 1).sum()), int((self.table["label"] == 1).sum()))


if __name__ == "__main__":
    unittest.main()
