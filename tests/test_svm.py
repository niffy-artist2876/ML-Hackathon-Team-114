import unittest

import numpy as np

from src.model.svm import (
    add_predictions,
    fit_svm,
    positive_probability,
    sample_training_candidates,
)
from tests.test_mlp import candidate_table


SMALL_SVM = {"n_components": 20, "calibration_cv": 3, "random_state": 7}


class SvmTests(unittest.TestCase):
    def setUp(self):
        self.table = candidate_table()

    def test_fit_predict_and_select_one_candidate_per_point(self):
        model = fit_svm(self.table, config=SMALL_SVM)
        probabilities = positive_probability(model, self.table)
        self.assertEqual(len(probabilities), len(self.table))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())
        predictions = add_predictions(self.table, probabilities, probability_column="svm_probability")
        selected = predictions[predictions["selected"]]
        self.assertEqual(len(selected), self.table["point_idx"].nunique())
        self.assertEqual(selected.groupby(["trip_id", "point_idx"]).size().max(), 1)
        self.assertGreater(float(selected["label"].mean()), 0.9)

    def test_missing_numeric_value_is_imputed_inside_pipeline(self):
        table = self.table.copy()
        table.loc[0, "speed_mps"] = np.nan
        model = fit_svm(table, config=SMALL_SVM)
        probabilities = positive_probability(model, table)
        self.assertEqual(len(probabilities), len(table))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_kernel_map_and_calibration_are_inside_pipeline(self):
        model = fit_svm(self.table, config=SMALL_SVM)
        self.assertEqual(list(model.named_steps), ["imputer", "scaler", "kernel_map", "classifier"])
        self.assertEqual(model.named_steps["kernel_map"].n_components, 20)

    def test_negative_sampling_keeps_positive_and_limits_negatives(self):
        sampled = sample_training_candidates(self.table, max_negative_candidates_per_point=2, random_state=3)
        for _, group in sampled.groupby(["trip_id", "point_idx"]):
            self.assertLessEqual(int((group["label"] == 0).sum()), 2)
        self.assertEqual(int((sampled["label"] == 1).sum()), int((self.table["label"] == 1).sum()))


if __name__ == "__main__":
    unittest.main()
