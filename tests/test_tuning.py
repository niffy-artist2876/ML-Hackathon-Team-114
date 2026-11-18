import unittest

import numpy as np
import pandas as pd

from src.model.metrics import evaluate_candidate_table
from src.model.tuning import PreparedFold


class TuningTests(unittest.TestCase):
    def test_objective_matches_evaluator_with_ties_and_candidate_misses(self):
        rows = pd.DataFrame({
            "trip_id": ["a"] * 3 + ["b"] * 4,
            "point_idx": [0, 0, 0, 0, 0, 1, 1],
            "candidate_rank": [1, 2, 3, 1, 2, 1, 2],
            "arc_id": [10, 20, 30, 20, 13, 10, 22],
            "label": [1, 0, 0, 0, 0, 0, 1],
            "score": np.log([0.6, 0.6, 0.1, 0.1, 0.9, 0.3, 0.6]),
        })
        fold = PreparedFold(
            0, np.empty((0, 1)), np.array([]), np.empty((len(rows), 1)),
            rows["label"].to_numpy(), np.array([0, 3, 5]),
        )
        metrics = evaluate_candidate_table(rows)
        self.assertEqual(fold.point_count, metrics["point_count"])
        self.assertEqual(fold.correct_count(np.exp(rows["score"].to_numpy())), metrics["top1_correct"])
