import unittest

import numpy as np
import pandas as pd

from src.model.metrics import aggregate_trip_metrics, evaluate_candidate_table, group_softmax


def sample_table():
    return pd.DataFrame({
        "trip_id": ["00000001"] * 4,
        "point_idx": [0, 0, 1, 1],
        "arc_id": [10, 11, 11, 10],
        "candidate_rank": [1, 2, 1, 2],
        "distance_m": [1.0, 5.0, 1.0, 5.0],
        "label": [1, 0, 1, 0],
        "large_gap": [False] * 4,
    })


class EvaluationTests(unittest.TestCase):
    def test_group_softmax_sums_to_one(self):
        table = sample_table()
        probabilities = group_softmax(table, score_col="distance_m")
        sums = table.assign(probability=probabilities).groupby(["trip_id", "point_idx"])["probability"].sum()
        np.testing.assert_allclose(sums.to_numpy(), [1.0, 1.0])

    def test_metrics_select_connected_best_candidates(self):
        table = sample_table()
        arcs = np.zeros((12, 2), dtype=int)
        arcs[10] = [0, 1]
        arcs[11] = [1, 2]
        result = evaluate_candidate_table(table.assign(score=-table["distance_m"]), arcs=arcs)
        self.assertEqual(result["point_count"], 2)
        self.assertEqual(result["missed_points"], 0)
        self.assertEqual(result["top1_correct"], 2)
        self.assertAlmostEqual(result["top1_accuracy"], 1.0)
        self.assertAlmostEqual(result["route_continuity"], 1.0)

    def test_candidate_miss_is_reported(self):
        table = sample_table()
        table.loc[table["point_idx"] == 1, "label"] = 0
        result = evaluate_candidate_table(table.assign(score=-table["distance_m"]))
        self.assertEqual(result["covered_points"], 1)
        self.assertEqual(result["missed_points"], 1)
        self.assertAlmostEqual(result["candidate_miss_rate"], 0.5)

    def test_summary_uses_point_weighted_accuracy(self):
        table = sample_table()
        metrics = evaluate_candidate_table(table.assign(score=-table["distance_m"]))
        trip_metrics = pd.DataFrame([{"partition": "train", "trip_id": "00000001", **metrics}])
        summary = aggregate_trip_metrics(trip_metrics).iloc[0]
        self.assertEqual(summary["points"], 2)
        self.assertAlmostEqual(summary["top1_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
