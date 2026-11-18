import itertools
import unittest

import numpy as np
import pandas as pd

from scripts.evaluate_hmm import path_metrics
from src.model.hmm import DirectedRoadNetwork, build_lattice, decode_classifier, decode_viterbi


class HmmTests(unittest.TestCase):
    def setUp(self):
        self.arcs = np.array([[0, 1], [1, 2], [2, 3], [4, 5]])
        self.nodes = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [10, 2], [20, 2]], dtype=float)
        self.road = DirectedRoadNetwork(self.arcs, self.nodes, max_route_m=100)
        self.candidates = pd.DataFrame({
            "point_idx": [0, 1, 1, 2],
            "arc_id": [0, 3, 1, 2],
            "candidate_rank": [1, 1, 2, 1],
            "projection_fraction": [0.5, 0.5, 0.5, 0.5],
            "score": [-0.1, -0.05, -0.7, -0.1],
        })

    def lattice(self, table=None, timestamps=None):
        return build_lattice(
            self.candidates if table is None else table, self.road,
            np.array([[5, 0], [15, 0], [25, 0]], dtype=float),
            np.arange(3) if timestamps is None else np.array(timestamps),
            max_gap_s=10, backward_jitter_m=1,
        )

    def test_viterbi_agrees_with_exhaustive_scoring(self):
        lattice = self.lattice()
        self.assertEqual(decode_classifier(lattice).arc_ids.tolist(), [0, 3, 2])
        choices = list(itertools.product(*(range(len(layer.arc_ids)) for layer in lattice.layers)))

        def path_score(path):
            emission = sum(layer.emissions[index] for layer, index in zip(lattice.layers, path))
            penalty = sum(
                abs(lattice.distances[t][path[t - 1], path[t]] - lattice.gps_step_m[t]) / 20
                for t in range(1, len(path))
            )
            return emission - penalty

        result = decode_viterbi(lattice, beta_m=20)
        self.assertEqual(tuple(result.choices), max(choices, key=path_score))
        self.assertEqual(result.arc_ids.tolist(), [0, 1, 2])

    def test_directed_routes_include_partial_and_bridge_arcs(self):
        distance = self.road.distances(
            np.array([0, 2]), np.array([0.5, 0.5]),
            np.array([0, 2]), np.array([0.8, 0.5]), backward_jitter_m=1,
        )
        self.assertAlmostEqual(distance[0, 0], 3)
        self.assertAlmostEqual(distance[0, 1], 20)
        self.assertTrue(np.isinf(distance[1, 0]))

    def test_gaps_and_missing_candidates_preserve_index(self):
        table = self.candidates.loc[self.candidates["point_idx"].ne(1)]
        lattice = self.lattice(table, [0, 1, 14])
        result = decode_viterbi(lattice, beta_m=20)
        self.assertEqual(result.arc_ids.tolist(), [0, -1, 2])
        self.assertEqual(result.break_before, ("start", "unmatched", "gps_gap"))
        reference = pd.DataFrame({"point_idx": [0, 2], "arc_id": [0, 2]})
        metrics = path_metrics(result, lattice, self.arcs, reference)
        self.assertEqual((metrics["top1_correct"], metrics["points"]), (2, 3))
        self.assertEqual(metrics["evaluated_links"], 0)
        self.assertFalse(metrics["trip_exact_match"])

    def test_unreachable_layer_restarts_sequence(self):
        table = self.candidates.loc[self.candidates["arc_id"].ne(1)]
        result = decode_viterbi(self.lattice(table), beta_m=20)
        self.assertEqual(result.arc_ids.tolist(), [0, 3, 2])
        self.assertEqual(result.break_before, ("start", "no_transition", "no_transition"))
