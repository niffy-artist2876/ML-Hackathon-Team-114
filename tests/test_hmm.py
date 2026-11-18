"""Unit tests for Phase 5 HMM + Viterbi decoding."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.hmm.viterbi import (
    HMMRouteDecoder,
    build_transition_graph,
    compute_emission_log_probs,
    decode_viterbi,
)
from src.model.hmm import PathSnapHMM


class HMMViterbiTests(unittest.TestCase):
    def setUp(self):
        # Synthetic 4-node, 4-arc directed road network
        # Node 0 -> Node 1 (Arc 0)
        # Node 1 -> Node 2 (Arc 1)
        # Node 2 -> Node 3 (Arc 2)
        # Node 0 -> Node 3 (Arc 3, shortcut)
        self.arcs = pd.DataFrame({
            "from_node": [0, 1, 2, 0],
            "to_node": [1, 2, 3, 3],
        })
        self.graph = build_transition_graph(self.arcs)

    def test_emission_normalization(self):
        candidates = pd.DataFrame({
            "arc_id": [0, 1, 2],
            "prob": [0.2, 0.5, 0.3],
        })
        arcs, log_emiss = compute_emission_log_probs(candidates, prob_col="prob")
        self.assertEqual(len(arcs), 3)
        self.assertTrue(np.all(np.isfinite(log_emiss)))
        probs = np.exp(log_emiss)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_log_probability_stability_zeros(self):
        candidates = pd.DataFrame({
            "arc_id": [0, 1],
            "prob": [0.0, 0.0],
        })
        arcs, log_emiss = compute_emission_log_probs(candidates, prob_col="prob")
        self.assertTrue(np.all(np.isfinite(log_emiss)))
        self.assertAlmostEqual(log_emiss[0], log_emiss[1])

    def test_connected_road_transition(self):
        # Arc 0 (0->1) connects to Arc 1 (1->2)
        self.assertIn(1, self.graph.get(0, set()))
        # Self-transition is allowed
        self.assertIn(0, self.graph.get(0, set()))

    def test_disconnected_road_transition(self):
        # Arc 1 (1->2) does NOT connect to Arc 0 (0->1)
        self.assertNotIn(0, self.graph.get(1, set()))

    def test_viterbi_synthetic_example_fixes_noisy_point(self):
        # 3 GPS points along route 0 -> 1 -> 2
        # Point 0: Candidates [0 (prob=0.9), 3 (prob=0.1)]
        # Point 1: Candidate 1 is true road, but noisy GPS gave Arc 3 a slightly higher score:
        #          Candidates [1 (prob=0.4), 3 (prob=0.6)]
        # Point 2: Candidates [2 (prob=0.9), 3 (prob=0.1)]
        # Classifier-only will pick: [0, 3, 2] -> disconnected!
        # HMM Viterbi with transition constraint should pick: [0, 1, 2] -> continuous!
        candidates = pd.DataFrame({
            "point_idx": [0, 0, 1, 1, 2, 2],
            "arc_id": [0, 3, 1, 3, 2, 3],
            "prob": [0.9, 0.1, 0.4, 0.6, 0.9, 0.1],
            "large_gap": [False] * 6,
        })
        decoded = decode_viterbi(candidates, self.graph, transition_penalty=-4.0)
        self.assertEqual(decoded.tolist(), [0, 1, 2])

    def test_directionality_enforcement(self):
        # Arc 0 is 0->1, Arc 1 is 1->2
        # Traveling backwards 1 -> 0 is not in graph
        self.assertNotIn(0, self.graph.get(1, set()))

    def test_large_gps_gap_relaxes_penalty(self):
        # Point 0 on Arc 0. Point 1 jumps to Arc 2 (disconnected from 0).
        # When large_gap=True, Viterbi allows the jump without severe penalty.
        candidates = pd.DataFrame({
            "point_idx": [0, 1],
            "arc_id": [0, 2],
            "prob": [0.8, 0.8],
            "large_gap": [False, True],
        })
        decoded = decode_viterbi(candidates, self.graph, transition_penalty=-10.0, large_gap_penalty=0.0)
        self.assertEqual(decoded.tolist(), [0, 2])

    def test_single_observation_trip(self):
        candidates = pd.DataFrame({
            "point_idx": [0, 0],
            "arc_id": [1, 2],
            "prob": [0.3, 0.7],
        })
        decoded = decode_viterbi(candidates, self.graph)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0], 2)

    def test_empty_candidates(self):
        empty_df = pd.DataFrame(columns=["point_idx", "arc_id", "prob"])
        decoded = decode_viterbi(empty_df, self.graph)
        self.assertEqual(len(decoded), 0)

    def test_reproducibility(self):
        candidates = pd.DataFrame({
            "point_idx": [0, 0, 1, 1],
            "arc_id": [0, 1, 1, 2],
            "prob": [0.6, 0.4, 0.7, 0.3],
        })
        run1 = decode_viterbi(candidates, self.graph)
        run2 = decode_viterbi(candidates, self.graph)
        np.testing.assert_array_equal(run1, run2)

    def test_compatibility_with_frozen_model(self):
        model_path = Path("models/final_model/model.joblib")
        if not model_path.is_file():
            self.skipTest("Frozen model artifact not found")
        hmm = PathSnapHMM(classifier_path=model_path)
        self.assertIsNotNone(hmm.pipeline)
        self.assertEqual(len(hmm.features), 13)


if __name__ == "__main__":
    unittest.main()
