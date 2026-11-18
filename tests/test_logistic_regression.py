"""Unit tests for Logistic Regression model pipeline and scoring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import FEATURE_COLUMNS
from src.model.logistic_regression import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    attach_candidate_predictions,
    build_pipeline,
    load_model_artifact,
    prepare_features,
    save_model_artifact,
)


class LogisticRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Create a small synthetic candidate dataframe matching FEATURE_COLUMNS exactly
        np.random.seed(42)
        n_points = 5
        k_candidates = 4

        rows = []
        for p in range(n_points):
            for c in range(k_candidates):
                row = {
                    "trip_id": "00000001",
                    "point_idx": p,
                    "arc_id": 1000 + p * 10 + c,
                    "label": 1 if c == 0 else 0,
                    "distance_m": float(5.0 + c * 10.0 + np.random.uniform(-1, 1)),
                    "distance_gap_m": float(c * 10.0),
                    "candidate_rank": c + 1,
                    "arc_length_m": 50.0 + c * 5.0,
                    "projection_fraction": 0.25,
                    "heading_cos": float(np.cos(np.deg2rad(c * 20))),
                    "heading_valid": 1.0,
                    "speed_mps": 12.5,
                    "speed_valid": 1.0,
                    "time_gap_s": 1.0,
                    "large_gap": 0.0,
                    "from_out_degree": 2.0,
                    "to_out_degree": 3.0,
                }
                rows.append(row)
        self.sample_df = pd.DataFrame(rows)

    def test_feature_isolation_and_identifier_rejection(self) -> None:
        # Assert identifiers cannot be used as features
        for id_col in IDENTIFIER_COLUMNS:
            with self.assertRaises(ValueError):
                prepare_features(self.sample_df, feature_cols=[id_col, "distance_m"])

        # Assert target label cannot be used as feature
        with self.assertRaises(ValueError):
            prepare_features(self.sample_df, feature_cols=[TARGET_COLUMN, "distance_m"])

        # Assert clean feature extraction
        X = prepare_features(self.sample_df, feature_cols=FEATURE_COLUMNS)
        self.assertEqual(X.shape, (len(self.sample_df), len(FEATURE_COLUMNS)))
        self.assertTrue(np.isfinite(X).all())

    def test_non_finite_feature_rejection(self) -> None:
        bad_df = self.sample_df.copy()
        bad_df.loc[0, "distance_m"] = np.nan
        with self.assertRaises(ValueError):
            prepare_features(bad_df, feature_cols=FEATURE_COLUMNS)

        bad_df.loc[0, "distance_m"] = np.inf
        with self.assertRaises(ValueError):
            prepare_features(bad_df, feature_cols=FEATURE_COLUMNS)

    def test_pipeline_fit_predict_and_group_probabilities(self) -> None:
        pipeline = build_pipeline(C=1.0, max_iter=200, random_state=42, class_weight="balanced")
        X = prepare_features(self.sample_df, feature_cols=FEATURE_COLUMNS)
        y = self.sample_df[TARGET_COLUMN].to_numpy(dtype=int)

        pipeline.fit(X, y)

        # Inference on candidates (without label or route)
        unlabeled_df = self.sample_df.drop(columns=[TARGET_COLUMN])
        scored = attach_candidate_predictions(pipeline, unlabeled_df, feature_cols=FEATURE_COLUMNS)

        self.assertIn("score", scored.columns)
        self.assertIn("prob", scored.columns)
        self.assertIn("group_prob", scored.columns)

        # Probabilities must be within [0, 1]
        self.assertTrue((scored["prob"] >= 0.0).all())
        self.assertTrue((scored["prob"] <= 1.0).all())

        # Group probabilities must sum to 1.0 for each (trip_id, point_idx)
        group_sums = scored.groupby(["trip_id", "point_idx"])["group_prob"].sum()
        np.testing.assert_allclose(group_sums.values, 1.0, atol=1e-6)

    def test_artifact_serialization_roundtrip(self) -> None:
        pipeline = build_pipeline(C=0.5, max_iter=100, random_state=123)
        X = prepare_features(self.sample_df, feature_cols=FEATURE_COLUMNS)
        y = self.sample_df[TARGET_COLUMN].to_numpy(dtype=int)
        pipeline.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "test_model.joblib"
            save_model_artifact(
                pipeline,
                artifact_path,
                metadata={"test_run": True, "notes": "unit test artifact"},
            )
            self.assertTrue(artifact_path.is_file())

            loaded = load_model_artifact(artifact_path)
            self.assertIn("pipeline", loaded)
            self.assertEqual(loaded["metadata"]["notes"], "unit test artifact")

            # Check that predictions match exactly between original and loaded
            X_test = prepare_features(self.sample_df, feature_cols=FEATURE_COLUMNS)
            orig_preds = pipeline.predict_proba(X_test)
            loaded_preds = loaded["pipeline"].predict_proba(X_test)
            np.testing.assert_allclose(orig_preds, loaded_preds)


if __name__ == "__main__":
    unittest.main()
