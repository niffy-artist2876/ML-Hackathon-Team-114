"""Model-level HMM route reconstruction pipeline for PathSnap Phase 5."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data.loader import read_arcs
from src.evaluation import route_continuity
from src.hmm.viterbi import HMMRouteDecoder, build_transition_graph, decode_viterbi


REPO_ROOT = Path(__file__).resolve().parents[2]


class PathSnapHMM:
    """End-to-end HMM sequence reconstruction using frozen classifier emissions."""

    def __init__(
        self,
        classifier_path: Path | str = REPO_ROOT / "models" / "final_model" / "model.joblib",
        transition_penalty: float = -4.0,
        emission_weight: float = 1.0,
        large_gap_penalty: float = 0.0,
    ) -> None:
        self.classifier_path = Path(classifier_path)
        if not self.classifier_path.is_file():
            raise FileNotFoundError(f"Frozen classifier artifact not found: {self.classifier_path}")
        self.pipeline = joblib.load(self.classifier_path)

        schema_path = self.classifier_path.parent / "schema.json"
        if schema_path.is_file():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            self.features = schema.get("features", [])
        else:
            self.features = [
                "distance_m", "distance_gap_m", "candidate_rank", "arc_length_m",
                "projection_fraction", "heading_cos", "heading_valid", "speed_mps",
                "speed_valid", "time_gap_s", "large_gap", "from_out_degree", "to_out_degree",
            ]

        self.transition_penalty = transition_penalty
        self.emission_weight = emission_weight
        self.large_gap_penalty = large_gap_penalty
        self.decoder = HMMRouteDecoder(
            transition_penalty=transition_penalty,
            emission_weight=emission_weight,
            large_gap_penalty=large_gap_penalty,
            prob_col="prob",
        )

    def score_candidates(self, table: pd.DataFrame) -> pd.DataFrame:
        """Score candidate rows using the frozen HistGradientBoosting classifier."""
        df = table.copy()
        feat_cols = [c for c in self.features if c in df.columns]
        probabilities = self.pipeline.predict_proba(df[feat_cols])[:, 1]
        df["prob"] = probabilities
        return df

    def evaluate_trip(
        self,
        table: pd.DataFrame,
        arcs: pd.DataFrame,
        trip_id: str,
        partition: str = "train",
        fold: int = -1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate both classifier-only and classifier + HMM on a single trip.

        Returns:
            (metrics_classifier_only, metrics_hmm)
        """
        scored = self.score_candidates(table)

        # Ground truth route
        point_col = "point_idx" if "point_idx" in scored.columns else "point_index"
        group_dfs = [group_df for _, group_df in scored.groupby(point_col, sort=True)]
        truth = np.array([
            int(gdf.loc[gdf["label"].eq(1), "arc_id"].iloc[0])
            if "label" in gdf.columns and gdf["label"].any()
            else -1
            for gdf in group_dfs
        ], dtype=int)

        covered_points = int(np.sum(truth >= 0))
        total_points = len(truth)
        miss_rate = (total_points - covered_points) / total_points if total_points > 0 else 0.0

        # Decode
        start_time = time.perf_counter()
        clf_route, hmm_route, diag = self.decoder.decode_trip(scored, arcs)
        inference_time = time.perf_counter() - start_time

        # Accuracies
        clf_correct = int(np.sum((clf_route == truth) & (truth >= 0)))
        hmm_correct = int(np.sum((hmm_route == truth) & (truth >= 0)))

        clf_top1 = clf_correct / total_points if total_points > 0 else 0.0
        hmm_top1 = hmm_correct / total_points if total_points > 0 else 0.0

        clf_continuity = route_continuity(clf_route, arcs)
        hmm_continuity = route_continuity(hmm_route, arcs)

        # Macro F1
        labels = scored["label"].to_numpy(dtype=int) if "label" in scored.columns else None
        if labels is not None:
            clf_pos = np.zeros(len(scored), dtype=int)
            hmm_pos = np.zeros(len(scored), dtype=int)

            for t, gdf in enumerate(group_dfs):
                c_arc = clf_route[t]
                h_arc = hmm_route[t]
                c_matches = gdf.index[gdf["arc_id"] == c_arc]
                if len(c_matches) > 0:
                    clf_pos[c_matches[0]] = 1
                h_matches = gdf.index[gdf["arc_id"] == h_arc]
                if len(h_matches) > 0:
                    hmm_pos[h_matches[0]] = 1

            clf_f1 = float(f1_score(labels, clf_pos, average="macro", zero_division=0))
            hmm_f1 = float(f1_score(labels, hmm_pos, average="macro", zero_division=0))
        else:
            clf_f1 = np.nan
            hmm_f1 = np.nan

        metrics_clf = {
            "trip_id": trip_id,
            "partition": partition,
            "fold": fold,
            "decoder": "classifier_only",
            "point_count": total_points,
            "covered_points": covered_points,
            "candidate_miss_rate": miss_rate,
            "top1_correct": clf_correct,
            "top1_accuracy": clf_top1,
            "route_continuity": clf_continuity,
            "candidate_macro_f1": clf_f1,
            "disconnected_transitions": diag["classifier_disconnected"],
            "evaluated_transitions": diag["evaluated_transitions"],
            "inference_seconds": inference_time,
            "disagreements": 0,
            "disagreement_rate_percent": 0.0,
        }

        metrics_hmm = {
            "trip_id": trip_id,
            "partition": partition,
            "fold": fold,
            "decoder": "classifier_plus_hmm",
            "point_count": total_points,
            "covered_points": covered_points,
            "candidate_miss_rate": miss_rate,
            "top1_correct": hmm_correct,
            "top1_accuracy": hmm_top1,
            "route_continuity": hmm_continuity,
            "candidate_macro_f1": hmm_f1,
            "disconnected_transitions": diag["hmm_disconnected"],
            "evaluated_transitions": diag["evaluated_transitions"],
            "inference_seconds": inference_time,
            "disagreements": diag["disagreements"],
            "disagreement_rate_percent": diag["disagreement_rate_percent"],
        }

        return metrics_clf, metrics_hmm
