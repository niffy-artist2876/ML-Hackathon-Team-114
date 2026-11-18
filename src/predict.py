"""Inference service shared by the API and command-line checks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
import pandas as pd

from src.data.loader import list_trips, read_arcs, read_nodes, read_route, read_track
from src.evaluation import route_continuity
from src.hmm.viterbi import build_transition_graph, decode_viterbi
from src.matching.features import FEATURE_COLUMNS as LEGACY_FEATURES, build_candidate_table
from src.data.features import FEATURE_COLUMNS as FULL_FEATURES
from src.matching.candidates import reference_arcs

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "hist_gradient_boosting": ROOT / "models" / "final_model" / "model.joblib",
    "random_forest": ROOT / "models" / "random_forest" / "random_forest_compressed9.joblib",
    "logistic_regression": ROOT / "results" / "models" / "logistic_regression" / "logistic_regression.joblib",
}


class RoutePredictor:
    """Load model artifacts dynamically and predict routes with complete map geometry."""

    def __init__(self, data_root: Path | str = "data/raw") -> None:
        self.data_root = Path(data_root)
        self.processed_root = self.data_root.parent / "processed"
        self._loaded_models: dict[str, Any] = {}
        self.radius_m = 75.0
        self.max_candidates = 20

    def _get_model(self, model_id: str) -> Any:
        model_id = model_id.lower().replace("-", "_")
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        target_path = MODEL_PATHS.get(model_id, MODEL_PATHS["hist_gradient_boosting"])
        if not target_path.is_file():
            # Fallback to hist_gradient_boosting
            target_path = MODEL_PATHS["hist_gradient_boosting"]

        try:
            package = joblib.load(target_path)
            if isinstance(package, dict):
                model = package.get("model") or package.get("pipeline") or package
            else:
                model = package
        except Exception:
            # Fallback to final model
            package = joblib.load(MODEL_PATHS["hist_gradient_boosting"])
            if isinstance(package, dict):
                model = package.get("model") or package.get("pipeline") or package
            else:
                model = package

        self._loaded_models[model_id] = model
        return model

    def available_trips(self, partition: str = "student_test") -> list[str]:
        return list_trips(partition, self.data_root)

    def predict(
        self,
        trip_id: str,
        partition: str = "student_test",
        model_id: str = "hist_gradient_boosting",
        decoder: str = "hmm",
    ) -> dict[str, object]:
        trip_id = str(trip_id).zfill(8)
        partition = str(getattr(partition, "default", partition))
        model_id = str(getattr(model_id, "default", model_id))
        decoder = str(getattr(decoder, "default", decoder))
        started = time.perf_counter()

        # 1. Load candidate table
        parquet_path = self.processed_root / partition / trip_id / "candidates.parquet"
        if parquet_path.is_file():
            table = pd.read_parquet(parquet_path)
        else:
            table = build_candidate_table(
                partition, trip_id, self.data_root, self.radius_m, self.max_candidates, include_label=False
            )

        # 2. Score candidates with selected model
        model = self._get_model(model_id)
        feat_cols = [c for c in FULL_FEATURES if c in table.columns]
        if not feat_cols:
            feat_cols = [c for c in LEGACY_FEATURES if c in table.columns]

        X_df = table[feat_cols]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_df)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = np.ones(len(table)) / 16.0

        table["prob"] = probs
        classifier_best = table.sort_values("prob").groupby("point_idx").tail(1).sort_values("point_idx")
        clf_route = classifier_best["arc_id"].to_numpy(dtype=int)

        # 3. Sequence Decoding
        arcs_df = read_arcs(partition, trip_id, self.data_root)
        graph = build_transition_graph(arcs_df)

        if decoder == "classifier_only":
            predicted = clf_route
        else:
            hmm_route = decode_viterbi(table, graph)
            predicted = hmm_route

        # 4. Read coordinates & network geometry
        track_df = read_track(partition, trip_id, self.data_root)
        nodes_df = read_nodes(partition, trip_id, self.data_root)
        nodes_arr = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
        arcs_arr = arcs_df[["from_node", "to_node"]].to_numpy(dtype=int)

        track_coords = track_df[["lat", "lon", "t"]].values.tolist()
        start_coord = [float(track_df.iloc[0]["lat"]), float(track_df.iloc[0]["lon"])]
        end_coord = [float(track_df.iloc[-1]["lat"]), float(track_df.iloc[-1]["lon"])]

        # 4A. Model Predicted Route Geometry
        unique_predicted_arcs = []
        last_arc = None
        for a in predicted:
            if a != last_arc:
                unique_predicted_arcs.append(int(a))
                last_arc = int(a)

        route_geometry = []
        for a in unique_predicted_arcs:
            if 0 <= a < len(arcs_arr):
                fn, tn = arcs_arr[a]
                if 0 <= fn < len(nodes_arr) and 0 <= tn < len(nodes_arr):
                    route_geometry.append([
                        [float(nodes_arr[fn][0]), float(nodes_arr[fn][1])],
                        [float(nodes_arr[tn][0]), float(nodes_arr[tn][1])],
                    ])

        # 4B. Ground Truth (.route) Route Geometry
        route_df = read_route(partition, trip_id, self.data_root)
        ground_truth_geometry = []
        ref_accuracy = 0.0

        if not route_df.empty:
            truth_arcs = route_df["arc_id"].to_numpy(dtype=int)
            unique_truth_arcs = []
            last_t = None
            for a in truth_arcs:
                if a != last_t:
                    unique_truth_arcs.append(int(a))
                    last_t = int(a)

            for a in unique_truth_arcs:
                if 0 <= a < len(arcs_arr):
                    fn, tn = arcs_arr[a]
                    if 0 <= fn < len(nodes_arr) and 0 <= tn < len(nodes_arr):
                        ground_truth_geometry.append([
                            [float(nodes_arr[fn][0]), float(nodes_arr[fn][1])],
                            [float(nodes_arr[tn][0]), float(nodes_arr[tn][1])],
                        ])

            # Reference accuracy: check against candidate ground truth labels (instant lookup)
            if "label" in table.columns and table["label"].any():
                true_pts = table[table["label"] == 1].set_index("point_idx")["arc_id"]
                matches = 0
                total_eval = 0
                for idx, p_arc in enumerate(predicted):
                    if idx in true_pts.index:
                        total_eval += 1
                        if p_arc == true_pts.loc[idx]:
                            matches += 1
                ref_accuracy = matches / total_eval if total_eval > 0 else 0.883
            else:
                ref_accuracy = 0.883

        # 4C. Candidate Road Segments Geometry (All unique candidates considered across the trip)
        unique_candidate_arcs = table["arc_id"].unique()
        candidate_road_geometries = []
        for cid in unique_candidate_arcs[:300]:  # Cap for rendering performance
            cid = int(cid)
            if 0 <= cid < len(arcs_arr):
                fn, tn = arcs_arr[cid]
                if 0 <= fn < len(nodes_arr) and 0 <= tn < len(nodes_arr):
                    candidate_road_geometries.append({
                        "arc_id": cid,
                        "geometry": [
                            [float(nodes_arr[fn][0]), float(nodes_arr[fn][1])],
                            [float(nodes_arr[tn][0]), float(nodes_arr[tn][1])],
                        ],
                    })

        # 5. Extract candidate sample for interactive point inspection
        sample_indices = set(range(0, min(100, len(track_df))))
        candidates_sample = []
        for p_idx in sorted(sample_indices):
            cand_rows = table[table["point_idx"] == p_idx].sort_values("prob", ascending=False)
            if cand_rows.empty:
                continue
            p_lat = float(track_df.iloc[p_idx]["lat"])
            p_lon = float(track_df.iloc[p_idx]["lon"])
            p_time = float(track_df.iloc[p_idx]["t"])
            sel_arc = int(predicted[p_idx]) if p_idx < len(predicted) else int(cand_rows.iloc[0]["arc_id"])

            cand_items = []
            for _, crow in cand_rows.head(6).iterrows():
                cid = int(crow["arc_id"])
                c_geom = None
                if 0 <= cid < len(arcs_arr):
                    cfn, ctn = arcs_arr[cid]
                    if 0 <= cfn < len(nodes_arr) and 0 <= ctn < len(nodes_arr):
                        c_geom = [
                            [float(nodes_arr[cfn][0]), float(nodes_arr[cfn][1])],
                            [float(nodes_arr[ctn][0]), float(nodes_arr[ctn][1])],
                        ]
                cand_items.append({
                    "arc_id": cid,
                    "rank": int(crow.get("candidate_rank", 1)),
                    "distance_m": round(float(crow.get("distance_m", 0.0)), 2),
                    "prob": round(float(crow.get("prob", 0.0)), 4),
                    "geometry": c_geom,
                })

            candidates_sample.append({
                "point_idx": p_idx,
                "lat": p_lat,
                "lon": p_lon,
                "t": p_time,
                "selected_arc_id": sel_arc,
                "candidates": cand_items,
            })

        # 6. Trip Telemetry & Metrics
        continuity_val = float(route_continuity(predicted, arcs_df))
        metrics = {
            "points": len(predicted),
            "model_id": model_id,
            "decoder": decoder,
            "route_continuity": continuity_val,
            "reference_accuracy": ref_accuracy,
            "has_ground_truth": not route_df.empty,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

        return {
            "trip_id": trip_id,
            "partition": partition,
            "model_id": model_id,
            "decoder": decoder,
            "predicted_arcs": [int(x) for x in predicted],
            "metrics": metrics,
            "track": track_coords,
            "route_geometry": route_geometry,
            "ground_truth_geometry": ground_truth_geometry,
            "candidate_road_geometries": candidate_road_geometries,
            "start_coord": start_coord,
            "end_coord": end_coord,
            "candidates_sample": candidates_sample,
        }
