"""Inference service shared by the API and command-line checks."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loader import list_trips, read_arcs, read_nodes, read_route, read_track
from src.evaluation import route_continuity
from src.hmm.viterbi import build_transition_graph, decode_viterbi
from src.matching.features import FEATURE_COLUMNS, build_candidate_table
from src.matching.candidates import reference_arcs


class RoutePredictor:
    """Load model artifacts once and predict routes without exposing labels."""

    def __init__(self, artifact: Path | str, data_root: Path | str = "data/raw") -> None:
        self.data_root = Path(data_root)
        package = joblib.load(artifact)
        self.model = package["model"]
        self.feature_columns = package.get("feature_columns", FEATURE_COLUMNS)
        self.radius_m = float(package.get("config", {}).get("radius_m", 75.0))
        self.max_candidates = int(package.get("config", {}).get("max_candidates", 20))

    def available_trips(self) -> list[str]:
        return list_trips("student_test", self.data_root)

    def predict(self, trip_id: str) -> dict[str, object]:
        trip_id = str(trip_id).zfill(8)
        if trip_id not in self.available_trips():
            raise ValueError(f"Unknown student-test trip: {trip_id}")
        started = time.perf_counter()
        table = build_candidate_table(
            "student_test", trip_id, self.data_root, self.radius_m, self.max_candidates, include_label=False
        )
        table["prob"] = self.model.predict_proba(table[self.feature_columns])[:, 1]
        classifier = table.sort_values("prob").groupby("point_idx").tail(1).sort_values("point_idx")
        arcs = read_arcs("student_test", trip_id, self.data_root)
        hmm = decode_viterbi(table, build_transition_graph(arcs))
        predicted = [int(value) for value in hmm]
        route = read_route("student_test", trip_id, self.data_root)["arc_id"].to_numpy(dtype=int)
        references = reference_arcs(
            read_track("student_test", trip_id, self.data_root),
            read_nodes("student_test", trip_id, self.data_root),
            arcs,
            pd.DataFrame({"arc_id": route}),
        )
        metrics = {
            "points": len(predicted),
            "classifier_route_continuity": route_continuity(classifier["arc_id"].to_numpy(dtype=int), arcs),
            "hmm_route_continuity": route_continuity(hmm, arcs),
            "hmm_reference_accuracy": float(np.mean(hmm == references)),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "server_scored": True,
            "reference_available_server_side": bool(len(route)),
        }
        return {"trip_id": trip_id, "predicted_arcs": predicted, "metrics": metrics}
