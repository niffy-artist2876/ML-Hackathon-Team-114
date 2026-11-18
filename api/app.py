"""FastAPI service for PathSnap route predictions and analytics."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.predict import RoutePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pathsnap.api")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / os.getenv("PATHSNAP_DATA_ROOT", "data/raw")

app = FastAPI(title="PathSnap API", version="2.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("PATHSNAP_UI_ORIGIN", "*")],
    allow_methods=["GET"],
    allow_headers=["*"],
)

predictor: RoutePredictor | None = None


def get_predictor() -> RoutePredictor:
    global predictor
    if predictor is None:
        predictor = RoutePredictor(DATA_ROOT)
    return predictor


@app.on_event("startup")
def load_predictor() -> None:
    get_predictor()
    logger.info("RoutePredictor initialized successfully with data root: %s", DATA_ROOT)


@app.get("/health")
def health() -> dict[str, object]:
    p = get_predictor()
    return {
        "status": "ok",
        "model_loaded": p is not None,
        "active_model": "hist_gradient_boosting",
        "available_trips_count": len(p.available_trips()) if p else 0,
        "version": "2.5.0",
    }


@app.get("/trips")
def trips(partition: str = "student_test") -> dict[str, object]:
    """Return list of available trips."""
    p = get_predictor()
    trip_ids = p.available_trips(partition=partition)
    rich_trips = []

    inv_path = ROOT / "results" / "tables" / "data_inventory.csv"
    if inv_path.is_file():
        inv_df = pd.read_csv(inv_path)
        part_inv = inv_df[inv_df["partition"] == partition]
        for tid, group in part_inv.groupby("trip_id"):
            t_row = group[group["file_type"] == "track"]
            a_row = group[group["file_type"] == "arcs"]
            r_row = group[group["file_type"] == "route"]
            rich_trips.append({
                "trip_id": str(tid),
                "partition": partition,
                "observations": int(t_row["rows"].iloc[0]) if not t_row.empty else 0,
                "network_arcs": int(a_row["rows"].iloc[0]) if not a_row.empty else 0,
                "route_arcs": int(r_row["rows"].iloc[0]) if not r_row.empty else 0,
                "status": "Ready for Matching",
            })
    else:
        for tid in trip_ids:
            rich_trips.append({
                "trip_id": tid,
                "partition": partition,
                "observations": 0,
                "status": "Ready for Matching",
            })

    return {
        "trip_ids": trip_ids,
        "trips": rich_trips,
        "total_trips": len(rich_trips),
        "partition": partition,
    }


@app.get("/dataset")
def dataset_inventory() -> dict[str, object]:
    inv_path = ROOT / "results" / "tables" / "data_inventory.csv"
    if not inv_path.is_file():
        return {"trips": []}

    df = pd.read_csv(inv_path)
    track_rows = df[df["file_type"] == "track"]
    arcs_rows = df[df["file_type"] == "arcs"]

    inventory = []
    for (part, tid), t_row in track_rows.groupby(["partition", "trip_id"]):
        a_row = arcs_rows[(arcs_rows["partition"] == part) & (arcs_rows["trip_id"] == tid)]
        inventory.append({
            "trip_id": str(tid),
            "partition": part,
            "observations": int(t_row["rows"].iloc[0]),
            "arcs": int(a_row["rows"].iloc[0]) if not a_row.empty else 0,
        })
    return {"total_trips": len(inventory), "trips": inventory}


@app.get("/models")
def get_models_metadata() -> dict[str, object]:
    """Return model selection options with verified benchmarks."""
    return {
        "models": [
            {
                "id": "hist_gradient_boosting",
                "name": "HistGradientBoosting",
                "display_name": "HistGradientBoosting (Phase 4 & 5 Winner)",
                "badge": "PRODUCTION WINNER",
                "family": "Gradient Boosted Trees",
                "cv_top1": "79.55%",
                "test_top1": "82.72%",
                "continuity": "88.34%",
                "point_log_loss": "0.5591",
                "size": "377 KB",
                "speed": "0.30s",
                "description": "Optimal balance of accuracy, sharpest HMM emission probability calibration, and instant inference.",
            },
            {
                "id": "random_forest",
                "name": "Random Forest",
                "display_name": "Random Forest (100 Trees Ensemble)",
                "badge": "BENCHMARK LEADER",
                "family": "Bagged Tree Ensemble",
                "cv_top1": "80.13%",
                "test_top1": "83.73%",
                "continuity": "90.18%",
                "point_log_loss": "0.5981",
                "size": "101 MB",
                "speed": "0.69s",
                "description": "Highest raw accuracy baseline; robust non-linear feature interaction capture.",
            },
            {
                "id": "logistic_regression",
                "name": "Logistic Regression",
                "display_name": "Calibrated Logistic Regression",
                "badge": "LINEAR BASELINE",
                "family": "Generalized Linear Model",
                "cv_top1": "74.16%",
                "test_top1": "78.63%",
                "continuity": "89.49%",
                "point_log_loss": "0.7697",
                "size": "2.7 KB",
                "speed": "1.90s",
                "description": "Interpretable linear probability baseline with group-softmax normalization.",
            },
        ],
    }


@app.get("/evaluation")
def get_evaluation_metrics() -> dict[str, object]:
    """Return verified benchmarks from quick_comparison.csv and classifier_vs_hmm.csv."""
    results: dict[str, object] = {}

    benchmark_csv = ROOT / "results" / "final_model_benchmark" / "quick_comparison.csv"
    if benchmark_csv.is_file():
        df = pd.read_csv(benchmark_csv)
        results["benchmark_models"] = df.to_dict(orient="records")

    hmm_comp_csv = ROOT / "results" / "phase5_hmm" / "classifier_vs_hmm.csv"
    if hmm_comp_csv.is_file():
        df_hmm = pd.read_csv(hmm_comp_csv)
        results["hmm_comparison"] = df_hmm.to_dict(orient="records")

    return results


@app.get("/predict/{trip_id}")
def predict(
    trip_id: str,
    model: str = Query("hist_gradient_boosting", description="Model ID to score candidates"),
    decoder: str = Query("hmm", description="Sequence decoder: hmm or classifier_only"),
    partition: str = Query("student_test", description="Dataset partition: student_test or train"),
) -> dict[str, object]:
    p = get_predictor()
    try:
        result = p.predict(trip_id, partition=partition, model_id=model, decoder=decoder)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info("trip_id=%s model=%s decoder=%s latency_ms=%s",
                trip_id, model, decoder, result["metrics"]["latency_ms"])
    return result
