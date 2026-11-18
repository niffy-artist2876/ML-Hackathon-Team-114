"""FastAPI service for route predictions.

Ground-truth route data is read only inside the server-side predictor and is
never included in the JSON response.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.predict import RoutePredictor


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pathsnap.api")
ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / os.getenv("PATHSNAP_ARTIFACT", "results/models/final_extra_trees.joblib")
DATA_ROOT = ROOT / os.getenv("PATHSNAP_DATA_ROOT", "data/raw")

app = FastAPI(title="PathSnap API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("PATHSNAP_UI_ORIGIN", "*")],
    allow_methods=["GET"],
    allow_headers=["*"],
)
predictor: RoutePredictor | None = None


@app.on_event("startup")
def load_artifacts() -> None:
    global predictor
    if not ARTIFACT.exists():
        logger.warning("Model artifact is missing: %s", ARTIFACT)
        return
    predictor = RoutePredictor(ARTIFACT, DATA_ROOT)
    logger.info("Loaded model artifact once at startup")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/trips")
def trips() -> dict[str, list[str]]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")
    return {"trip_ids": predictor.available_trips()}


@app.get("/predict/{trip_id}")
def predict(trip_id: str) -> dict[str, object]:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded")
    try:
        result = predictor.predict(trip_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("trip_id=%s points=%s latency_ms=%s", result["trip_id"], result["metrics"]["points"], result["metrics"]["latency_ms"])
    return result
