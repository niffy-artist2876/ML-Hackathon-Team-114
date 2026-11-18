# Frozen Final Model Artifact — PathSnap

This directory contains the frozen, deployment-ready machine learning pipeline for PathSnap Phase 5 (HMM + Viterbi integration) and API serving.

## Frozen Model Specifications

- **Selected Model:** `HistGradientBoostingClassifier` within an `sklearn.pipeline.Pipeline`
- **Artifact:** `model.joblib` (377 KB binary)
- **Pre-processing:** `SimpleImputer(strategy='median')`
- **Features (13):**
  - `distance_m`: Perpendicular distance from GPS point to candidate road arc
  - `distance_gap_m`: Difference in distance relative to nearest candidate
  - `candidate_rank`: Proximity rank of candidate
  - `arc_length_m`: Metric length of road segment
  - `projection_fraction`: Position of orthogonal projection along arc [0, 1]
  - `heading_cos`: Cosine difference between GPS heading and arc azimuth
  - `heading_valid`: Binary flag indicating reliable vehicle heading
  - `speed_mps`: Estimated vehicle velocity in meters/second
  - `speed_valid`: Binary flag indicating valid speed estimate
  - `time_gap_s`: Delta time between consecutive GPS pings
  - `large_gap`: Binary indicator for sensor blackout / timestamp jumps
  - `from_out_degree`: Graph connectivity out-degree of start node
  - `to_out_degree`: Graph connectivity out-degree of end node

## How to Load and Predict

```python
import joblib
import pandas as pd

# Load frozen pipeline
pipeline = joblib.load("models/final_model/model.joblib")

# Prepare features (DataFrame with 13 columns listed above)
# X_candidates: pd.DataFrame
probabilities = pipeline.predict_proba(X_candidates)[:, 1]
```

## Why This Model Won

1. **Top-Tier Generalisation:** 79.55% 5-fold CV Top-1, 82.72% on student-test held-out trips.
2. **Superior HMM Emission Calibration:** Best point log loss (0.5591 on test), preventing probability saturation in Viterbi trellis decoding.
3. **Ultra-Fast & Portable:** 377 KB binary size (loads instantly without Git LFS), 0.30s inference across 10 complete trips.
