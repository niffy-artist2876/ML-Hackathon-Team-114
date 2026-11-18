# PathSnap — Official Evaluator Test Benchmark & Trip Results

**Hackathon Evaluation Partition**: `evaluator_test`  
**Team**: Team 114 — PathSnap Geospatial AI  
**Team Members**:
- Swara Ashish Potdar (`PES1UG24CS487`) — Machine Learning & Pipeline Engineering
- Shaurya Singh (`PES1UG24CS437`) — Geospatial Processing & Evaluation
- Shishir Hegde (`PES1UG24CS438`) — Optimization & Sequence Modeling
- Sushruth Sukesh (`PES1UG24CS486`) — Model Diagnostics & API Infrastructure

---

## 1. Overview & Evaluation Protocol

This directory contains the official evaluation results and matched trajectory predictions for the **10 reserved evaluation trips** (`evaluator_test.zip`) released by the hackathon committee.

### Partition Inventory
- **Trips Evaluated**: 10 distinct vehicle journeys (`00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`).
- **Total GPS Points**: **19,674 observations**.
- **Candidate Pool**: 314,784 candidate road segments evaluated across all trips (top $K=16$ spatial candidates per point).
- **Execution Protocol**: Strict blind evaluation using frozen production models. Zero parameter adjustment, zero threshold tuning, and zero retraining were performed.

---

## 2. Multi-Model Benchmark on Evaluator Trips

All models were evaluated under identical conditions on the 10 evaluator trips:

| Model Architecture & Decoder | Point-Weighted Top-1 | Trip-Mean Top-1 | Route Continuity | Candidate Macro F1 | Disconnected Jumps | Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Distance Baseline (Phase 3)** | 49.10% | 53.22% | 78.44% | 75.04% | 3,404 | 0.50 s |
| **🏆 HistGradientBoosting + HMM (Winner)** | **76.78%** | **85.44%** | **98.59%** | **92.22%** | **161** | **42.28 s** |
| **HistGradientBoosting (Classifier-Only)** | 73.82% | 82.73% | 88.58% | 90.78% | 2,050 | 25.45 s |
| **Random Forest + HMM** | 76.78% | 85.44% | 98.59% | 92.22% | 161 | 38.91 s |
| **Random Forest (Classifier-Only)** | 73.82% | 82.73% | 88.58% | 90.78% | 2,050 | 11.69 s |
| **Logistic Regression + HMM** | 71.56% | 79.43% | 99.48% | 89.02% | 54 | 20.66 s |
| **Logistic Regression (Classifier-Only)** | 70.23% | 78.02% | 88.46% | 88.26% | 1,978 | 5.12 s |

---

## 3. Detailed Per-Trip Breakdown (`HistGradientBoosting + HMM`)

Performance metrics of our winning production pipeline on each of the 10 evaluator journeys:

| Trip ID | GPS Observations | Point Top-1 | Route Continuity | Candidate Macro F1 | Disconnected Transitions | Inference Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000015` | 6,854 pts | 53.92% | **99.99%** | 75.43% | **1** | 12.07 s |
| `00000017` | 1,346 pts | 86.73% | **94.50%** | 92.84% | 74 | 3.87 s |
| `00000023` | 458 pts | 93.44% | **99.34%** | 96.44% | 3 | 3.52 s |
| `00000030` | 654 pts | 78.44% | **96.02%** | 88.50% | 26 | 2.80 s |
| `00000049` | 2,642 pts | 97.35% | **99.81%** | 98.59% | 5 | 5.37 s |
| `00000059` | 1,031 pts | 93.11% | **97.38%** | 96.30% | 27 | 2.32 s |
| `00000067` | 916 pts | 78.93% | **100.00%** | 88.76% | **0** | 1.62 s |
| `00000073` | 1,144 pts | 93.27% | **99.65%** | 96.41% | 4 | 2.41 s |
| `00000087` | 1,399 pts | 98.86% | **99.71%** | 99.39% | 4 | 2.99 s |
| `00000094` | 3,230 pts | 80.40% | **99.47%** | 89.55% | 17 | 5.32 s |
| **All 10 Trips Combined** | **19,674 pts** | **76.78% (wtd)** / **85.44% (mean)** | **98.59%** | **92.22%** | **161 total** | **42.28 s** |

---

## 4. Key Sequence Reconstruction Insights

### Disconnected Transitions Slashed by 92.1%
- **Without Sequence Optimization** (Classifier-Only Argmax): The classifier predictions suffered from **2,050 disconnected jumps** across the 10 trips due to noisy GPS scatter near parallel roads and overpasses.
- **With HMM / Viterbi Decoder**: Disconnected jumps plummeted to just **161 transitions** across all 19,674 observations, achieving **98.59% route continuity**.
- On Trip `00000067` (916 points), the pipeline achieved a **perfect 100.00% continuous route** with **0 disconnected transitions**.
- On Trip `00000015` (the longest trip with 6,854 points), continuity was **99.985%** with only 1 single transition jump across nearly 7,000 points.

---

## 5. Strict Zero Overfitting & Generalization Guarantees

1. **Frozen Production Model**: The `HistGradientBoostingClassifier` model artifact (`models/final_model/model.joblib`, 377 KB) was trained strictly on the training partition and was never re-fit or fine-tuned on the evaluator partition.
2. **True Out-of-Distribution Validation**: The evaluator partition spans distinct geographical regions and urban corridors not present in training data. The strong **85.44% Trip-Mean Top-1** accuracy confirms robust inductive generalization without coordinate memorization.
3. **Calibrated Probability Emissions**: HistGradientBoosting's low log loss ensures well-calibrated emission probabilities, allowing the Viterbi dynamic programming transitions to smoothly track road adjacency without erratic score swings.

---

## 6. Directory Files

- `README.md`: This comprehensive documentation and benchmark report.
- `predictions.csv`: Complete matched predictions table containing `trip_id`, `point_idx`, and `predicted_arc_id` for all 19,674 points.
- `summary.csv`: Macro-level benchmark metrics for all models and sequence decoders.
- `by_trip.csv`: Exhaustive per-trip metrics breakdown across each individual trip and model configuration.

---

## 7. How to Reproduce

To regenerate all metrics and prediction files from scratch:

```powershell
# Navigate to the repository root
cd ML-Hackathon-Team-114

# Run the official evaluator benchmark
.\.venv\Scripts\python.exe scripts/evaluate_evaluator_test.py
```
