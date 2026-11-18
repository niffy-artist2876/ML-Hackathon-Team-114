# Supervised Logistic Regression Model Report

## 1. Model Architecture & Preprocessing
- **Algorithm**: `LogisticRegression` (scikit-learn) with `solver='lbfgs'`.
- **Scaling**: `StandardScaler` fitted strictly on training candidate rows inside each fold (zero data leakage).
- **Hyperparameters**: $C = 1.0$, $\text{class\_weight} = 'balanced'$, $\text{max\_iter} = 300$, $\text{random\_state} = 42$.
- **Features Used (13)**: distance_m, distance_gap_m, candidate_rank, arc_length_m, projection_fraction, heading_cos, heading_valid, speed_mps, speed_valid, time_gap_s, large_gap, from_out_degree, to_out_degree.
- **Features Excluded**: `trip_id`, `point_idx`, `arc_id`, and `label`.

## 2. Learned Feature Weights (Standardized)
```text
distance_gap_m        -11.794860
candidate_rank         -1.754414
from_out_degree        -0.168354
speed_mps              -0.144453
heading_valid          -0.024556
time_gap_s             -0.000401
speed_valid            -0.000260
large_gap               0.002712
arc_length_m            0.232933
to_out_degree           0.347215
projection_fraction     0.356075
heading_cos             0.906209
distance_m              3.494743
Intercept: -5.8364
```

## 3. Five-Fold Trip-Grouped Cross-Validation Results
- Point-weighted Top-1 Accuracy: **65.90%**
- Candidate Macro F1: **86.39%**
- Point-mean Log Loss: **1.3555**
- Route Continuity: **88.55%**
- Candidate Miss Rate: **6.39%**

## 4. Student-Test Evaluation (10 Trips)
- Point-weighted Top-1 Accuracy: **77.09%** (Trip mean: 78.63%)
- Candidate Macro F1: **88.62%**
- Point-mean Log Loss: **0.7697**
- Route Continuity: **89.49%**
- Candidate Miss Rate: **0.0783%** (15 misses / 19,160 points)

## 5. Comparison Against Nearest-Road Distance Baseline
| Partition | Metric | Distance Baseline | Logistic Regression | Diff |
|---|---|---:|---:|---:|
| **Student-Test** | Top-1 Accuracy | 54.42% | 77.09% | +22.67% |
| **Student-Test** | Candidate Macro F1 | 76.77% | 88.62% | +11.85% |
| **Student-Test** | Point Log Loss | 0.9911 | 0.7697 | -0.2214 |
| **Student-Test** | Route Continuity | 82.03% | 89.49% | +7.46% |
| **Train (CV)** | Top-1 Accuracy | 50.03% | 65.90% | +15.87% |

## 6. Execution Timings
- Total Execution: **149.29 s**
- 5-Fold Cross-Validation: **115.68 s**
- Final Model Fit (3.33M rows): **8.39 s**
- Student-Test Inference & Evaluation: **1.90 s**
