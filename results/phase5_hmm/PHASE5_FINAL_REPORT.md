# PathSnap — Phase 5: HMM + Viterbi Sequence Reconstruction Final Report

**Date:** November 18, 2026  
**Pipeline Engine:** Frozen `HistGradientBoostingClassifier` (`models/final_model/model.joblib`) + Dynamic Programming Viterbi Decoder  
**Benchmark Artifacts:**  
- Fold Summaries: [`results/phase5_hmm/hmm_cv_by_fold.csv`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/hmm_cv_by_fold.csv)  
- Trip Summaries: [`results/phase5_hmm/hmm_cv_by_trip.csv`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/hmm_cv_by_trip.csv)  
- Overall CV Summary: [`results/phase5_hmm/hmm_cv_summary.csv`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/hmm_cv_summary.csv)  
- Student-Test By Trip: [`results/phase5_hmm/hmm_student_test_by_trip.csv`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/hmm_student_test_by_trip.csv)  
- Head-to-Head Comparison: [`results/phase5_hmm/classifier_vs_hmm.csv`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/classifier_vs_hmm.csv)  
- Configuration: [`results/phase5_hmm/hmm_config.json`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/hmm_config.json)  
- Run Metadata: [`results/phase5_hmm/run_metadata.json`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/results/phase5_hmm/run_metadata.json)

---

## 1. Executive Summary & Recommendation

| Metric | Classifier-Only (Independent Argmax) | Classifier + HMM / Viterbi | Net Impact |
|---|:---:|:---:|:---:|
| **5-Fold CV Point Top-1 Accuracy** | 76.32% | **79.48%** | **+3.16%** |
| **5-Fold CV Trip-Mean Top-1 Accuracy** | 80.97% | **85.19%** | **+4.22%** |
| **5-Fold CV Route Continuity** | 88.99% | **99.14%** | **+10.15%** |
| **5-Fold CV Disconnected Transitions** | 22,944 jumps | **1,795 jumps** | **-92.18% reduction** |
| **Student-Test Point Top-1 Accuracy** | 80.66% | **84.62%** | **+3.96%** |
| **Student-Test Trip-Mean Top-1 Accuracy** | 82.76% | **86.74%** | **+3.98%** |
| **Student-Test Route Continuity** | 88.33% | **99.32%** | **+10.99%** |
| **Student-Test Disconnected Transitions** | 2,235 jumps | **131 jumps** | **-94.14% reduction** |
| **Student-Test Candidate Macro F1** | 90.82% | **92.94%** | **+2.12%** |

### Final Recommendation: **RETAIN HMM / VITERBI DECODING AS THE PRODUCTION DEFAULT**
The empirical benchmark proves decisively that HMM sequence decoding **does not sacrifice accuracy** to achieve continuity; on the contrary, **both accuracy and route continuity improve significantly**:
- **Route Continuity increases from 88.33% to 99.32%** on held-out student-test trips.
- **Top-1 Accuracy increases from 82.76% to 86.74%**, because topological transition constraints prune false-positive road arcs that scored high purely by proximity noise (e.g. parallel side streets or overpasses).
- Impossible topological transitions are slashed by **94.14%** (from 2,235 disconnected breaks down to 131).

---

## 2. How the Hidden Markov Model Works in PathSnap

Map matching is fundamentally a sequence labeling problem: given a sequence of noisy GPS observations $O = (o_1, o_2, \dots, o_T)$, we seek the most likely sequence of underlying road segments $S = (s_1, s_2, \dots, s_T)$.

### A. Hidden States
For every observation $t$, the candidate road generation step produces $K_t$ candidate road arcs $\mathcal{C}_t = \{c_{t, 1}, \dots, c_{t, K_t}\}$. Each candidate arc constitutes a candidate hidden state at time step $t$.

### B. Observation Emission Probabilities
The frozen `HistGradientBoostingClassifier` outputs correctness probability $p(c_{t, k}) \in [0, 1]$ based on the 13 geometric, directional, kinematic, and topological features.
To form a valid observation emission distribution across the candidate set $\mathcal{C}_t$:
$$
P(o_t \mid s_t = c_{t, k}) = \frac{\max(p(c_{t, k}), \epsilon)}{\sum_{j=1}^{K_t} \max(p(c_{t, j}), \epsilon)}, \quad \epsilon = 10^{-12}
$$
In log-space:
$$
\log P(o_t \mid s_t = c_{t, k}) = \log \max(p(c_{t, k}), \epsilon) - \log \sum_{j=1}^{K_t} \max(p(c_{t, j}), \epsilon)
$$
This guarantees that candidate probabilities sum strictly to 1.0 per GPS point, preventing observation length bias.

### C. Transition Probabilities & Network Graph
Transitions between consecutive candidate roads $u = s_{t-1}$ and $v = s_t$ are governed by the directed road network graph (`.arcs`):
1. **Self-Transition ($u == v$):** The vehicle remains on the same directed road segment.
   $$\log P(v \mid u) = 0.0$$
2. **Connected Successor ($u.\text{to\_node} == v.\text{from\_node}$):** Direct topological connectivity in the directed road graph.
   $$\log P(v \mid u) = 0.0$$
3. **Disconnected Transition ($u \ne v$ and $u.\text{to\_node} \ne v.\text{from\_node}$):**
   - **Regular intervals ($\Delta t \le 30$s):** Vehicle could not have physically jumped between disconnected roads without traveling unobserved segments. A steep log penalty is assigned:
     $$\log P(v \mid u) = \text{transition\_penalty} = -4.0$$
   - **Large GPS sensor blackouts ($\Delta t > 30$s, `large_gap == True`):** When the GPS device drops signal or pauses for extended durations, strict topological continuity between $t-1$ and $t$ is unrealistic. The penalty is relaxed:
     $$\log P(v \mid u) = \text{large\_gap\_penalty} = 0.0$$

---

## 3. Viterbi Dynamic Programming Algorithm

To find the global sequence $S^* = \arg\max_S \prod_{t=1}^T P(o_t \mid s_t) P(s_t \mid s_{t-1})$, PathSnap runs the Viterbi algorithm strictly in log-space:

1. **Initialization ($t = 0$):**
   $$V_0(k) = \log P(o_0 \mid c_{0, k})$$
2. **Forward Recursion ($t = 1, \dots, T-1$):**
   $$V_t(j) = \max_{1 \le i \le K_{t-1}} \left[ V_{t-1}(i) + \log P(c_{t, j} \mid c_{t-1, i}) \right] + \log P(o_t \mid c_{t, j})$$
   $$B_t(j) = \arg\max_{1 \le i \le K_{t-1}} \left[ V_{t-1}(i) + \log P(c_{t, j} \mid c_{t-1, i}) \right]$$
3. **Termination & Backtracking:**
   $$k_T^* = \arg\max_{1 \le j \le K_T} V_T(j), \quad k_{t-1}^* = B_t(k_t^*)$$

Vectorization in NumPy over $K_{t-1} \times K_t$ matrices yields full trip sequence decoding in **less than 10 milliseconds** per trip.

---

## 4. Empirical Evaluation Results

### A. Five-Fold Cross-Validation Summary (80 Training Trips, 208,417 Points)

| Fold | Trips | Points | Classifier Top-1 | HMM Top-1 | Top-1 Gain | Classifier Continuity | HMM Continuity | Continuity Gain | Disagreement Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fold 0** | 15 | 40,781 | 79.79% | **83.67%** | +3.88% | 88.07% | **99.18%** | +11.11% | 8.87% |
| **Fold 1** | 16 | 45,984 | 82.26% | **86.41%** | +4.15% | 89.17% | **99.13%** | +9.96% | 8.52% |
| **Fold 2** | 16 | 39,266 | 82.35% | **86.68%** | +4.33% | 89.28% | **99.12%** | +9.84% | 8.35% |
| **Fold 3** | 16 | 40,517 | 78.48% | **82.90%** | +4.42% | 88.89% | **99.07%** | +10.18% | 8.58% |
| **Fold 4** | 17 | 41,869 | 81.82% | **86.13%** | +4.31% | 89.44% | **99.20%** | +9.76% | 7.97% |
| **Overall Mean** | **80** | **208,417** | **80.97%** | **85.19%** | **+4.22%** | **88.99%** | **99.14%** | **+10.15%** | **8.45%** |

---

### B. Held-Out Student-Test Evaluation (10 Trips, 19,160 Points)

| Trip ID | Points | Classifier Top-1 | HMM Top-1 | Accuracy Gain | Classifier Continuity | HMM Continuity | Continuity Gain | Disconnected (Clf $\to$ HMM) | Disagreement Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `00000000` | 2,503 | 91.93% | **93.93%** | +2.00% | 94.36% | **99.80%** | +5.44% | 141 $\to$ 5 | 4.43% |
| `00000007` | 1,354 | 91.14% | **92.39%** | +1.25% | 95.71% | **100.0%** | +4.29% | 58 $\to$ 0 | 2.88% |
| `00000009` | 1,049 | 82.17% | **85.89%** | +3.72% | 84.35% | **98.47%** | +14.12% | 164 $\to$ 16 | 12.87% |
| `00000040` | 2,995 | 84.74% | **88.21%** | +3.47% | 94.09% | **99.90%** | +5.81% | 177 $\to$ 3 | 6.98% |
| `00000052` | 262 | 93.13% | **96.56%** | +3.43% | 92.34% | **97.70%** | +5.36% | 20 $\to$ 6 | 5.73% |
| `00000056` | 3,958 | 83.30% | **89.92%** | +6.62% | 81.96% | **98.43%** | +16.47% | 714 $\to$ 62 | 14.86% |
| `00000065` | 768 | 79.04% | **82.42%** | +3.38% | 87.09% | **99.35%** | +12.26% | 99 $\to$ 5 | 9.90% |
| `00000076` | 1,486 | 76.85% | **84.19%** | +7.34% | 82.29% | **98.52%** | +16.23% | 263 $\to$ 22 | 14.94% |
| `00000077` | 3,911 | 64.54% | **66.79%** | +2.25% | 88.44% | **99.92%** | +11.48% | 452 $\to$ 3 | 8.87% |
| `00000091` | 874 | 80.78% | **87.07%** | +6.29% | 83.16% | **98.97%** | +15.81% | 147 $\to$ 9 | 14.07% |
| **Test Mean** | **19,160** | **82.76%** | **86.74%** | **+3.98%** | **88.33%** | **99.11%** | **+10.99%** | **2,235 $\to$ 131** | **9.73%** |

---

## 5. UI Integration & Live Map Rendering

The PathSnap UI has been updated to provide full transparency:
1. **Model Selector:** Live dropdown switches between `HistGradientBoosting (Winner)`, `Random Forest`, and `Logistic Regression`. The entire map, route coordinates, and telemetry re-predict dynamically with respect to the selected model.
2. **Three Distinct Road Layers:**
   - **Model Predicted Route** (`#350702` Black Bean / Rust bold polyline)
   - **Ground Truth (.route) Road Segments** (`#10B981` Emerald Green dashed line)
   - **Candidate Road Segments** (`#F3B034` Ochre dashed segments with interactive tooltips)
   - **GPS Trajectory Points** (`#F1721E` Glowing observation pins)
3. **Sequence Decoder Toggle:** Seamlessly switch between HMM/Viterbi (connected route) and Classifier-Only (point argmax).

---

## 6. Test Suite & System Integrity

- **New Test Suite:** [`tests/test_hmm.py`](file:///c:/Users/swara/OneDrive/Desktop/ml_hackathon/ML-Hackathon-Team-114/tests/test_hmm.py) added with 11 comprehensive tests:
  - Observation emission probability normalization ($\sum p = 1.0$)
  - Numerical stability under zero probabilities
  - Synthetic 3-point trellis verification (HMM correcting noisy outlier)
  - Connected vs disconnected road transitions
  - Directed network forward/reverse orientation
  - Large GPS time gap penalty relaxation
  - Single-observation trips
  - Empty candidate sets
  - Reproducibility
  - Compatibility with frozen pipeline `models/final_model/model.joblib`
- **Total Test Suite Status:** **All 37 unit tests passed in 5.57 seconds** (`python -m unittest discover -s tests -v`).
- **Data Isolation:** Zero contact with evaluator partition; no data leakage.

---

## 7. Final Conclusion
The PathSnap map matching pipeline with **HistGradientBoosting + HMM Viterbi Decoding** is frozen, thoroughly validated across 80 training trips and 10 held-out trips, verified with 37 unit tests, and fully ready for final evaluator scoring.
