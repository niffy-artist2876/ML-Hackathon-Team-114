# PathSnap — data pipeline and Random Forest

PathSnap matches GPS observations to directed road arcs. This branch includes
data verification, candidate generation, a distance-only baseline, and the
Phase 4 Random Forest. See the [Random Forest results](#random-forest-results)
for the validation and held-out scores.

## Reproduce on Windows (Python 3.11)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-data-lock.txt
.venv\Scripts\python.exe scripts/verify_data.py --extract
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/run_data_stage.py
```

Run from the repository root. Extraction reads only allowlisted train and
student_test files from the existing Student folder.zip and never overwrites
different raw files. The archive was already tracked before this change;
that conflicts with the team's no-raw-data-in-Git rule and needs team review.
This change does not remove it or rewrite history.

## Outputs

- `results/tables/`: inventory, quality checks, projection checks, alignment
  diagnostics, candidate comparison, baseline metrics and train trip folds.
- `results/logs/`: run configuration, versions and input/output hashes.
- `results/predictions/`: baseline arc IDs per observation, without labels.
- `data/processed/<partition>/<trip>/candidates.parquet`: selected labelled
  candidate table; local and ignored by Git.
- `data/processed/<partition>/<trip>/reference_alignment.parquet`: local
  derived references, never an inference input.
- `results/maps/`: three local reference-containing Folium review maps,
  ignored by Git. The script also generates static figures.

The completed run processed 208,417 training observations and 19,160
student-test observations. Nearest-road point-weighted Top-1 was 50.0261%
and 54.4154%, respectively. These measure agreement with derived reference
labels, not independently verified point-level ground truth.

## Review before model training

The selected k=16 setting has 6.3934% training candidate misses, exceeding
the configured 1% target. 23,268 training observations are farther than 50 m
from their aligned reference segment. Visual alignment review is pending;
do not treat this draft as approved training data. See `docs/data_stage.md`.

AI assistance was used to implement and debug this data-stage work. Team
members must review the logic, write their own interpretation and disclose
assistance in the organisers' required format. No detector-evasion claim is made.

## Phase 0-2 handoff for teammates

The first three phases prepared the shared data and experiment inputs:

- Phase 0: reproducible environment setup and raw-data verification.
- Phase 1: data-quality, geometry, route-alignment, and candidate diagnostics.
- Phase 2: model-ready candidate features, labels, baseline predictions, and
  trip-level cross-validation folds.

See `docs/phase_0_2_handoff.md` for the original data-stage handoff checklist.

The fresh run checked 360 files across 80 training trips and 10 student-test
trips with no file errors. It processed 208,417 training observations and
19,160 student-test observations. The data-stage tests pass in the prepared
environment.

The selected `k=16` candidate set still misses 6.3934% of training reference
arcs, and 23,268 training observations are more than 50 m from their aligned
reference segment. These derived labels are not approved final training data
until the team reviews the diagnostics. The local `data/raw/` and
`data/processed/` directories are ignored and must be regenerated with:

```powershell
.venv\Scripts\python.exe scripts/verify_data.py --extract
.venv\Scripts\python.exe scripts/run_data_stage.py
```

Review candidate misses and alignment before treating these scores as final.
Keep student-test labels out of fitting and model selection, and keep reference
route files out of inference.

## Phase 3 evaluation harness

The first Phase 3 implementation is a reusable candidate-grouped evaluator.
It currently evaluates a distance-only softmax baseline and can later accept
scores from each trained model without changing the metric definitions.

Run it after generating local processed tables:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/evaluate_candidates.py
```

Outputs are written to `results/evaluation/`. The evaluator reports candidate
miss rate, point-weighted and trip-mean Top-1 accuracy, candidate-row Macro F1,
point log loss on covered points, route continuity, and mean selected rank.

## Phase 4 Random Forest

The Random Forest scores each candidate arc using the 13 shared features and
picks the highest-scoring arc for each GPS point. A candidate is labelled
positive when it matches the aligned reference arc. The recorded run used
100 trees, median imputation fitted on each training split, and deterministic
negative sampling during fitting. The final fit retained all 195,092 positive
rows and up to four negatives per point, using 1,028,760 of 3,334,672 training
candidate rows. Evaluation used the full candidate sets.

### Random Forest results

**Top-1 accuracy was 70.58% in five-fold trip-level validation and 82.06% on the
held-out student test.** The 85.10% train-fit score measures the final model
on trips used to fit it. Each validation trip was predicted by a model fitted
on the other folds. Student-test trips were scored by the final model after
training.

| Metric | Train fit | 5-fold validation | Student test | What it measures |
| --- | ---: | ---: | ---: | --- |
| Trips | 80 | 80 | 10 | Trips evaluated. The 80 validation trips are counted once each across the five folds. |
| GPS points | 208,417 | 208,417 | 19,160 | Observations evaluated, including candidate misses. |
| Correct top-1 picks | 177,369 | 147,102 | 15,722 | Observations where the selected arc matches the aligned reference. |
| Candidate misses | 13,325 (6.39%) | 13,325 (6.39%) | 15 (0.08%) | Observations where the reference arc is absent from the candidate set. |
| **Point-weighted Top-1** | **85.10%** | **70.58%** | **82.06%** | Correct picks divided by all points. Misses count as errors. |
| Top-1 on covered points | 90.92% | 75.40% | 82.12% | Correct picks divided by points whose reference arc is available. |
| Mean trip Top-1 | 89.28% | 80.13% | 83.73% | Average Top-1 across trips, giving each trip equal weight. |
| Mean trip candidate Macro F1 | 94.60% | 89.57% | 91.34% | Per-trip average of positive- and negative-class F1 over candidate rows, then averaged across trips. The selected candidate is predicted positive. |
| Point log loss | 0.408 | 1.266 | 0.598 | Penalizes low probability on the reference arc after normalizing scores within each covered point. Lower is better. |
| Route continuity | 90.88% | 90.17% | 90.18% | Share of evaluated adjacent picks that stay on the same arc or join end-to-start in the directed graph. The evaluator skips marked gaps. |
| Mean trip route continuity | 90.91% | 88.30% | 90.05% | Route continuity averaged with equal weight for each trip with evaluated transitions. |
| Mean selected candidate rank | 1.55 | 1.39 | 1.40 | Per-trip mean distance rank of the selected arcs, averaged across trips. Rank 1 is nearest. |

The train-fit score exceeds validation by **14.52 percentage points**, which
indicates the model performs better on trips it has seen. The student-test
split has much higher candidate coverage and contains ten trips. Compare
models within the same split. All accuracy figures measure agreement with
point-level reference labels derived by route alignment.

### Comparison with the distance-only baseline

The Phase 3 baseline ranks candidates by distance and requires no fitting.
Both methods use the same candidate sets, reference labels, and evaluator.

| Metric | Validation baseline | Validation Random Forest | Student-test baseline | Student-test Random Forest |
| --- | ---: | ---: | ---: | ---: |
| Point-weighted Top-1 | 50.03% | **70.58%** | 54.42% | **82.06%** |
| Mean trip Top-1 | 57.57% | **80.13%** | 56.42% | **83.73%** |
| Mean trip candidate Macro F1 | 77.53% | **89.57%** | 76.77% | **91.34%** |
| Point log loss | 1.391 | **1.266** | 0.991 | **0.598** |
| Route continuity | 82.82% | **90.17%** | 82.03% | **90.18%** |

Random Forest improved Top-1 by **20.55 percentage points** on the 80
validation trips and **27.64 percentage points** on student test. It can pick
a farther candidate when features such as projection position or heading
support it. The forest selects arcs point by point, so continuity measures
how often those picks connect. A connected sequence can still contain errors.

### Validation by fold

| Fold | Validation trips | GPS points | Candidate miss rate | Distance Top-1 | Random Forest Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 15 | 41,747 | 20.01% | 34.61% | 43.83% |
| 1 | 16 | 41,739 | 0.40% | 57.58% | 75.40% |
| 2 | 16 | 41,737 | &lt;0.01% | 51.34% | 76.45% |
| 3 | 16 | 41,417 | 9.54% | 53.54% | 77.65% |
| 4 | 17 | 41,777 | 2.03% | 53.08% | 79.63% |

Random Forest beat the distance baseline in every fold. Fold 0 had the lowest
accuracy and the most candidate misses. Roughly one in five points in that
fold had no correct candidate for either method to select.

### Runtime and feature importance

| Stage | Recorded time |
| --- | ---: |
| Fit all five validation models | 159.34 s |
| Score all validation trips | 7.58 s |
| Fit final model | 35.61 s |
| Score all 10 student-test trips | 0.69 s |

Fit times include feature conversion, imputation, and forest fitting. Inference
times cover scoring existing candidate rows. Candidate generation, sampling,
metric calculation, and file writing are outside these reported timings.

Distance to arc (23.22%), candidate distance rank (23.06%), and the distance gap
from the nearest candidate (22.40%) together accounted for 68.68% of the final
forest's impurity-based importance. These are related features, so their shares
describe how the forest used them and should be interpreted together.

### Reproduce and compare

After the data stage, run from the repository root with the environment set up
above.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/train_random_forest.py
```

For team comparisons, use the same candidate tables, feature specification,
trip-level folds, and Phase 3 metrics. Record any training-only sampling in
the comparison table.

- [Random Forest configuration](configs/random_forest.json) and [shared trip folds](results/tables/train_trip_folds.csv).
- [Train-fit summary](results/random_forest/train_fit_summary.csv), [validation summary](results/random_forest/cv_summary.csv), and [student-test summary](results/random_forest/student_test_summary.csv).
- [Distance baseline](results/evaluation/distance_baseline_summary.csv) and [team comparison row](results/random_forest/comparison_row.csv).
- [Validation by trip](results/random_forest/cv_by_trip.csv), [student test by trip](results/random_forest/student_test_by_trip.csv), [fold timings](results/random_forest/cv_by_fold.csv), and [full feature importance](results/random_forest/feature_importance.csv).
- [Saved model and loading instructions](models/random_forest/README.md).
