# PathSnap — data pipeline and Random Forest

PathSnap matches GPS observations to directed road arcs. This branch includes
data verification, candidate generation, a distance-only baseline, and the
Phase 4 Random Forest. See the [Random Forest results](#random-forest-results)
for the validation and held-out scores.

## Model comparison

| Model | Validation Top-1 | Student-test Top-1 | Validation log loss | Student-test log loss | Student-test route continuity | Details |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Distance baseline (Phase 3) | 50.03% | 54.42% | 1.391 | 0.991 | 82.03% | [Section](#phase-3-evaluation-harness) |
| Random Forest | **70.58%** | **82.06%** | 1.266 | 0.598 | **90.18%** | [Section](#phase-4-random-forest) |
| HistGradientBoosting | 70.30% | 80.65% | 0.965 | **0.559** | 88.34% | [Section](#phase-4-histgradientboosting) |
| MLP | 70.22% | 81.44% | **0.921** | 0.585 | 89.67% | [Section](#phase-4-mlp-and-svm) |
| SVM | 69.19% | 77.09% | 1.194 | 0.864 | 89.38% | [Section](#phase-4-mlp-and-svm) |
| Random Forest, Bayesian-tuned | 70.82% | 82.03% | 0.926 | 0.651 | 90.34% | [Section](#phase-5-random-forest-tuning) |
| Random Forest, Bayesian-tuned + HMM | 67.70% | 84.78% | — | — | 96.48% | [Section](#phase-6-hmmviterbi-decoding) |
| ExtraTrees, Bayesian-tuned, classifier only¹ | 74.10% ± 4.44 | 77.88% | — | — | 86.26% | [Section](#full-pipeline-hmm-decoding-api-and-ui) |
| ExtraTrees, Bayesian-tuned + HMM decoding¹ | — | 78.71% | — | — | 97.83% | [Section](#full-pipeline-hmm-decoding-api-and-ui) |

Validation uses the five fixed trip-level folds; student-test trips were scored
once by each final model. Bold marks the best of the four Phase 4 models that
share the Phase 3 candidates and evaluator. Lower log loss is better.

¹ ExtraTrees comes from the separate full-pipeline work. Its hyperparameters were
chosen by Bayesian optimization (8 Optuna trials maximizing mean validation
Top-1), and the validation figure is the best trial scored on the same folds, so it
is optimistic. It is the mean ± standard deviation across folds rather than
point-weighted. ExtraTrees also uses its own candidate set
(2,837,039 training candidate rows versus 3,334,672) and measures log loss
per candidate row, so its numbers are not directly comparable and its log loss is
omitted. See `results/tables/extra_trees_bayesian_validation_summary.csv` and
`results/logs/extra_trees_bayesian_best.json`.

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

## Phase 4 HistGradientBoosting

This candidate classifier uses the same 13 features, five fixed trip folds,
fold-fitted median imputation, and training-only negative sampling as the
Random Forest run. The baseline fits 100 boosting rounds with binary log loss,
a learning rate of 0.1, and at most 31 leaves per tree. Early stopping is
disabled so validation stays at the trip level. See the
[training configuration](configs/hist_gradient_boosting.json).

After generating the processed data, run from the repository root.

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/train_hist_gradient_boosting.py
```

To check one fold without fitting a final model or evaluating student-test
trips, specify a separate output directory.

```powershell
.venv\Scripts\python.exe scripts/train_hist_gradient_boosting.py --fold 0 --output-dir ../tmp/pathsnap-hgb-smoke-fold-0
```

The complete run writes the fitted pipeline to
`models/hist_gradient_boosting/model.joblib` and the metrics and scored candidate
tables to `results/hist_gradient_boosting/`. Each point gets one selected arc.
The candidate predictions include `hgb_probability` and `score`, with scores
stored as log probabilities for the shared evaluator.
The prediction tables are stored with Git LFS. Run `git lfs pull` to download
them if the checkout contains pointer files.

### HistGradientBoosting results

The fixed configuration was validated on five whole-trip folds and refitted
on all 80 training trips before scoring the 10 student-test trips.

| Metric | Train fit | 5-fold validation | Student test |
| --- | ---: | ---: | ---: |
| Point-weighted Top-1 | 76.32% | 70.30% | 80.65% |
| Mean trip candidate Macro F1 | 90.12% | 89.27% | 90.80% |
| Point log loss | 0.521 | 0.965 | 0.559 |
| Route continuity | 89.07% | 89.48% | 88.34% |
| Candidate miss rate | 6.39% | 6.39% | 0.08% |

The Random Forest scored higher on Top-1 for validation (70.58%) and
student test (82.06%). HistGradientBoosting had lower point log loss on both
splits and its five fold fits took 22.60 s in this run. Candidate misses are
shared by both models because they use the same candidates.

- [Team comparison row](results/hist_gradient_boosting/comparison_row.csv)
- [Validation by fold](results/hist_gradient_boosting/cv_by_fold.csv)
- [Validation summary](results/hist_gradient_boosting/cv_summary.csv)
- [Student-test summary](results/hist_gradient_boosting/student_test_summary.csv)

## Phase 4 MLP and SVM

The MLP and SVM score each candidate arc using the same 13 shared features,
trip-level folds, training-only negative sampling, and Phase 3 evaluator as the
Random Forest and HistGradientBoosting. Both pipelines fit median imputation and
standard scaling on each training split only.

- **MLP:** two hidden layers (64 and 32 ReLU units), Adam with batch size 512, L2 penalty 0.0001, and early stopping on a 10% split of the training rows.
- **SVM:** a Nystroem approximation of the RBF kernel (100 components, gamma 0.1) followed by a linear SVM (C=1.0, balanced class weights). An exact kernel SVM is impractical on about a million candidate rows. Sigmoid (Platt) calibration with 3 internal folds converts SVM margins into probabilities for the log-loss metric.

The final fits retained all positive rows and up to 4 negatives per point, using 1,028,760 of 3,334,672 training candidate rows. Evaluation used the full candidate sets.

### MLP and SVM results

**The MLP reached 70.22% Top-1 in five-fold trip-level validation and 81.44% on the held-out student test. The SVM reached 69.19% and 77.09%.**

| MLP metric | Train fit | 5-fold validation | Student test |
| --- | ---: | ---: | ---: |
| Correct top-1 picks | 157,776 | 146,348 | 15,603 |
| **Point-weighted Top-1** | **75.70%** | **70.22%** | **81.44%** |
| Mean trip Top-1 | 80.75% | 79.50% | 83.12% |
| Mean trip candidate Macro F1 | 90.00% | 89.24% | 91.01% |
| Point log loss | 0.564 | 0.921 | 0.585 |
| Route continuity | 89.57% | 89.74% | 89.67% |
| Mean trip route continuity | 88.11% | 87.90% | 89.35% |
| Mean selected candidate rank | 1.44 | 1.39 | 1.39 |

The MLP train-fit Top-1 exceeds validation by 5.48 percentage points.

| SVM metric | Train fit | 5-fold validation | Student test |
| --- | ---: | ---: | ---: |
| Correct top-1 picks | 149,739 | 144,205 | 14,771 |
| **Point-weighted Top-1** | **71.85%** | **69.19%** | **77.09%** |
| Mean trip Top-1 | 78.37% | 78.24% | 79.72% |
| Mean trip candidate Macro F1 | 88.70% | 88.57% | 89.20% |
| Point log loss | 0.848 | 1.194 | 0.864 |
| Route continuity | 87.08% | 88.48% | 89.38% |
| Mean trip route continuity | 86.80% | 86.89% | 89.17% |
| Mean selected candidate rank | 1.42 | 1.38 | 1.36 |

The SVM train-fit Top-1 exceeds validation by 2.66 percentage points.

Candidate misses are identical across models because every model scores the same
candidate sets: 13,325 training points (6.39%) and 15 student-test points (0.08%)
have no correct candidate. All accuracy figures measure agreement with point-level
reference labels derived by route alignment.

### Comparison across Phase 4 models

| Metric | Distance baseline | Random Forest | HistGradientBoosting | MLP | SVM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation Top-1 | 50.03% | **70.58%** | 70.30% | 70.22% | 69.19% |
| Student-test Top-1 | 54.42% | **82.06%** | 80.65% | 81.44% | 77.09% |
| Validation mean trip Top-1 | 57.57% | **80.13%** | 79.55% | 79.50% | 78.24% |
| Validation Macro F1 | 77.53% | **89.57%** | 89.27% | 89.24% | 88.57% |
| Validation log loss | 1.391 | 1.266 | 0.965 | **0.921** | 1.194 |
| Student-test log loss | 0.991 | 0.598 | **0.559** | 0.585 | 0.864 |
| Validation route continuity | 82.82% | **90.17%** | 89.48% | 89.74% | 88.48% |

Bold marks the best learned model. The distance-baseline columns use the Phase 3
training and student-test summaries.

The Random Forest has the highest validation and student-test Top-1 and was the best model on 48 of 80 validation trips. It beat the MLP on 61 of 80 validation trips, so its small pooled lead is consistent. The MLP had the lowest validation log loss and a much smaller train-validation gap than the Random Forest, so its probabilities may be the better input to the connectivity-aware decoder. The SVM was the weakest learned model on Top-1 in both splits.

### Validation by fold

| Fold | Validation trips | GPS points | Distance | Random Forest | HistGradientBoosting | MLP | SVM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15 | 41,747 | 34.61% | **43.83%** | 43.37% | 43.49% | 42.85% |
| 1 | 16 | 41,739 | 57.58% | 75.40% | 75.10% | **75.53%** | 72.96% |
| 2 | 16 | 41,737 | 51.34% | **76.45%** | 75.84% | 75.85% | 75.23% |
| 3 | 16 | 41,417 | 53.54% | 77.65% | **78.03%** | 77.65% | 76.93% |
| 4 | 17 | 41,777 | 53.08% | **79.63%** | 79.24% | 78.63% | 78.04% |

### Runtime and feature importance

| Stage | MLP | SVM |
| --- | ---: | ---: |
| Fit all five validation models | 671.27 s | 623.28 s |
| Score all validation trips | 3.76 s | 10.99 s |
| Fit final model | 113.58 s | 161.41 s |
| Score all 10 student-test trips | 0.39 s | 1.00 s |

scikit-learn's MLP trains on a single CPU core, unlike the Random Forest, which
uses every core. The SVM fit time includes the kernel approximation and the three
calibration fits.

- **MLP** top permutation importances (increase in candidate-row log loss): `distance_gap_m` (1.467), `distance_m` (0.258), `candidate_rank` (0.164), `heading_cos` (0.161).
- **SVM** top permutation importances (increase in candidate-row log loss): `distance_gap_m` (0.422), `candidate_rank` (0.226), `distance_m` (0.150), `heading_cos` (0.111).

MLP and SVM importances come from permuting each feature on a fixed 50,000-row
sample of the final training rows. They are not directly comparable with the
Random Forest's impurity-based importances.

### Reproduce

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/train_mlp.py
.venv\Scripts\python.exe scripts/train_svm.py
```

- [MLP configuration](configs/mlp.json) and [SVM configuration](configs/svm.json).
- MLP: [validation summary](results/mlp/cv_summary.csv), [student-test summary](results/mlp/student_test_summary.csv), [comparison row](results/mlp/comparison_row.csv), and [saved model](models/mlp/README.md).
- SVM: [validation summary](results/svm/cv_summary.csv), [student-test summary](results/svm/student_test_summary.csv), [comparison row](results/svm/comparison_row.csv), and [saved model](models/svm/README.md).

## Full pipeline, HMM decoding, API and UI

PathSnap map-matches noisy GPS observations to directed road arcs. The data
stage creates metric features and derived training labels; the model stage
trains five classifiers with trip-grouped cross-validation, records Optuna
tuning trials, and compares classifier-only decoding with a connectivity-aware
Viterbi decoder.

### Run the model, tuning, and HMM evaluation

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

The run writes comparison tables and logs to `results/`, and exports the model
to `results/models/final_extra_trees.joblib`. Do not use the evaluator
partition, and do not treat generated reference labels as independently
verified ground truth.

### Run the API

```powershell
python scripts/run_api.py
```

Open `ui/index.html` in a browser after the API has started. Set the API URL
when the UI and API run on separate laptops. The API returns predicted arc IDs
and aggregate metrics only; reference route data remains server-side.

## Phase 5 Random Forest tuning

The Optuna TPE study searches tree depth, minimum leaf size, and the number of
features considered at each split. It starts by repeating the original Random
Forest configuration on the same five trip folds. Each complete trial's
objective is total correct arc picks divided by total validation GPS points.
The sampler, preprocessing, and 100-tree budget stay fixed. Weak trials may
be pruned after at least three folds. The search reads the training partition.

Run the following from the repository root after building the processed tables.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-tuning.txt
.venv\Scripts\python.exe scripts/tune_random_forest.py --trials 10
.venv\Scripts\python.exe scripts/train_random_forest.py --config results/random_forest_tuning/tuned_config.json
```

The tuning script checkpoints trials in a local journal and exports a trial CSV
and the selected configuration. The refit command runs all five folds again
to save out-of-fold scores, fits on all 80 training trips, and scores the ten
student-test trips. It writes to `results/random_forest_tuning/evaluation/` and
saves the model at `models/random_forest_tuned/model.joblib`. Git LFS stores
the published model and candidate score tables. Run `git lfs pull` to download
them if the checkout contains pointer files.

This pass evaluated ten distinct settings. Eight finished all five folds and
two were pruned. One interrupted duplicate and one skipped duplicate are
also recorded in the trial log. The best setting used `max_depth=16`,
`min_samples_leaf=16`, and `max_features=0.5`. It raised validation Top-1
from 70.58% to 70.82% with the same 100 trees.
See the [trial log](results/random_forest_tuning/trials.csv) and
[selected configuration](results/random_forest_tuning/tuned_config.json).

| Random Forest | Validation Top-1 | Validation point log loss | Student-test Top-1 |
| --- | ---: | ---: | ---: |
| Original | 70.58% | 1.266 | 82.06% |
| Tuned | 70.82% | 0.926 | 82.03% |

The tuned forest gained 491 correct validation picks and made six fewer
correct student-test picks than the original forest.

## Phase 6 HMM/Viterbi decoding

The hidden states are candidate arc IDs at each GPS point. The tuned Random
Forest's log candidate scores serve as emissions. A transition compares
projected GPS displacement with the shortest directed road distance between
the two candidate projections, including any intermediate arcs. The Viterbi
decoder penalizes the absolute distance difference scaled by `beta_m`.
It starts a new segment at long GPS gaps and when all incoming paths are
unreachable. Points with no candidates retain their original index.

```powershell
.venv\Scripts\python.exe scripts/evaluate_hmm.py
```

The evaluator selects `beta_m` on the out-of-fold training scores and applies
that value to student test. It reports point-weighted Top-1, exact trip
matches, direct arc continuity, bounded directed reachability, and restart
counts in `results/hmm/`. The classifier-only and HMM predictions use
identical candidate rows and reference labels. The decoder itself takes
candidate scores and geometry, and the evaluator uses labels to grade its path.

### HMM results

The out-of-fold search selected `beta_m=80`. The student-test run used that
fixed value.

| Split and decoder | Point-weighted Top-1 | Mean trip Top-1 | Direct route continuity | Bounded directed reachability |
| --- | ---: | ---: | ---: | ---: |
| Validation, tuned classifier | 70.82% | 80.29% | 90.33% | 91.38% |
| Validation, tuned classifier + HMM | 67.70% | 80.45% | 95.80% | 99.96% |
| Student test, tuned classifier | 82.03% | 83.86% | 90.34% | 95.52% |
| Student test, tuned classifier + HMM | 84.78% | 86.76% | 96.48% | 99.97% |

Direct continuity counts consecutive arcs that are the same or share the
correct directed endpoint. Bounded reachability also counts transitions
connected by intermediate arcs within 300 metres. Both rates exclude marked
GPS gaps. The decoder restarted at 87 unreachable validation transitions and
five unreachable student-test transitions, and produced an arc for every
point in these runs.

The HMM gained 528 correct picks on student test and improved eight of its
ten trips. On validation it improved 57 of 80 trips, but losses on a few trips
outweighed those gains in the point-weighted score. Trips `00000021`,
`00000082`, and `00000084` lost 8,631 correct picks combined. The reference arc
was available for all but three of their observations, so candidate misses
cannot explain most of those losses. The HMM produced more connected paths,
with point-level accuracy varying by trip.

- [Decoder comparison](results/hmm/decoder_comparison.csv)
- [Per-trip results](results/hmm/decoder_by_trip.csv)
- [Validation search for beta](results/hmm/beta_search.csv)
- [Selected decoder configuration](results/hmm/decoder_config.json)
