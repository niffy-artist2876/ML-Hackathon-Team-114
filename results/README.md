# Data-stage result artifacts

These committed files are evidence from the reproducible Phase 0-2 run.

- `tables/data_inventory.csv`: file-level inventory for all development trips.
- `tables/data_quality.csv`: finite values, coordinate ranges, time gaps,
  graph checks, and route checks.
- `tables/projection_checks.csv`: projected-versus-geodesic arc checks.
- `tables/alignment_diagnostics.csv`: derived reference alignment distances,
  ambiguity, and occurrence diagnostics.
- `tables/candidate_settings_by_trip.csv`: candidate counts and misses for
  k=4, 8, and 16 per trip.
- `tables/candidate_settings_summary.csv`: partition-level candidate summary.
- `tables/nearest_baseline_by_trip.csv`: nearest-arc baseline per trip.
- `tables/nearest_baseline_summary.csv`: partition-level baseline metrics.
- `tables/train_trip_folds.csv`: five GroupKFold assignments by trip ID.
- `predictions/*_nearest.csv.gz`: nearest-arc prediction per observation,
  containing `trip_id`, `point_idx`, and `arc_id` only.
- `logs/data_verification.json`: verification command, versions, counts, and
  source hashes.
- `logs/data_stage.json`: data-stage configuration, metrics, versions, and
  output hashes.

The local `data/raw/` and `data/processed/` trees are ignored and must be
regenerated. Reference-containing Folium maps are not part of the Phase 0-2
handoff.

## Phase 3 evaluation outputs

- `evaluation/distance_baseline_by_trip.csv`: per-trip distance-only scores.
- `evaluation/distance_baseline_summary.csv`: partition-level aggregates.
- `evaluation/run_config.json`: score method and metric configuration.

## Phase 4 Random Forest outputs

- `random_forest/cv_by_fold.csv`: fold sizes and training/inference timings.
- `random_forest/cv_by_trip.csv`: held-out metrics for each validation trip.
- `random_forest/cv_summary.csv`: pooled and trip-mean cross-validation metrics.
- `random_forest/train_fit_summary.csv`: in-sample final-fit metrics used to
  inspect overfitting against cross-validation.
- `random_forest/student_test_by_trip.csv`: held-out metrics per student-test
  trip.
- `random_forest/student_test_summary.csv`: final student-test metrics.
- `random_forest/comparison_row.csv`: model-comparison table row.
- `random_forest/feature_importance.csv`: final model feature importances.
- `random_forest/run_metadata.json`: configuration, fold IDs, row counts, and
  output inventory.

The three `*_predictions.parquet` files contain candidate-level probabilities
and selected arcs. They are generated artifacts and can be regenerated with
`scripts/train_random_forest.py`.

## Phase 4 MLP outputs

- `mlp/cv_by_fold.csv`: fold sizes and training/inference timings.
- `mlp/cv_by_trip.csv`: held-out metrics for each validation trip.
- `mlp/cv_summary.csv`: pooled and trip-mean cross-validation metrics.
- `mlp/train_fit_summary.csv`: in-sample final-fit metrics used to
  inspect overfitting against cross-validation.
- `mlp/student_test_by_trip.csv`: held-out metrics per student-test
  trip.
- `mlp/student_test_summary.csv`: final student-test metrics.
- `mlp/comparison_row.csv`: model-comparison table row.
- `mlp/feature_importance.csv`: final model permutation importances.
- `mlp/run_metadata.json`: configuration, fold IDs, row counts, and
  output inventory.

The three `*_predictions.parquet` files contain candidate-level probabilities
and selected arcs. They are generated artifacts and can be regenerated with
`scripts/train_mlp.py`.

## Phase 4 SVM outputs

- `svm/cv_by_fold.csv`: fold sizes and training/inference timings.
- `svm/cv_by_trip.csv`: held-out metrics for each validation trip.
- `svm/cv_summary.csv`: pooled and trip-mean cross-validation metrics.
- `svm/train_fit_summary.csv`: in-sample final-fit metrics used to
  inspect overfitting against cross-validation.
- `svm/student_test_by_trip.csv`: held-out metrics per student-test
  trip.
- `svm/student_test_summary.csv`: final student-test metrics.
- `svm/comparison_row.csv`: model-comparison table row.
- `svm/feature_importance.csv`: final model permutation importances.
- `svm/run_metadata.json`: configuration, fold IDs, row counts, and
  output inventory.

The three `*_predictions.parquet` files contain candidate-level probabilities
and selected arcs. They are generated artifacts and can be regenerated with
`scripts/train_svm.py`.
