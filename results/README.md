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
