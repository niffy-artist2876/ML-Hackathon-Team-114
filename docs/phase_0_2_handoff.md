# Phase 0-2 handoff checklist

## Phase 0 - reproducibility and verification

Completed:

- Created an isolated Python environment.
- Installed the data-stage dependencies.
- Extracted only the allowlisted `train` and `student_test` files.
- Verified all 360 development files.
- Confirmed 80 train trips and 10 student-test trips.
- Ran all four data-stage unit tests successfully.
- Recorded package versions and file hashes in `results/logs/`.

Commands:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-data-lock.txt
.venv\Scripts\python.exe scripts/verify_data.py --extract
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Phase 1 - data and reference diagnostics

Completed:

- Raw file loading with explicit schemas.
- Coordinate range and finite-value checks.
- Timestamp ordering, duplicate timestamp, and gap checks.
- Node and arc endpoint validation.
- Weak/strong graph-component checks and isolated-node checks.
- Route arc-ID and adjacent-endpoint checks.
- Per-trip UTM projection.
- Exact projected point-to-segment candidate search.
- Monotone route-occurrence alignment with Huber spatial loss.

Review gates still open:

- `k=16` misses 6.3934% of training reference candidates.
- 23,268 training observations are farther than 50 m from the aligned
  reference segment.
- Alignment is derived reference data and may be wrong for loops, outliers,
  route gaps, or incomplete reference routes.
- `manual_review_status` remains `pending` in `configs/data_stage.json`.

## Phase 2 - model-ready tables

Completed:

- Candidate settings evaluated at `k=4`, `k=8`, and `k=16`.
- Training-only selection froze `k=16` because it had the lowest miss rate,
  although it did not reach the 1% target.
- Candidate features generated for every development trip.
- Offline labels attached only to training/evaluation artifacts.
- Label-free inference table implemented.
- Nearest-arc baseline predictions saved for all 90 trips.
- Five GroupKFold trip assignments saved for the 80 training trips.
- Projection, alignment, candidate, baseline, and data-quality reports saved.

The local derived Parquet tables are intentionally ignored by Git. Teammates
must run `scripts/verify_data.py --extract` and `scripts/run_data_stage.py` to
recreate them.

## Do not do in Phase 3

- Do not train on student-test labels.
- Do not use `.route` in inference code.
- Do not randomly split individual GPS observations across folds.
- Do not treat the nearest-arc baseline as a learned model.
- Do not report derived-reference scores as independently verified ground truth.

## Phase 3 entry point

Start with a candidate-grouped evaluation harness. For each GPS observation,
score its candidate arcs, select the highest-scoring candidate, and calculate
point-level Top-1, Macro F1, Log Loss, candidate miss rate, and route
continuity. Then train and compare the required five models using the saved
trip folds.
