# PathSnap — data-stage pipeline

This branch adds development-data verification, metric geometry, reference
alignment, directed-road candidates, features, a nearest-road baseline and
shared trip folds. It does not train models or implement an HMM, API or UI.

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
