# Random Forest model artifact

This directory contains the final Phase 4 candidate-level Random Forest fit.

## Model

- `random_forest_compressed9.joblib`: compressed trained scikit-learn Pipeline.
- The artifact was fitted on all labelled training trips after the fixed
  five-fold validation run.
- The pipeline includes the median imputer and the Random Forest classifier.
- The model uses the 13 feature columns listed in
  `configs/random_forest.json`.

The model file is stored with Git LFS because it is larger than GitHub's
ordinary file-size limit. After cloning, run `git lfs pull` if the pointer is
present instead of the binary artifact.

Load it with:

```python
import joblib

model = joblib.load("models/random_forest/random_forest_compressed9.joblib")
```

Regenerate the artifact and all evaluation outputs with:

```powershell
.venv_compat\Scripts\python.exe scripts\train_random_forest.py
```
