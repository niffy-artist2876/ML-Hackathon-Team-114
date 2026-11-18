# MLP model artifact

This directory contains the final Phase 4 candidate-level MLP fit.

## Model

- `mlp_compressed9.joblib`: compressed trained scikit-learn Pipeline.
- The artifact was fitted on all labelled training trips after the fixed
  five-fold validation run.
- The pipeline includes the median imputer, the standard scaler, and the
  MLP classifier.
- The model uses the 13 feature columns listed in `configs/mlp.json`.

The model file is stored with Git LFS. After cloning, run `git lfs pull` if
the pointer is present instead of the binary artifact.

Load it with:

```python
import joblib

model = joblib.load("models/mlp/mlp_compressed9.joblib")
```

Regenerate the artifact and all evaluation outputs with:

```powershell
.venv\Scripts\python.exe scripts\train_mlp.py
```
