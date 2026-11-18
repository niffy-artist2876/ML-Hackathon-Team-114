# SVM model artifact

This directory contains the final Phase 4 candidate-level SVM fit.

## Model

- `svm_compressed9.joblib`: compressed trained scikit-learn Pipeline.
- The artifact was fitted on all labelled training trips after the fixed
  five-fold validation run.
- The pipeline includes the median imputer, the standard scaler, a Nystroem
  RBF kernel approximation, and a sigmoid-calibrated linear SVM.
- The model uses the 13 feature columns listed in `configs/svm.json`.

The model file is stored with Git LFS. After cloning, run `git lfs pull` if
the pointer is present instead of the binary artifact.

Load it with:

```python
import joblib

model = joblib.load("models/svm/svm_compressed9.joblib")
```

Regenerate the artifact and all evaluation outputs with:

```powershell
.venv\Scripts\python.exe scripts\train_svm.py
```
