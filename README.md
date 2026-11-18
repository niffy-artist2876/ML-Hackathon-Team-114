# PathSnap

PathSnap map-matches noisy GPS observations to directed road arcs. The data
stage creates metric features and derived training labels; the model stage
trains five classifiers with trip-grouped cross-validation, records Optuna
tuning trials, and compares classifier-only decoding with a connectivity-aware
Viterbi decoder.

## Run the model, tuning, and HMM evaluation

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

The run writes comparison tables and logs to `results/`, and exports the model
to `results/models/final_extra_trees.joblib`. Do not use the evaluator
partition, and do not treat generated reference labels as independently
verified ground truth.

## Run the API

```powershell
python scripts/run_api.py
```

Open `ui/index.html` in a browser after the API has started. Set the API URL
when the UI and API run on separate laptops. The API returns predicted arc IDs
and aggregate metrics only; reference route data remains server-side.
