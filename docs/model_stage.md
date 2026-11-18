# Model, HMM, and evaluation stage

`scripts/run_pipeline.py` is the single reproducible entry point for the
remaining modelling work. It builds candidate features from the raw trip
files, keeps trips intact in five `GroupKFold` folds, trains five model
families, and records fold-level metrics and timings.

The five models are logistic regression, random forest, extra trees, histogram
gradient boosting, and an MLP. Optuna's seeded TPE sampler tunes logistic
regression regularisation on the same trip folds. Trial results are saved in
`results/tables/bayesian_trials.csv`; model metrics are saved in
`results/tables/model_comparison.csv`.

The HMM uses directed arc connectivity from `.arcs`. A transition receives a
zero log penalty when the next arc starts where the previous arc ends (or is
the same arc), and a configurable negative penalty otherwise. Viterbi uses
the classifier probabilities as emissions. Classifier-only and HMM routes are
compared in `results/tables/decoder_comparison.csv`.

The student-test route files are used only for the documented comparison. The
evaluator partition is not read by the runner.
