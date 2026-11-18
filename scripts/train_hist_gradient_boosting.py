"""Train, validate, and test the Phase 4 Histogram Gradient Boosting classifier."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_candidate_model import run_experiment
from src.model.hist_gradient_boosting import fit_hist_gradient_boosting


if __name__ == "__main__":
    run_experiment(
        model_name="hist_gradient_boosting",
        display_name="HistGradientBoosting",
        fit_model=fit_hist_gradient_boosting,
        probability_col="hgb_probability",
        description=__doc__,
    )
