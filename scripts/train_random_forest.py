"""Train, validate, and test the Phase 4 Random Forest classifier."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_candidate_model import run_experiment
from src.model.random_forest import fit_random_forest


if __name__ == "__main__":
    run_experiment(
        model_name="random_forest",
        display_name="Random Forest",
        fit_model=fit_random_forest,
        probability_col="rf_probability",
        description=__doc__,
    )
