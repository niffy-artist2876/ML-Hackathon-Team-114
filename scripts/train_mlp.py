"""Train, validate, fit, and test the Phase 4 MLP model.

The model classifies candidate arcs independently for each GPS observation.
Cross-validation is grouped by complete trip using the existing fixed folds.
The student-test partition is never used for fitting or model selection; it is
scored only after the final model is fit on all labelled training trips.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.model.mlp import fit_mlp
from src.model.phase4_training import ModelSpec, run_training

MLP_SPEC = ModelSpec(
    display_name="MLP",
    probability_column="mlp_probability",
    fit=fit_mlp,
    comparison_parameters=(
        "hidden_layer_sizes", "activation", "alpha", "batch_size",
        "learning_rate_init", "max_iter", "early_stopping", "random_state",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "mlp.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(MLP_SPEC, config_path=args.config, output_dir=args.output_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
