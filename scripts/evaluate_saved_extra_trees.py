"""Evaluate an already-trained ExtraTrees model on student_test trips.

This avoids retraining the five-model comparison when the final artifact is
already available. The student_test route is used only for local evaluation;
it is never passed to model fitting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import list_trips, read_arcs
from src.evaluation import route_continuity
from src.hmm.viterbi import build_transition_graph, decode_viterbi
from src.matching.features import build_candidate_table


def classifier_route(table: pd.DataFrame) -> np.ndarray:
    """Select the highest-probability candidate for each GPS point."""
    return (
        table.sort_values("prob")
        .groupby("point_idx")
        .tail(1)
        .sort_values("point_idx")["arc_id"]
        .to_numpy(dtype=int)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--partition", choices=["train", "student_test"], default="student_test")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("results/models/final_extra_trees.joblib")
    )
    parser.add_argument("--radius-m", type=float, default=75.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    return parser.parse_args()


def attach_predictions(table: pd.DataFrame, model: object, feature_columns: list[str]) -> pd.DataFrame:
    result = table.copy()
    result["prob"] = model.predict_proba(result[feature_columns])[:, 1]
    return result


def truth_from_table(table: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [int(group.loc[group["label"].eq(1), "arc_id"].iloc[0])
         if group["label"].eq(1).any() else -1
         for _, group in table.groupby("point_idx", sort=True)],
        dtype=int,
    )


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"ExtraTrees artifact not found: {args.model_path}")
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])

    rows: list[dict[str, object]] = []
    trips = list_trips(args.partition, args.data_root)
    for trip in trips:
        table = build_candidate_table(
            args.partition, trip, args.data_root, args.radius_m, args.max_candidates
        )
        predicted = attach_predictions(table, model, feature_columns)
        arcs = read_arcs(args.partition, trip, args.data_root)
        classifier = classifier_route(predicted)
        graph = build_transition_graph(arcs)
        hmm = decode_viterbi(predicted, graph)
        truth = truth_from_table(table)
        for decoder, route in (("classifier_only", classifier), ("hmm", hmm)):
            rows.append(
                {
                    "trip_id": trip,
                    "decoder": decoder,
                    "top1_accuracy": float(np.mean(route == truth)),
                    "route_continuity": route_continuity(route, arcs),
                    "n_points": len(route),
                }
            )

    frame = pd.DataFrame(rows)
    output = args.output_dir
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    suffix = "student_test" if args.partition == "student_test" else "train_fit"
    frame.to_csv(output / "tables" / f"extra_trees_{suffix}_by_trip.csv", index=False)
    summary = frame.groupby("decoder", as_index=False)[
        ["top1_accuracy", "route_continuity"]
    ].mean()
    summary["top1_percent"] = summary["top1_accuracy"] * 100.0
    summary["route_continuity_percent"] = summary["route_continuity"] * 100.0
    summary.to_csv(output / "tables" / f"extra_trees_{suffix}_summary.csv", index=False)

    comparison_path = output / "tables" / "model_comparison.csv"
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        validation = comparison[
            comparison["model"].eq("extra_trees") & comparison["fold"].ne("full")
        ].copy()
        if not validation.empty:
            metric_columns = [
                "top1_accuracy",
                "candidate_miss_rate",
                "route_continuity",
                "macro_f1_candidate",
                "log_loss_candidate",
            ]
            validation_summary = pd.DataFrame({
                "metric": metric_columns,
                "mean": [validation[column].mean() for column in metric_columns],
                "std": [validation[column].std(ddof=1) for column in metric_columns],
            })
            validation_summary["mean_percent"] = validation_summary["mean"] * 100.0
            validation_summary["std_percent"] = validation_summary["std"] * 100.0
            validation_summary.to_csv(
                output / "tables" / "extra_trees_validation_summary.csv", index=False
            )
    payload = {
        "model": "extra_trees",
        "partition": args.partition,
        "trips": len(trips),
        "classifier_only_top1_percent": float(
            summary.loc[summary.decoder.eq("classifier_only"), "top1_percent"].iloc[0]
        ),
        "hmm_top1_percent": float(summary.loc[summary.decoder.eq("hmm"), "top1_percent"].iloc[0]),
        "classifier_only_route_continuity_percent": float(
            summary.loc[summary.decoder.eq("classifier_only"), "route_continuity_percent"].iloc[0]
        ),
        "hmm_route_continuity_percent": float(
            summary.loc[summary.decoder.eq("hmm"), "route_continuity_percent"].iloc[0]
        ),
    }
    (output / "logs" / f"extra_trees_{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
