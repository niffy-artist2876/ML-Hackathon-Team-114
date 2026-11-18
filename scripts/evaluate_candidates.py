"""Run the Phase 3 candidate-scoring evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.loader import read_arcs
from src.model.metrics import aggregate_trip_metrics, evaluate_candidate_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "evaluation.json")
    parser.add_argument("--partition", choices=["train", "student_test", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "evaluation")
    return parser.parse_args()


def score_distance(table: pd.DataFrame, temperature_m: float) -> pd.DataFrame:
    if temperature_m <= 0:
        raise ValueError("temperature_m must be positive")
    result = table.copy()
    result["score"] = -result["distance_m"].to_numpy(dtype=float) / temperature_m
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    partitions = config["partitions"] if args.partition == "all" else [args.partition]
    raw_root = REPO / "data" / "raw"
    processed_root = REPO / "data" / "processed"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trip_rows = []
    for partition in partitions:
        trip_root = processed_root / partition
        if not trip_root.is_dir():
            raise FileNotFoundError(f"Processed partition not found: {trip_root}. Run the data stage first.")
        for trip_dir in sorted(p for p in trip_root.iterdir() if p.is_dir()):
            candidate_path = trip_dir / "candidates.parquet"
            if not candidate_path.is_file():
                raise FileNotFoundError(f"Candidate table not found: {candidate_path}")
            table = pd.read_parquet(candidate_path)
            if "label" not in table.columns:
                raise ValueError(f"Evaluation requires labels in {candidate_path}")
            table = score_distance(table, float(config["temperature_m"]))
            arcs = read_arcs(partition, trip_dir.name, root=raw_root).to_numpy(dtype=int)
            metrics = evaluate_candidate_table(table, arcs=arcs)
            trip_rows.append({"partition": partition, "trip_id": trip_dir.name, **metrics})

    trip_metrics = pd.DataFrame(trip_rows).sort_values(["partition", "trip_id"])
    summary = aggregate_trip_metrics(trip_metrics)
    trip_metrics.to_csv(args.output_dir / "distance_baseline_by_trip.csv", index=False)
    summary.to_csv(args.output_dir / "distance_baseline_summary.csv", index=False)
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
