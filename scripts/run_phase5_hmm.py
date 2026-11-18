"""Execute Phase 5: HMM + Viterbi sequence decoding and benchmark against Classifier-only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.loader import read_arcs
from src.model.hmm import PathSnapHMM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 HMM + Viterbi Evaluation")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results" / "phase5_hmm")
    parser.add_argument("--folds-path", type=Path, default=REPO / "results" / "tables" / "train_trip_folds.csv")
    parser.add_argument("--transition-penalty", type=float, default=-4.0)
    parser.add_argument("--emission-weight", type=float, default=1.0)
    parser.add_argument("--large-gap-penalty", type=float, default=0.0)
    return parser.parse_args()


def load_candidate_tables(partition: str) -> dict[str, pd.DataFrame]:
    root = REPO / "data" / "processed" / partition
    if not root.is_dir():
        raise FileNotFoundError(f"Processed partition directory not found: {root}")
    tables = {}
    for trip_dir in sorted(root.iterdir()):
        if trip_dir.is_dir():
            candidate_path = trip_dir / "candidates.parquet"
            if candidate_path.is_file():
                tables[trip_dir.name] = pd.read_parquet(candidate_path)
    return tables


def compute_summary_row(df: pd.DataFrame, partition_name: str, decoder_name: str) -> dict[str, Any]:
    total_points = int(df["point_count"].sum())
    total_covered = int(df["covered_points"].sum())
    total_correct = int(df["top1_correct"].sum())
    total_evaluated_trans = int(df["evaluated_transitions"].sum())
    total_disconnected_trans = int(df["disconnected_transitions"].sum())
    total_connected_trans = total_evaluated_trans - total_disconnected_trans
    total_disagreements = int(df["disagreements"].sum())

    point_weighted_top1 = total_correct / total_points if total_points > 0 else 0.0
    trip_mean_top1 = float(df["top1_accuracy"].mean())
    route_continuity = total_connected_trans / total_evaluated_trans if total_evaluated_trans > 0 else 1.0
    trip_mean_continuity = float(df["route_continuity"].mean())
    macro_f1 = float(df["candidate_macro_f1"].mean())
    miss_rate = (total_points - total_covered) / total_points if total_points > 0 else 0.0
    mean_inference_time = float(df["inference_seconds"].sum())
    disagreement_rate = (total_disagreements / total_points) * 100.0 if total_points > 0 else 0.0

    return {
        "partition": partition_name,
        "decoder": decoder_name,
        "trips": len(df),
        "point_count": total_points,
        "covered_points": total_covered,
        "candidate_miss_rate": miss_rate,
        "point_weighted_top1": point_weighted_top1,
        "trip_mean_top1": trip_mean_top1,
        "route_continuity": route_continuity,
        "trip_mean_route_continuity": trip_mean_continuity,
        "candidate_macro_f1": macro_f1,
        "disconnected_transitions": total_disconnected_trans,
        "evaluated_transitions": total_evaluated_trans,
        "total_disagreements": total_disagreements,
        "disagreement_rate_percent": disagreement_rate,
        "total_inference_seconds": mean_inference_time,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PATHSNAP — PHASE 5: HMM + VITERBI SEQUENCE RECONSTRUCTION BENCHMARK")
    print("=" * 80)

    # 1. Initialize HMM pipeline
    hmm = PathSnapHMM(
        transition_penalty=args.transition_penalty,
        emission_weight=args.emission_weight,
        large_gap_penalty=args.large_gap_penalty,
    )
    print(f"Loaded frozen classifier from: {hmm.classifier_path}")
    print(f"Features ({len(hmm.features)}): {hmm.features}")
    print(f"HMM parameters: transition_penalty={args.transition_penalty}, "
          f"emission_weight={args.emission_weight}, large_gap_penalty={args.large_gap_penalty}")

    # 2. Load fold definitions
    folds_df = pd.read_csv(args.folds_path, dtype={"trip_id": str})
    folds_df["trip_id"] = folds_df["trip_id"].str.zfill(8)
    folds = dict(zip(folds_df["trip_id"], folds_df["fold"].astype(int)))

    # 3. Load train candidate tables
    print("\nLoading training candidate tables...")
    train_tables = load_candidate_tables("train")
    print(f"Loaded {len(train_tables)} training trips.")

    # 4. Five-Fold Cross-Validation Run
    cv_clf_trip_records: list[dict[str, Any]] = []
    cv_hmm_trip_records: list[dict[str, Any]] = []

    print("\nRunning 5-fold cross-validation on training trips...")
    fold_ids = sorted(set(folds.values()))

    for fold in fold_ids:
        val_trips = sorted([t for t, f in folds.items() if f == fold and t in train_tables])
        print(f"  [Fold {fold}] Evaluating {len(val_trips)} trips...", end="", flush=True)
        fold_start = time.perf_counter()

        for trip_id in val_trips:
            table = train_tables[trip_id]
            arcs = read_arcs("train", trip_id, root=REPO / "data" / "raw")
            m_clf, m_hmm = hmm.evaluate_trip(table, arcs, trip_id=trip_id, partition="train", fold=fold)
            cv_clf_trip_records.append(m_clf)
            cv_hmm_trip_records.append(m_hmm)

        print(f" done in {time.perf_counter() - fold_start:.2f}s")

    df_cv_clf = pd.DataFrame(cv_clf_trip_records).sort_values(["fold", "trip_id"])
    df_cv_hmm = pd.DataFrame(cv_hmm_trip_records).sort_values(["fold", "trip_id"])

    # Combined CV per trip
    df_cv_by_trip = pd.concat([df_cv_clf, df_cv_hmm], ignore_index=True).sort_values(["trip_id", "decoder"])
    df_cv_by_trip.to_csv(args.output_dir / "hmm_cv_by_trip.csv", index=False)

    # Fold level aggregations
    fold_rows = []
    for fold in fold_ids:
        f_clf = df_cv_clf[df_cv_clf["fold"] == fold]
        f_hmm = df_cv_hmm[df_cv_hmm["fold"] == fold]
        fold_rows.append(compute_summary_row(f_clf, f"fold_{fold}", "classifier_only"))
        fold_rows.append(compute_summary_row(f_hmm, f"fold_{fold}", "classifier_plus_hmm"))
    df_cv_by_fold = pd.DataFrame(fold_rows)
    df_cv_by_fold.to_csv(args.output_dir / "hmm_cv_by_fold.csv", index=False)

    # Overall CV summary
    cv_summary_rows = [
        compute_summary_row(df_cv_clf, "cross_validation", "classifier_only"),
        compute_summary_row(df_cv_hmm, "cross_validation", "classifier_plus_hmm"),
    ]
    df_cv_summary = pd.DataFrame(cv_summary_rows)
    df_cv_summary.to_csv(args.output_dir / "hmm_cv_summary.csv", index=False)

    # 5. Student-Test Evaluation
    print("\nLoading and evaluating student-test candidate tables (10 trips)...")
    student_tables = load_candidate_tables("student_test")
    test_clf_records = []
    test_hmm_records = []

    for trip_id in sorted(student_tables):
        table = student_tables[trip_id]
        arcs = read_arcs("student_test", trip_id, root=REPO / "data" / "raw")
        m_clf, m_hmm = hmm.evaluate_trip(table, arcs, trip_id=trip_id, partition="student_test", fold=-1)
        test_clf_records.append(m_clf)
        test_hmm_records.append(m_hmm)

    df_test_clf = pd.DataFrame(test_clf_records).sort_values("trip_id")
    df_test_hmm = pd.DataFrame(test_hmm_records).sort_values("trip_id")

    df_test_by_trip = pd.concat([df_test_clf, df_test_hmm], ignore_index=True).sort_values(["trip_id", "decoder"])
    df_test_by_trip.to_csv(args.output_dir / "hmm_student_test_by_trip.csv", index=False)

    test_summary_rows = [
        compute_summary_row(df_test_clf, "student_test", "classifier_only"),
        compute_summary_row(df_test_hmm, "student_test", "classifier_plus_hmm"),
    ]
    df_test_summary = pd.DataFrame(test_summary_rows)
    df_test_summary.to_csv(args.output_dir / "hmm_student_test_summary.csv", index=False)

    # 6. Classifier vs HMM Comparison Table
    comparison_rows = cv_summary_rows + test_summary_rows
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(args.output_dir / "classifier_vs_hmm.csv", index=False)

    # 7. Configuration and Metadata JSON
    config_dict = {
        "model": "HistGradientBoostingClassifier",
        "classifier_artifact": str(hmm.classifier_path.relative_to(REPO)),
        "features": hmm.features,
        "hmm_parameters": {
            "transition_penalty": args.transition_penalty,
            "emission_weight": args.emission_weight,
            "large_gap_penalty": args.large_gap_penalty,
        },
        "normalization": "group_normalized_softmax_log_emissions",
        "connectivity_rule": "arcs[prev, to_node] == arcs[curr, from_node] or prev == curr",
        "large_gap_threshold_seconds": 30.0,
    }
    with open(args.output_dir / "hmm_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    metadata_dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_trips": len(train_tables),
        "validation_folds": len(fold_ids),
        "student_test_trips": len(student_tables),
        "cv_point_weighted_top1_clf": cv_summary_rows[0]["point_weighted_top1"],
        "cv_point_weighted_top1_hmm": cv_summary_rows[1]["point_weighted_top1"],
        "cv_trip_mean_top1_clf": cv_summary_rows[0]["trip_mean_top1"],
        "cv_trip_mean_top1_hmm": cv_summary_rows[1]["trip_mean_top1"],
        "cv_continuity_clf": cv_summary_rows[0]["route_continuity"],
        "cv_continuity_hmm": cv_summary_rows[1]["route_continuity"],
        "test_trip_mean_top1_clf": test_summary_rows[0]["trip_mean_top1"],
        "test_trip_mean_top1_hmm": test_summary_rows[1]["trip_mean_top1"],
        "test_continuity_clf": test_summary_rows[0]["route_continuity"],
        "test_continuity_hmm": test_summary_rows[1]["route_continuity"],
    }
    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON SUMMARY:")
    print("=" * 80)
    cols_to_print = [
        "partition", "decoder", "trips", "point_weighted_top1",
        "trip_mean_top1", "route_continuity", "disconnected_transitions",
        "disagreement_rate_percent", "total_inference_seconds",
    ]
    print(df_comparison[cols_to_print].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
