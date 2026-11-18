"""Evaluate classifier-only and HMM paths with the same out-of-fold scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_candidate_model import load_folds, resolve_path, write_json
from src.data.geometry import check_trip, project_coordinates
from src.data.loader import read_arcs, read_nodes, read_track
from src.model.hmm import (
    DecodedTrip,
    DirectedRoadNetwork,
    TripLattice,
    build_lattice,
    decode_classifier,
    decode_viterbi,
)


CANDIDATE_KEYS = ["trip_id", "point_idx", "arc_id"]


def prepare_trip(partition: str, trip_id: str, scores: pd.DataFrame, config: dict):
    check_trip(partition, trip_id)
    raw = REPO / "data" / "raw"
    processed = REPO / "data" / "processed" / partition / trip_id / "candidates.parquet"
    candidates = pd.read_parquet(processed, columns=[
        *CANDIDATE_KEYS, "candidate_rank", "projection_fraction", "label",
    ])
    features = candidates.drop(columns="label").merge(
        scores[[*CANDIDATE_KEYS, "score"]],
        on=CANDIDATE_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not features["_merge"].eq("both").all():
        raise ValueError(f"Candidate score coverage mismatch for {partition}/{trip_id}")
    track = read_track(partition, trip_id, root=raw).to_numpy(dtype=float)
    nodes = read_nodes(partition, trip_id, root=raw).to_numpy(dtype=float)
    arcs = read_arcs(partition, trip_id, root=raw).to_numpy(dtype=int)
    xy, node_xy, _ = project_coordinates(track[:, :2], nodes)
    network = DirectedRoadNetwork(arcs, node_xy, max_route_m=config["max_route_m"])
    lattice = build_lattice(
        features, network, xy, track[:, 2],
        max_gap_s=config["max_gap_s"], backward_jitter_m=config["backward_jitter_m"],
    )
    return lattice, arcs, candidates.loc[candidates["label"].eq(1), ["point_idx", "arc_id"]]


def path_metrics(
    decoded: DecodedTrip, lattice: TripLattice, arcs: np.ndarray, reference: pd.DataFrame,
) -> dict:
    chosen = decoded.arc_ids
    if reference["point_idx"].duplicated().any():
        raise ValueError("Multiple reference arcs for a GPS point")
    truth = np.full(len(chosen), -1, dtype=int)
    truth[reference["point_idx"].to_numpy(dtype=int)] = reference["arc_id"].to_numpy(dtype=int)
    correct = (truth >= 0) & (chosen == truth)
    adjacent = (
        (chosen[:-1] >= 0)
        & (chosen[1:] >= 0)
        & (np.asarray(lattice.break_before[1:]) != "gps_gap")
    )
    previous, current = chosen[:-1][adjacent], chosen[1:][adjacent]
    direct = (previous == current) | (arcs[previous, 1] == arcs[current, 0])
    reachable = 0
    for index in np.flatnonzero(adjacent) + 1:
        matrix = lattice.distances[index]
        if matrix is not None:
            reachable += int(np.isfinite(matrix[decoded.choices[index - 1], decoded.choices[index]]))
    return {
        "points": len(chosen),
        "covered_points": len(reference),
        "top1_correct": int(correct.sum()),
        "top1_accuracy": float(correct.mean()),
        "trip_exact_match": bool(correct.all()),
        "direct_links": int(direct.sum()),
        "reachable_links": reachable,
        "evaluated_links": int(adjacent.sum()),
        "gps_gaps": lattice.break_before.count("gps_gap"),
        "unreachable_restarts": decoded.break_before.count("no_transition"),
        "unmatched_points": int((chosen < 0).sum()),
    }


def summarize(trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (partition, decoder, beta), group in trips.groupby(
        ["partition", "decoder", "beta_m"], dropna=False, sort=False,
    ):
        points = int(group["points"].sum())
        links = int(group["evaluated_links"].sum())
        rows.append({
            "partition": partition, "decoder": decoder, "beta_m": beta,
            "trips": len(group), "points": points, "top1_correct": int(group["top1_correct"].sum()),
            "top1_accuracy": group["top1_correct"].sum() / points,
            "trip_mean_top1_accuracy": group["top1_accuracy"].mean(),
            "candidate_miss_rate": 1 - group["covered_points"].sum() / points,
            "exact_trips": int(group["trip_exact_match"].sum()),
            "route_continuity": group["direct_links"].sum() / links if links else np.nan,
            "bounded_reachability": group["reachable_links"].sum() / links if links else np.nan,
            "evaluated_links": links,
            "gps_gaps": int(group["gps_gaps"].sum()),
            "unreachable_restarts": int(group["unreachable_restarts"].sum()),
            "unmatched_points": int(group["unmatched_points"].sum()),
        })
    return pd.DataFrame(rows)


def evaluate_partition(partition: str, scores: pd.DataFrame, config: dict, beta_values: list[float]):
    rows = []
    paths: dict[float, list[pd.DataFrame]] = {beta: [] for beta in beta_values}
    grouped = scores.groupby("trip_id", sort=True)
    for position, (trip_id, trip_scores) in enumerate(grouped, 1):
        lattice, arcs, reference = prepare_trip(partition, trip_id, trip_scores, config)
        base = decode_classifier(lattice)
        if "selected" in trip_scores:
            stored = trip_scores.loc[trip_scores["selected"]].sort_values("point_idx")
            if not np.array_equal(base.arc_ids[base.arc_ids >= 0], stored["arc_id"].to_numpy()):
                raise ValueError(f"Independent decoder disagrees with saved classifier picks for {trip_id}")
        rows.append({
            "partition": partition, "trip_id": trip_id, "decoder": "classifier", "beta_m": np.nan,
            **path_metrics(base, lattice, arcs, reference),
        })
        for beta in beta_values:
            result = decode_viterbi(lattice, beta)
            rows.append({
                "partition": partition, "trip_id": trip_id, "decoder": "hmm", "beta_m": beta,
                **path_metrics(result, lattice, arcs, reference),
            })
            paths[beta].append(pd.DataFrame({
                "trip_id": trip_id,
                "point_idx": np.arange(len(base.arc_ids)),
                "classifier_arc_id": base.arc_ids,
                "hmm_arc_id": result.arc_ids,
                "break_before": result.break_before,
            }))
        if position == 1 or position % 10 == 0 or position == grouped.ngroups:
            print(f"[{partition}] decoded {position}/{grouped.ngroups} trips", flush=True)
    return pd.DataFrame(rows), paths


def read_scores(path: Path) -> pd.DataFrame:
    columns = [*CANDIDATE_KEYS, "score", "selected"]
    if path.name == "cv_predictions.parquet":
        columns.append("fold")
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "hmm.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    beta_values = config["beta_grid_m"]
    if not beta_values or len(set(beta_values)) != len(beta_values):
        raise ValueError("beta_grid_m must be nonempty and contain distinct values")
    score_dir = resolve_path(config["score_directory"])
    out = resolve_path(config["output_directory"])
    out.mkdir(parents=True, exist_ok=True)
    cv_scores = read_scores(score_dir / "cv_predictions.parquet")
    folds = load_folds(resolve_path(config["fold_file"]), set(cv_scores["trip_id"]))
    if not cv_scores["trip_id"].map(folds).eq(cv_scores["fold"]).all():
        raise ValueError("Validation scores do not cover the shared out-of-fold assignment")
    validation, cv_paths = evaluate_partition("train", cv_scores, config, beta_values)
    validation_summary = summarize(validation)
    best = validation_summary.loc[validation_summary["decoder"].eq("hmm")].sort_values(
        ["top1_accuracy", "route_continuity", "beta_m"], ascending=[False, False, True],
    ).iloc[0]
    beta = float(best["beta_m"])
    validation_summary.to_csv(out / "beta_search.csv", index=False)
    write_json(out / "decoder_config.json", config | {"selected_beta_m": beta})
    print(f"[validation] selected beta_m={beta:g}, Top-1={best['top1_accuracy']:.4%}", flush=True)

    test_scores = read_scores(score_dir / "student_test_predictions.parquet")
    expected_test_trips = {
        path.parent.name for path in (REPO / "data" / "processed" / "student_test").glob("*/candidates.parquet")
    }
    if set(test_scores["trip_id"]) != expected_test_trips:
        raise ValueError("Student-test scores do not cover every processed test trip")
    test, test_paths = evaluate_partition("student_test", test_scores, config, [beta])
    comparison = pd.concat([
        validation.loc[validation["beta_m"].isna() | validation["beta_m"].eq(beta)], test,
    ], ignore_index=True)
    comparison.to_csv(out / "decoder_by_trip.csv", index=False)
    summary = summarize(comparison)
    summary.to_csv(out / "decoder_comparison.csv", index=False)
    pd.concat(cv_paths[beta], ignore_index=True).to_parquet(out / "cv_paths.parquet", index=False)
    pd.concat(test_paths[beta], ignore_index=True).to_parquet(out / "student_test_paths.parquet", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
