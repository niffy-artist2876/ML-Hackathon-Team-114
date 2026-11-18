"""Build development tables, candidate diagnostics, baseline, maps and folds.

Run verify_data.py --extract first. No learned model is fitted here.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.geometry import load_geometry, projection_check
from src.data.features import features_for_geometry
from src.data.reference import align_reference, attach_labels
from src.data.visualize import review_map, review_figure, inventory_figure

SETTINGS = (4, 8, 16)
TARGET_MISS_RATE = .01
MAP_TRIPS = {("train", "00000001"), ("train", "00000069"), ("train", "00000090")}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.time()
    dirs = {name: REPO / "results" / name for name in ("tables", "logs", "figures", "maps", "predictions")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    processed = REPO / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(dirs["tables"] / "data_inventory.csv", dtype={"trip_id": str})
    quality = pd.read_csv(dirs["tables"] / "data_quality.csv", dtype={"trip_id": str})
    if len(quality) != 90 or inventory["error"].notna().any():
        raise ValueError("Run successful verification on all 90 development trips first")
    inventory_figure(inventory, quality, dirs["figures"] / "inventory.png")
    candidate_rows, baseline_rows, projection_rows, alignment_rows = [], [], [], []
    selected_k = None
    for partition in ("train", "student_test"):
        trip_ids = sorted(quality.loc[quality.partition == partition, "trip_id"])
        for position, trip_id in enumerate(trip_ids, 1):
            trip_started = time.perf_counter()
            g = load_geometry(partition, trip_id)
            projection_rows.append({"partition": partition, "trip_id": trip_id, **projection_check(g)})
            feature_started = time.perf_counter()
            candidates = features_for_geometry(g, max(SETTINGS))
            feature_seconds = time.perf_counter() - feature_started
            alignment, route = align_reference(g)
            labelled = attach_labels(candidates, alignment)
            n = len(g.xy)
            for k in SETTINGS:
                subset = labelled.loc[labelled.candidate_rank <= k]
                hits = int(subset.label.sum())
                candidate_rows.append({"partition": partition, "trip_id": trip_id, "k": k,
                                       "points": n, "candidate_rows": len(subset), "hits": hits,
                                       "misses": n - hits, "candidate_miss_rate": (n - hits) / n,
                                       "candidates_per_point": len(subset) / n})
            nearest = labelled.loc[labelled.candidate_rank == 1]
            correct = int(nearest.label.sum())
            baseline_rows.append({"partition": partition, "trip_id": trip_id, "points": n,
                                  "correct": correct, "top1_accuracy": correct / n,
                                  "feature_seconds_k16": feature_seconds,
                                  "total_seconds": time.perf_counter() - trip_started})
            alignment_rows.append({"partition": partition, "trip_id": trip_id, "points": n,
                                   "route_occurrences": len(route),
                                   "ambiguous_points": int(alignment.spatially_ambiguous.sum()),
                                   "far_points_over_50m": int(alignment.reference_far.sum()),
                                   "distance_median_m": float(alignment.reference_distance_m.median()),
                                   "distance_p95_m": float(alignment.reference_distance_m.quantile(.95)),
                                   "distance_max_m": float(alignment.reference_distance_m.max()),
                                   "occurrence_backward_steps": int((alignment.route_occurrence.diff() < 0).sum())})
            dest = processed / partition / trip_id
            dest.mkdir(parents=True, exist_ok=True)
            labelled.to_parquet(dest / "candidates_k16.parquet", index=False)
            alignment.to_parquet(dest / "reference_alignment.parquet", index=False)
            predictions = nearest[["trip_id", "point_idx", "arc_id"]]
            predictions.to_csv(dirs["predictions"] / f"{partition}_{trip_id}_nearest.csv.gz", index=False,
                               compression={"method": "gzip", "mtime": 0})
            if (partition, trip_id) in MAP_TRIPS:
                review_map(g, route, alignment, predictions.arc_id.to_numpy(), dirs["maps"] / f"{partition}_{trip_id}.html")
                review_figure(g, route, alignment, dirs["figures"] / f"{partition}_{trip_id}_alignment.png")
            print(f"{partition} {position}/{len(trip_ids)} {trip_id}: Top1={correct/n:.4f}; {time.perf_counter()-trip_started:.1f}s", flush=True)
        if partition == "train":
            totals = pd.DataFrame(candidate_rows).groupby("k")[["points", "misses"]].sum()
            totals["miss_rate"] = totals.misses / totals.points
            eligible = totals.index[totals.miss_rate <= TARGET_MISS_RATE].tolist()
            selected_k = int(min(eligible) if eligible else totals.sort_values(["miss_rate", "k"]).index[0])
            print(f"Frozen k={selected_k} using training trips only", flush=True)

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(dirs["tables"] / "candidate_settings_by_trip.csv", index=False)
    candidate_totals = candidate_df.groupby(["partition", "k"], as_index=False)[["points", "candidate_rows", "hits", "misses"]].sum()
    candidate_totals["candidate_miss_rate"] = candidate_totals.misses / candidate_totals.points
    candidate_totals["candidates_per_point"] = candidate_totals.candidate_rows / candidate_totals.points
    candidate_totals.to_csv(dirs["tables"] / "candidate_settings_summary.csv", index=False)
    baseline = pd.DataFrame(baseline_rows)
    baseline.to_csv(dirs["tables"] / "nearest_baseline_by_trip.csv", index=False)
    summary = baseline.groupby("partition", as_index=False).agg(points=("points", "sum"), correct=("correct", "sum"),
                        trip_mean_top1=("top1_accuracy", "mean"), feature_seconds_mean=("feature_seconds_k16", "mean"))
    summary["point_weighted_top1"] = summary.correct / summary.points
    summary.to_csv(dirs["tables"] / "nearest_baseline_summary.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(dirs["tables"] / "projection_checks.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(dirs["tables"] / "alignment_diagnostics.csv", index=False)
    for row in quality.itertuples():
        dest = processed / row.partition / row.trip_id
        frame = pd.read_parquet(dest / "candidates_k16.parquet")
        frame.loc[frame.candidate_rank <= selected_k].to_parquet(dest / "candidates.parquet", index=False)

    train_counts = inventory.loc[(inventory.partition == "train") & (inventory.file_type == "track")].sort_values("trip_id")
    groups = np.repeat(train_counts.trip_id.to_numpy(), train_counts.rows.to_numpy(dtype=int))
    folds = {}
    for fold, (_, validation) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups)):
        for trip in np.unique(groups[validation]):
            folds[trip] = fold
    pd.DataFrame({"trip_id": sorted(folds), "fold": [folds[t] for t in sorted(folds)]}).to_csv(dirs["tables"] / "train_trip_folds.csv", index=False)
    config = {"schema_version": 1, "candidate_method": "exact projected point-to-segment nearest k",
              "candidate_settings": list(SETTINGS), "selected_k": selected_k, "selection_partition": "train",
              "target_train_miss_rate": TARGET_MISS_RATE,
              "selection_rule": "smallest k at <=1% point-weighted misses; else minimum misses, smaller k tie-break",
              "segment_index_spacing_m": 25, "alignment": "monotone route-occurrence DP with Huber spatial loss",
              "alignment_huber_delta_m": 25, "heading_min_displacement_m": 3, "motion_max_gap_s": 10,
              "fold_file": "results/tables/train_trip_folds.csv", "fold_sha256": sha256(dirs["tables"] / "train_trip_folds.csv"),
              "seed": 0, "manual_review_status": "pending", "datum_assumption": "WGS84 longitude, latitude"}
    (REPO / "configs").mkdir(exist_ok=True)
    (REPO / "configs" / "data_stage.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    outputs = list(dirs["tables"].glob("*.csv")) + list(dirs["predictions"].glob("*.gz")) + list(processed.rglob("*.parquet"))
    sources = list((REPO / "src" / "data").glob("*.py")) + list((REPO / "scripts").glob("*.py"))
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "pyproj", "folium", "pyarrow", "matplotlib")}
    log = {"command": ".venv/Scripts/python.exe scripts/run_data_stage.py", "started_unix": started,
           "elapsed_seconds": time.time()-started, "config": config, "versions": versions,
           "baseline": summary.to_dict(orient="records"),
           "source_hashes": {str(p.relative_to(REPO)): sha256(p) for p in sources},
           "output_hashes": {str(p.relative_to(REPO)): sha256(p) for p in outputs}}
    (dirs["logs"] / "data_stage.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    (REPO / "requirements-data-lock.txt").write_text("\n".join(f"{p}=={v}" for p,v in versions.items())+"\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(candidate_totals.to_string(index=False))


if __name__ == "__main__":
    main()
