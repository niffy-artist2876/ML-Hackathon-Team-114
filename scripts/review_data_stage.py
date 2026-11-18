"""Read-only label/metric audit; regenerates review visuals, never labels.

Counts are evidence categories, not proven physical causes. Only explicit
train/student_test inputs are read. Original run logs are preserved.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.geometry import load_geometry, segment_distances
from src.data.features import build_inference_table
from src.data.reference import align_reference
from src.data.loader import read_route, read_track
from src.data.visualize import review_map, review_figure

TABLES = REPO / "results/tables"
REVIEW_TRIPS = ("00000001", "00000069", "00000090")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    start = time.time()
    old_log = json.loads((REPO / "results/logs/data_stage.json").read_text())
    hash_failures = []
    for rel, expected in old_log["output_hashes"].items():
        path = REPO / rel
        if not path.is_file() or sha(path) != expected:
            hash_failures.append(rel)
    inventory = pd.read_csv(TABLES / "data_inventory.csv", dtype={"trip_id": str})
    raw_hash_failures = []
    for r in inventory.itertuples():
        assert r.partition in ("train", "student_test")
        p = REPO / "data/raw" / r.partition / r.trip_id / f"{r.trip_id}.{r.file_type}"
        if sha(p) != r.sha256:
            raw_hash_failures.append(str(p.relative_to(REPO)))
    source_status = {}
    for rel, expected in old_log["source_hashes"].items():
        b = (REPO / rel).read_bytes()
        source_status[rel] = "identical" if hashlib.sha256(b).hexdigest() == expected else (
            "line_endings_only" if hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest() == expected else "changed")
    quality = pd.read_csv(TABLES / "data_quality.csv", dtype={"trip_id": str})
    old_candidates = pd.read_csv(TABLES / "candidate_settings_by_trip.csv", dtype={"trip_id": str})
    old_baseline = pd.read_csv(TABLES / "nearest_baseline_by_trip.csv", dtype={"trip_id": str})
    evidence, metrics, point_flags = [], [], []
    for r in quality.itertuples():
        assert r.partition in ("train", "student_test")
        folder = REPO / "data/processed" / r.partition / r.trip_id
        a = pd.read_parquet(folder / "reference_alignment.parquet")
        c = pd.read_parquet(folder / "candidates_k16.parquet")
        chosen = pd.read_parquet(folder / "candidates.parquet")
        pd.testing.assert_frame_equal(chosen, c, check_exact=True)  # frozen k=16
        n = len(a)
        np.testing.assert_array_equal(a.point_idx, np.arange(n))
        assert not c.duplicated(["point_idx", "arc_id"]).any()
        assert c.groupby("point_idx").size().eq(16).all()
        expected_label = (c.arc_id.to_numpy() == a.reference_arc_id.to_numpy()[c.point_idx.to_numpy()]).astype("int8")
        np.testing.assert_array_equal(c.label, expected_label)
        nearest = c.loc[c.candidate_rank == 1].sort_values("point_idx")
        pred = pd.read_csv(REPO / "results/predictions" / f"{r.partition}_{r.trip_id}_nearest.csv.gz", dtype={"trip_id": "string"})
        pd.testing.assert_frame_equal(pred.reset_index(drop=True), nearest[["trip_id", "point_idx", "arc_id"]].reset_index(drop=True))
        ref = a.reference_arc_id.to_numpy()
        correct = int((nearest.arc_id.to_numpy() == ref).sum())
        old = old_baseline.loc[(old_baseline.partition == r.partition) & (old_baseline.trip_id == r.trip_id)].iloc[0]
        assert int(old.correct) == correct and int(old.points) == n
        metrics.append({"partition": r.partition, "trip_id": r.trip_id, "points": n, "correct": correct})
        for k in (4, 8, 16):
            sub = c.loc[c.candidate_rank <= k]
            hits = int(sub.label.sum())
            previous = old_candidates.loc[(old_candidates.partition == r.partition) & (old_candidates.trip_id == r.trip_id) & (old_candidates.k == k)].iloc[0]
            assert (int(previous.hits), int(previous.misses), int(previous.candidate_rows)) == (hits, n-hits, len(sub))
        timestamps = read_track(r.partition, r.trip_id, root=REPO / "data/raw").t.to_numpy()
        dt = np.r_[0., np.diff(timestamps)]
        near_gap = (dt > 10) | np.r_[dt[1:] > 10, False]
        far = a.reference_distance_m.to_numpy() > 50
        route_near = a.nearest_route_distance_m.to_numpy() <= 50
        network_near = nearest.distance_m.to_numpy() <= 50
        # Exclusive geometric evidence buckets, without assigning unproven causes.
        order_conflict = far & route_near
        route_coverage = far & ~route_near & network_near
        network_distance = far & ~route_near & ~network_near
        assert int(order_conflict.sum()+route_coverage.sum()+network_distance.sum()) == int(far.sum())
        row = {"partition": r.partition, "trip_id": r.trip_id, "points": n,
               "far_points": int(far.sum()), "near_other_reference_arc": int(order_conflict.sum()),
               "near_network_but_not_reference": int(route_coverage.sum()),
               "far_from_entire_network": int(network_distance.sum()),
               "far_immediately_after_gap_over10s": int((far & (dt > 10)).sum()),
               "far_adjacent_to_gap_over10s": int((far & near_gap).sum()),
               "far_on_trip_with_route_endpoint_break": int(far.sum()) if r.route_adjacent_endpoint_mismatches else 0,
               "route_endpoint_breaks": int(r.route_adjacent_endpoint_mismatches),
               "max_gap_s": float(dt.max())}
        evidence.append(row)
        flags = a[["trip_id", "point_idx"]].copy()
        flags["partition"] = r.partition
        flags["assigned_distance_m"] = a.reference_distance_m
        flags["nearest_reference_distance_m"] = a.nearest_route_distance_m
        flags["nearest_network_distance_m"] = nearest.distance_m.to_numpy()
        flags["gap_before_s"] = dt
        flags["category"] = np.select([order_conflict, route_coverage, network_distance],
                                        ["near_other_reference_arc", "near_network_not_reference", "far_from_network"], default="within50m")
        point_flags.append(flags.loc[far])
    evidence = pd.DataFrame(evidence)
    evidence.to_csv(TABLES / "alignment_review_by_trip.csv", index=False)
    # Point-level derived reference diagnostics remain local, never inference input.
    pd.concat(point_flags, ignore_index=True).to_parquet(REPO / "data/processed/alignment_review_far_points.parquet", index=False)
    totals = evidence.drop(columns="max_gap_s").groupby("partition").sum(numeric_only=True)
    totals["max_gap_s"] = evidence.groupby("partition").max_gap_s.max()
    totals.to_csv(TABLES / "alignment_review_summary.csv")
    baseline = pd.DataFrame(metrics).groupby("partition")[["points", "correct"]].sum()
    baseline["point_weighted_top1"] = baseline.correct / baseline.points
    saved = pd.read_csv(TABLES / "nearest_baseline_summary.csv").set_index("partition")
    np.testing.assert_allclose(baseline.point_weighted_top1, saved.loc[baseline.index, "point_weighted_top1"], rtol=0, atol=1e-14)
    for partition in ("train", "student_test"):
        for k in (4, 8, 16):
            rows = old_candidates.loc[(old_candidates.partition == partition) & (old_candidates.k == k)]
            summary = pd.read_csv(TABLES / "candidate_settings_summary.csv")
            s = summary.loc[(summary.partition == partition) & (summary.k == k)].iloc[0]
            assert int(rows.misses.sum()) == int(s.misses)
            assert np.isclose(rows.misses.sum()/rows.points.sum(), s.candidate_miss_rate)
    folds = pd.read_csv(TABLES / "train_trip_folds.csv", dtype={"trip_id": str})
    assert len(folds) == 80 and folds.trip_id.nunique() == 80 and set(folds.fold) == set(range(5))
    assert set(folds.trip_id) == set(quality.loc[quality.partition == "train", "trip_id"])
    regenerated = []
    for trip in REVIEW_TRIPS:
        folder = REPO / "data/processed/train" / trip
        a = pd.read_parquet(folder / "reference_alignment.parquet")
        g = load_geometry("train", trip)
        fresh_a, route = align_reference(g)
        pd.testing.assert_frame_equal(a, fresh_a, check_exact=True)
        # Real trip replay with any route/label file read made a hard failure.
        original_open = __import__("builtins").open
        original_path_open = Path.open
        def guard_open(file, *args, **kwargs):
            if isinstance(file, (str, Path)) and (str(file).endswith(".route") or "processed" in str(file)):
                raise AssertionError(f"Forbidden inference input: {file}")
            return original_open(file, *args, **kwargs)
        def guard_path_open(path, *args, **kwargs):
            if path.suffix == ".route" or "processed" in path.parts:
                raise AssertionError(f"Forbidden inference input: {path}")
            return original_path_open(path, *args, **kwargs)
        with patch("builtins.open", guard_open), patch.object(Path, "open", guard_path_open):
            fresh_c = build_inference_table("train", trip, k=16)
        saved_c = pd.read_parquet(folder / "candidates.parquet").drop(columns="label")
        pd.testing.assert_frame_equal(fresh_c, saved_c, check_exact=True)
        pred = fresh_c.loc[fresh_c.candidate_rank == 1, "arc_id"].to_numpy()
        html = REPO / "results/maps" / f"train_{trip}.html"
        review_map(g, route, a, pred, html)
        assert "tile.openstreetmap.org" not in html.read_text(encoding="utf-8")
        review_figure(g, route, a, REPO / "results/figures" / f"train_{trip}_alignment.png")
        regenerated.append(trip)
        print("Replayed labels/features and regenerated review maps:", trip, flush=True)
    # Investigate the two largest affected trips and an order-conflict example.
    investigations = []
    for trip in ("00000054", "00000033", "00000066"):
        g = load_geometry("train", trip)
        folder = REPO / "data/processed/train" / trip
        a = pd.read_parquet(folder / "reference_alignment.parquet")
        c = pd.read_parquet(folder / "candidates.parquet")
        near = c.loc[c.candidate_rank == 1].reset_index(drop=True)
        route = read_route("train", trip, root=REPO / "data/raw").arc_id.to_numpy(dtype=int)
        # Independent full-network distance scan: no spatial index involved.
        ids = np.unique(np.r_[np.linspace(0, len(a)-1, 40, dtype=int),
                              np.flatnonzero(a.reference_far)[::max(1, int(a.reference_far.sum())//40)]])
        for idx in ids:
            d, _ = segment_distances(g.xy[idx], g.index.starts, g.index.ends)
            np.testing.assert_allclose(d.min(), near.distance_m.iloc[idx], rtol=0, atol=1e-8)
            np.testing.assert_allclose(d[route].min(), a.nearest_route_distance_m.iloc[idx], rtol=0, atol=1e-8)
        off = near.distance_m.to_numpy() > 50
        changes = np.flatnonzero(np.diff(np.r_[False, off, False].astype(int)))
        runs = list(zip(changes[::2], changes[1::2]))
        longest = max(runs, key=lambda pair: pair[1]-pair[0]) if runs else (0, 0)
        lo, hi = longest
        investigations.append({"trip_id": trip, "brute_force_points_checked": len(ids),
                               "network_distance_median_m": float(near.distance_m.median()),
                               "network_distance_p95_m": float(near.distance_m.quantile(.95)),
                               "longest_off_network_run_points": int(hi-lo),
                               "longest_off_network_run_duration_s": float(g.timestamps[hi-1]-g.timestamps[lo]) if hi>lo else 0,
                               "longest_off_network_run_extent_m": float(np.linalg.norm(np.ptp(g.xy[lo:hi], axis=0))) if hi>lo else 0})
        review_figure(g, route, a, REPO / "results/figures" / f"train_{trip}_investigation.png")
    pd.DataFrame(investigations).to_csv(TABLES / "alignment_case_investigations.csv", index=False)
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines())
    outputs = list((REPO / "results").rglob("*"))
    for partition in ("train", "student_test"):
        outputs.extend((REPO / "data/processed" / partition).rglob("*.parquet"))
    relative_paths = [p.relative_to(REPO).as_posix() for p in outputs if p.is_file()]
    ignored_result = subprocess.run(["git", "check-ignore", "-z", "--stdin"], cwd=REPO,
                                    input="\0".join(relative_paths)+"\0", text=True, capture_output=True)
    if ignored_result.returncode not in (0, 1):
        raise RuntimeError(ignored_result.stderr)
    ignored_paths = set(filter(None, ignored_result.stdout.split("\0")))
    manifest = []
    for p in outputs:
        if not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        ignored = rel in ignored_paths
        manifest.append({"path": rel, "bytes": p.stat().st_size,
                         "status": "tracked" if rel in tracked else "ignored_local" if ignored else "untracked_local"})
    pd.DataFrame(manifest).to_csv(TABLES / "data_stage_output_manifest.csv", index=False)
    log = {"elapsed_seconds": time.time()-start, "command": ".venv/Scripts/python.exe scripts/review_data_stage.py",
           "original_outputs_hash_checked": len(old_log["output_hashes"]), "original_output_hash_failures": hash_failures,
           "raw_files_hash_checked": len(inventory), "raw_hash_failures": raw_hash_failures,
           "source_status_vs_original_run": source_status, "all_90_trip_labels_predictions_and_metrics_verified": True,
           "exact_raw_replay_trips": regenerated, "inference_route_and_processed_reads_blocked_on_replay": True,
           "independent_full_network_distance_checks": investigations,
           "folds_check": "80 unique training trips in 5 folds; no student-test trips",
           "evidence_totals": totals.to_dict(orient="index"), "baseline_recomputed": baseline.to_dict(orient="index"),
           "review_script_sha256": sha(Path(__file__)), "browser_review": "not performed; inspect regenerated local HTML or static maps",
           "scope": "No labels altered; explicit train/student_test input paths only. Logs do not prove all historical external access."}
    (REPO / "results/logs/data_stage_review.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    if hash_failures or raw_hash_failures:
        raise SystemExit("Investigate original artifact hash differences")


if __name__ == "__main__":
    main()
