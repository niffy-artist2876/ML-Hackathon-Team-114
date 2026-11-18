"""Read-only data checks for the two permitted development partitions.

Optional extraction copies only explicitly allowlisted files; existing files
must match the archive and are never overwritten. No modelling is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.loader import read_track, read_nodes, read_arcs, read_route

PARTITIONS = {"train": 80, "student_test": 10}
READERS = {"track": read_track, "nodes": read_nodes,
           "arcs": read_arcs, "route": read_route}
COLUMNS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_allowed(archive, root):
    count = 0
    with zipfile.ZipFile(archive) as source:
        # Only metadata is enumerated. Never open members outside this pattern.
        for info in source.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts) != 4 or parts[0] != "Student folder":
                continue
            _, partition, trip, filename = parts
            if partition not in PARTITIONS or not (len(trip) == 8 and trip.isdigit()):
                continue
            if filename not in {f"{trip}.{ext}" for ext in READERS}:
                continue
            target = root / partition / trip / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as stream:
                if target.exists():
                    if hashlib.file_digest(stream, "sha256").hexdigest() != digest(target):
                        raise ValueError(f"Existing raw file differs: {target}")
                else:
                    with target.open("xb") as output:
                        shutil.copyfileobj(stream, output)
            count += 1
    return count


def invalid_ids(values, size):
    return ~np.isfinite(values) | (values != np.floor(values)) | (values < 0) | (values >= size)


def validate_trip(partition, trip, root):
    rows, tables, errors = [], {}, []
    stats = {"partition": partition, "trip_id": trip}
    for ext, reader in READERS.items():
        path = root / partition / trip / f"{trip}.{ext}"
        record = {"partition": partition, "trip_id": trip, "file_type": ext,
                  "exists": path.is_file(), "bytes": None, "rows": None,
                  "bad_column_rows": None, "sha256": None, "error": ""}
        if not path.is_file():
            record["error"] = "missing file"
        else:
            record.update(bytes=path.stat().st_size, sha256=digest(path))
            with path.open() as stream:
                widths = [len(line.split()) for line in stream if line.strip()]
            record["rows"] = len(widths)
            record["bad_column_rows"] = sum(w != COLUMNS[ext] for w in widths)
            try:
                if record["bad_column_rows"]:
                    raise ValueError("Unexpected column count")
                table = reader(partition, trip, root=root)
                values = table.to_numpy(dtype=float)
                tables[ext] = values
                stats[f"{ext}_missing_cells"] = int(np.isnan(values).sum())
                stats[f"{ext}_nonfinite_cells"] = int((~np.isfinite(values)).sum())
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
        if record["error"]:
            errors.append(f"{ext}: {record['error']}")
        rows.append(record)

    for ext in ("track", "nodes"):
        if ext not in tables:
            continue
        xy = tables[ext][:, :2]
        valid = np.isfinite(xy).all(axis=1) & (np.abs(xy[:, 0]) <= 180) & (np.abs(xy[:, 1]) <= 90)
        stats[f"{ext}_invalid_coordinates"] = int((~valid).sum())
        if valid.any():
            for col, name in enumerate(("lon", "lat")):
                stats[f"{ext}_{name}_min"] = float(xy[valid, col].min())
                stats[f"{ext}_{name}_max"] = float(xy[valid, col].max())
    if "track" in tables:
        dt = np.diff(tables["track"][:, 2])
        finite = dt[np.isfinite(dt)]
        stats.update(time_steps=len(dt), time_backwards=int((dt < 0).sum()),
                     duplicate_timestamps=int((dt == 0).sum()),
                     gaps_over_1s=int((dt > 1).sum()), gaps_over_10s=int((dt > 10).sum()),
                     time_gap_max_s=float(finite.max()) if len(finite) else None,
                     time_gap_median_s=float(np.median(finite)) if len(finite) else None)
    if "nodes" in tables and "arcs" in tables:
        n = len(tables["nodes"])
        arcs = tables["arcs"]
        valid = ~invalid_ids(arcs, n).any(axis=1)
        stats["arcs_invalid_node_reference"] = int((~valid).sum())
        edges = arcs[valid].astype(np.int64)
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        stats["weak_components"] = int(connected_components(graph, directed=True, connection="weak", return_labels=False))
        stats["strong_components"] = int(connected_components(graph, directed=True, connection="strong", return_labels=False))
        stats["isolated_nodes"] = int(n - len(np.unique(edges)))
        stats["self_loop_arcs"] = int((edges[:, 0] == edges[:, 1]).sum())
        if "route" in tables:
            route = tables["route"].ravel()
            bad = invalid_ids(route, len(arcs))
            stats["route_invalid_arc_ids"] = int(bad.sum())
            stats["route_repeated_occurrences"] = int(len(route[~bad]) - len(np.unique(route[~bad])))
            if len(route) and not bad.any():
                route_arcs = arcs[route.astype(np.int64)]
                stats["route_adjacent_endpoint_mismatches"] = int((route_arcs[:-1, 1] != route_arcs[1:, 0]).sum())
    stats["errors"] = " | ".join(errors)
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    started = time.time()
    root = REPO / "data" / "raw"
    tables_dir = REPO / "results" / "tables"
    logs_dir = REPO / "results" / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_allowed(REPO / "Student folder.zip", root) if args.extract else 0
    inventory, quality, counts = [], [], {}
    for partition, expected in PARTITIONS.items():
        folder = root / partition
        trips = sorted(p.name for p in folder.iterdir() if p.is_dir() and len(p.name) == 8 and p.name.isdigit())
        counts[partition] = len(trips)
        if len(trips) != expected:
            raise ValueError(f"{partition}: expected {expected} trips, found {len(trips)}")
        for index, trip in enumerate(trips, 1):
            rows, stats = validate_trip(partition, trip, root)
            inventory.extend(rows)
            quality.append(stats)
            print(f"{partition} {index}/{len(trips)} {trip}", flush=True)
    pd.DataFrame(inventory).to_csv(tables_dir / "data_inventory.csv", index=False)
    pd.DataFrame(quality).to_csv(tables_dir / "data_quality.csv", index=False)
    smoke_trip = sorted((root / "train").iterdir())[0].name
    smoke = {ext: list(reader("train", smoke_trip, root=root).shape) for ext, reader in READERS.items()}
    log = {"command": "python scripts/verify_data.py" + (" --extract" if args.extract else ""),
           "started_unix": started, "elapsed_seconds": time.time() - started,
           "partition_counts": counts, "allowed_archive_files_checked": extracted,
           "smoke_test": {"partition": "train", "trip_id": smoke_trip, "shapes": smoke},
           "file_errors": [row for row in inventory if row["error"]],
           "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
           "script_sha256": digest(Path(__file__)), "loader_sha256": digest(REPO / "src/data/loader.py")}
    (logs_dir / "data_verification.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))
    if log["file_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
