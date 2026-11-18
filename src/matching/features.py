"""Build the agreed candidate table for one trip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import read_arcs, read_nodes, read_route, read_track
from src.geo.projection import project_dataframe, utm_crs_for_coordinates
from src.matching.candidates import generate_candidates, reference_arcs


FEATURE_COLUMNS = [
    "distance_m",
    "candidate_rank",
    "arc_length_m",
    "point_speed_mps",
    "heading_cosine",
]


def build_candidate_table(
    partition: str,
    trip_id: str,
    root: str = "data/raw",
    radius_m: float = 75.0,
    max_candidates: int = 20,
    include_label: bool = True,
) -> pd.DataFrame:
    """Build features for a trip, optionally adding the training-only label."""
    track = read_track(partition, trip_id, root=root)
    nodes = read_nodes(partition, trip_id, root=root)
    arcs = read_arcs(partition, trip_id, root=root)
    crs = utm_crs_for_coordinates(
        pd.concat([nodes["lon"], track["lon"]]),
        pd.concat([nodes["lat"], track["lat"]]),
    )
    candidates = generate_candidates(track, nodes, arcs, radius_m, max_candidates)
    node_xy = project_dataframe(nodes, crs=crs)[["x_m", "y_m"]].to_numpy(dtype=float)
    endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    segment = node_xy[endpoints[:, 1]] - node_xy[endpoints[:, 0]]
    arc_length = np.linalg.norm(segment, axis=1)

    projected_track = project_dataframe(track, crs=crs)
    xy = projected_track[["x_m", "y_m"]].to_numpy(dtype=float)
    delta = np.vstack([np.zeros((1, 2)), np.diff(xy, axis=0)])
    displacement = np.linalg.norm(delta, axis=1)
    time = pd.to_numeric(track["t"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(time, prepend=time[0])
    speed = np.divide(displacement, dt, out=np.zeros(len(dt)), where=dt > 0)
    point_heading = np.divide(delta, displacement[:, None], out=np.zeros_like(delta), where=displacement[:, None] > 0)
    arc_heading = np.divide(segment, arc_length[:, None], out=np.zeros_like(segment), where=arc_length[:, None] > 0)

    candidates["trip_id"] = str(trip_id)
    candidates["arc_length_m"] = candidates["arc_id"].map(pd.Series(arc_length)).astype(float)
    candidates["point_speed_mps"] = candidates["point_idx"].map(pd.Series(speed)).astype(float)
    candidates["heading_cosine"] = [
        float(np.dot(point_heading[p], arc_heading[a]))
        for p, a in zip(candidates["point_idx"], candidates["arc_id"])
    ]
    columns = ["trip_id", "point_idx", "arc_id", *FEATURE_COLUMNS]
    if include_label:
        route = read_route(partition, trip_id, root=root)
        truth = reference_arcs(track, nodes, arcs, route, crs=crs)
        candidates["label"] = (candidates["arc_id"].to_numpy() == truth[candidates["point_idx"].to_numpy()]).astype(int)
        columns.append("label")
    return candidates[columns].sort_values(["point_idx", "candidate_rank"]).reset_index(drop=True)
