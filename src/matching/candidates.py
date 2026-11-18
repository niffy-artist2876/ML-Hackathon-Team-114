"""Generate arc candidates and training labels for a trip."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.geo.projection import project_dataframe, utm_crs_for_coordinates


def _segments(nodes: pd.DataFrame, arcs: pd.DataFrame, crs=None) -> tuple[np.ndarray, np.ndarray]:
    projected = project_dataframe(nodes, crs=crs)
    xy = projected[["x_m", "y_m"]].to_numpy(dtype=float)
    endpoints = arcs[["from_node", "to_node"]].to_numpy(dtype=int)
    if endpoints.size and (endpoints.min() < 0 or endpoints.max() >= len(xy)):
        raise ValueError("Arc endpoint is outside the node table")
    return xy[endpoints[:, 0]], xy[endpoints[:, 1]]


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    denominator = np.einsum("ij,ij->i", vector, vector)
    fraction = np.divide(
        np.einsum("ij,ij->i", point - start, vector),
        denominator,
        out=np.zeros(len(start), dtype=float),
        where=denominator > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start + fraction[:, None] * vector
    return np.linalg.norm(closest - point, axis=1)


def reference_arcs(
    track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame, route: pd.DataFrame, crs=None
) -> np.ndarray:
    """Assign each GPS point to the closest arc in the ordered route."""
    if crs is None:
        crs = utm_crs_for_coordinates(
            pd.concat([nodes["lon"], track["lon"]]),
            pd.concat([nodes["lat"], track["lat"]]),
        )
    start, end = _segments(nodes, arcs, crs)
    route_ids = route["arc_id"].to_numpy(dtype=int)
    if len(route_ids) == 0:
        raise ValueError("Route is empty")
    if route_ids.min() < 0 or route_ids.max() >= len(arcs):
        raise ValueError("Route contains an invalid arc ID")
    points = project_dataframe(track, crs=crs)[["x_m", "y_m"]].to_numpy(dtype=float)
    result = np.empty(len(points), dtype=int)
    for index, point in enumerate(points):
        distances = _point_segment_distance(point, start[route_ids], end[route_ids])
        result[index] = route_ids[int(np.argmin(distances))]
    return result


def generate_candidates(
    track: pd.DataFrame,
    nodes: pd.DataFrame,
    arcs: pd.DataFrame,
    radius_m: float = 75.0,
    max_candidates: int = 20,
) -> pd.DataFrame:
    """Return nearby arc IDs for every GPS point.

    A node KD-tree narrows the search, then exact point-to-segment distances
    rank the incident arcs. The fallback query keeps every point non-empty.
    """
    if radius_m <= 0 or max_candidates <= 0:
        raise ValueError("radius_m and max_candidates must be positive")
    crs = utm_crs_for_coordinates(
        pd.concat([nodes["lon"], track["lon"]]),
        pd.concat([nodes["lat"], track["lat"]]),
    )
    start, end = _segments(nodes, arcs, crs)
    node_xy = project_dataframe(nodes, crs=crs)[["x_m", "y_m"]].to_numpy(dtype=float)
    tree = cKDTree(node_xy)
    incident: dict[int, set[int]] = defaultdict(set)
    for arc_id, (from_node, to_node) in enumerate(arcs[["from_node", "to_node"]].to_numpy(dtype=int)):
        incident[int(from_node)].add(arc_id)
        incident[int(to_node)].add(arc_id)

    points = project_dataframe(track, crs=crs)[["x_m", "y_m"]].to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for point_idx, point in enumerate(points):
        node_ids = tree.query_ball_point(point, radius_m)
        arc_ids = {arc_id for node_id in node_ids for arc_id in incident[int(node_id)]}
        if not arc_ids:
            nearest_nodes = tree.query(point, k=min(8, len(node_xy)))[1]
            arc_ids = {arc_id for node_id in np.atleast_1d(nearest_nodes) for arc_id in incident[int(node_id)]}
        candidate_ids = np.fromiter(arc_ids, dtype=int)
        distances = _point_segment_distance(point, start[candidate_ids], end[candidate_ids])
        order = np.argsort(distances)[:max_candidates]
        for rank, position in enumerate(order):
            rows.append(
                {
                    "point_idx": point_idx,
                    "arc_id": int(candidate_ids[position]),
                    "distance_m": float(distances[position]),
                    "candidate_rank": rank,
                }
            )
    return pd.DataFrame(rows, columns=["point_idx", "arc_id", "distance_m", "candidate_rank"])
