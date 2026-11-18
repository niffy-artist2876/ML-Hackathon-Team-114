"""Offline reference alignment only. Never import into inference code.

Each state is an OCCURRENCE in the ordered reference route, so repeated arc IDs
remain distinct. Dynamic programming finds the minimum spatial cost under a
nondecreasing occurrence constraint; skips and stays are allowed. This is label
construction, not an HMM or a learned prediction decoder.
"""
import numpy as np
import pandas as pd

from src.data.geometry import ROOT, check_trip, segment_distances
from src.data.loader import read_route


def monotone_alignment(points, starts, ends, huber_delta=25.0):
    n, m = len(points), len(starts)
    if not n or not m:
        raise ValueError("Cannot align empty observations or reference route")
    # int32 backpointers: 4*N*M bytes, freed before the next trip.
    back = np.empty((n, m), dtype=np.int32)
    previous = np.zeros(m)
    indices = np.arange(m, dtype=np.int32)
    for i, point in enumerate(points):
        distances, _ = segment_distances(point, starts, ends)
        # Huber spatial loss reduces domination by very distant GPS outliers.
        cost = np.where(distances <= huber_delta, 0.5 * distances ** 2,
                        huber_delta * (distances - 0.5 * huber_delta))
        prefix = np.minimum.accumulate(previous)
        is_new_min = np.r_[True, previous[1:] < prefix[:-1]]
        parent = np.maximum.accumulate(np.where(is_new_min, indices, 0))
        back[i] = parent
        previous = prefix + cost
        previous -= previous.min()
    states = np.empty(n, dtype=np.int64)
    states[-1] = np.argmin(previous)
    for i in range(n - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def align_reference(geometry, root=ROOT):
    check_trip(geometry.partition, geometry.trip_id)
    route_values = read_route(geometry.partition, geometry.trip_id, root=root).to_numpy(dtype=float).ravel()
    if not len(route_values) or not np.isfinite(route_values).all() or (route_values != np.floor(route_values)).any():
        raise ValueError("Empty/invalid route")
    route = route_values.astype(np.int64)
    if (route < 0).any() or (route >= len(geometry.arcs)).any():
        raise ValueError("Reference arc outside original arc table")
    starts, ends = geometry.index.starts[route], geometry.index.ends[route]
    states = monotone_alignment(geometry.xy, starts, ends)
    records = []
    for i, (point, state) in enumerate(zip(geometry.xy, states)):
        distance, fraction = segment_distances(point, starts, ends)
        different = route != route[state]
        alternative = float(distance[different].min()) if different.any() else np.nan
        records.append({"trip_id": geometry.trip_id, "point_idx": i,
                        "reference_arc_id": int(route[state]), "route_occurrence": int(state),
                        "reference_distance_m": distance[state], "reference_fraction": fraction[state],
                        "nearest_route_distance_m": distance.min(),
                        "alternative_arc_distance_m": alternative,
                        "spatially_ambiguous": bool(alternative <= distance[state] + 2),
                        "reference_far": bool(distance[state] > 50)})
    return pd.DataFrame(records), route


def attach_labels(candidate_table, alignment):
    reference = alignment.set_index("point_idx")["reference_arc_id"]
    result = candidate_table.copy()
    result["label"] = (result["arc_id"].to_numpy() == result["point_idx"].map(reference).to_numpy()).astype("int8")
    return result
