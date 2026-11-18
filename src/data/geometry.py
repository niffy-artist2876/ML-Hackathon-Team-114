"""Label-free projected geometry and exact segment-distance candidate lookup."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree

from src.data.loader import read_track, read_nodes, read_arcs

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
ALLOWED = ("train", "student_test")


def check_trip(partition, trip_id):
    if partition not in ALLOWED:
        raise ValueError("Development access is restricted to train and student_test")
    if not isinstance(trip_id, str) or len(trip_id) != 8 or not trip_id.isdigit():
        raise ValueError("trip_id must be an eight-digit string")


def project_coordinates(track_lonlat, nodes_lonlat):
    lon, lat = np.median(track_lonlat, axis=0)
    if not (-80 <= lat <= 84):
        raise ValueError("Trip outside supported UTM latitude range")
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    def transform(x):
        result = np.column_stack(transformer.transform(x[:, 0], x[:, 1]))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite projected coordinate")
        return result
    return transform(track_lonlat), transform(nodes_lonlat), epsg


def segment_distances(point, starts, ends):
    vectors = ends - starts
    length_sq = np.einsum("ij,ij->i", vectors, vectors)
    numerator = np.einsum("ij,ij->i", point - starts, vectors)
    fraction = np.divide(numerator, length_sq, out=np.zeros_like(numerator), where=length_sq > 0)
    fraction = np.clip(fraction, 0, 1)
    closest = starts + fraction[:, None] * vectors
    return np.linalg.norm(point - closest, axis=1), fraction


class SegmentIndex:
    """Sample segments at <= spacing metres, then compute exact distances.

    Every segment point is within spacing/2 of a sample. A radius query with
    that margin therefore includes all segments inside the desired radius.
    Sampling is solely an index; distances/features use original endpoints.
    """
    def __init__(self, starts, ends, spacing=25.0):
        if not len(starts) or spacing <= 0:
            raise ValueError("A nonempty network and positive spacing are required")
        self.starts, self.ends, self.spacing = starts, ends, spacing
        lengths = np.linalg.norm(ends - starts, axis=1)
        intervals = np.maximum(1, np.ceil(lengths / spacing).astype(int))
        counts = intervals + 1
        self.sample_arc = np.repeat(np.arange(len(starts)), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        fractions = (np.arange(counts.sum()) - offsets) / np.repeat(intervals, counts)
        samples = starts[self.sample_arc] + fractions[:, None] * (ends - starts)[self.sample_arc]
        self.tree = cKDTree(samples)

    def nearest(self, point, k):
        k = min(k, len(self.starts))
        if k < 1:
            raise ValueError("k must be positive")
        sample_count = min(max(k * 4, 32), len(self.sample_arc))
        while True:
            _, idx = self.tree.query(point, k=sample_count)
            ids = np.unique(self.sample_arc[np.atleast_1d(idx)])
            if len(ids) >= k or sample_count == len(self.sample_arc):
                break
            sample_count = min(sample_count * 2, len(self.sample_arc))
        distances, _ = segment_distances(point, self.starts[ids], self.ends[ids])
        bound = np.partition(distances, k - 1)[k - 1]
        nearby = self.tree.query_ball_point(point, bound + self.spacing / 2 + 1e-7)
        ids = np.unique(self.sample_arc[nearby])
        distances, fraction = segment_distances(point, self.starts[ids], self.ends[ids])
        order = np.lexsort((ids, distances))[:k]
        return ids[order], distances[order], fraction[order]


@dataclass
class TripGeometry:
    partition: str
    trip_id: str
    lonlat: np.ndarray
    timestamps: np.ndarray
    node_lonlat: np.ndarray
    arcs: np.ndarray
    xy: np.ndarray
    node_xy: np.ndarray
    epsg: int
    index: SegmentIndex


def load_geometry(partition, trip_id, root=ROOT):
    check_trip(partition, trip_id)
    track = read_track(partition, trip_id, root=root).to_numpy(dtype=float)
    nodes = read_nodes(partition, trip_id, root=root).to_numpy(dtype=float)
    arc_values = read_arcs(partition, trip_id, root=root).to_numpy(dtype=float)
    if not len(track) or not len(nodes) or not np.isfinite(track).all() or not np.isfinite(nodes).all():
        raise ValueError("Empty/nonfinite input; inspect data-quality log")
    if not np.isfinite(arc_values).all() or (arc_values != np.floor(arc_values)).any():
        raise ValueError("Invalid node IDs")
    arcs = arc_values.astype(np.int64)
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError("Node ID outside original node table")
    xy, node_xy, epsg = project_coordinates(track[:, :2], nodes)
    index = SegmentIndex(node_xy[arcs[:, 0]], node_xy[arcs[:, 1]])
    return TripGeometry(partition, trip_id, track[:, :2], track[:, 2], nodes, arcs, xy, node_xy, epsg, index)


def projection_check(geometry):
    # Compare a deterministic sample of actual arcs with ellipsoidal distances.
    ids = np.linspace(0, len(geometry.arcs) - 1, min(200, len(geometry.arcs)), dtype=int)
    a, b = geometry.arcs[ids].T
    ll = geometry.node_lonlat
    _, _, geodesic = Geod(ellps="WGS84").inv(ll[a, 0], ll[a, 1], ll[b, 0], ll[b, 1])
    projected = np.linalg.norm(geometry.node_xy[b] - geometry.node_xy[a], axis=1)
    valid = geodesic > 1
    relative = np.abs(projected[valid] / geodesic[valid] - 1)
    return {"epsg": geometry.epsg, "sample_arcs": int(valid.sum()),
            "max_relative_distance_error": float(relative.max()) if len(relative) else None,
            "median_relative_distance_error": float(np.median(relative)) if len(relative) else None}
