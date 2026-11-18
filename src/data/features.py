"""Inference candidate tables: this module never reads reference routes."""
import numpy as np
import pandas as pd

from src.data.geometry import ROOT, load_geometry

FEATURE_COLUMNS = ["distance_m", "distance_gap_m", "candidate_rank", "arc_length_m",
                   "projection_fraction", "heading_cos", "heading_valid", "speed_mps",
                   "speed_valid", "time_gap_s", "large_gap", "from_out_degree", "to_out_degree"]


def features_for_geometry(geometry, k=16):
    xy = geometry.xy
    n = len(xy)
    delta = np.zeros_like(xy)
    dt = np.zeros(n)
    delta[1:] = np.diff(xy, axis=0)
    dt[1:] = np.diff(geometry.timestamps)
    displacement = np.linalg.norm(delta, axis=1)
    speed_valid = (dt > 0) & (dt <= 10)
    heading_valid = speed_valid & (displacement >= 3)
    headings = np.divide(delta, displacement[:, None], out=np.zeros_like(delta), where=displacement[:, None] > 0)
    speed = np.divide(displacement, dt, out=np.zeros(n), where=speed_valid)
    starts, ends = geometry.index.starts, geometry.index.ends
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    unit = np.divide(vectors, lengths[:, None], out=np.zeros_like(vectors), where=lengths[:, None] > 0)
    degree = np.bincount(geometry.arcs[:, 0], minlength=len(geometry.node_xy))
    blocks = []
    for point_idx, point in enumerate(xy):
        ids, distance, fraction = geometry.index.nearest(point, k)
        valid_heading = heading_valid[point_idx] & (lengths[ids] > 0)
        cosine = unit[ids] @ headings[point_idx]
        cosine[~valid_heading] = 0
        blocks.append(pd.DataFrame({
            "trip_id": geometry.trip_id, "point_idx": point_idx, "arc_id": ids,
            "distance_m": distance, "distance_gap_m": distance - distance[0],
            "candidate_rank": np.arange(1, len(ids) + 1), "arc_length_m": lengths[ids],
            "projection_fraction": fraction, "heading_cos": np.clip(cosine, -1, 1),
            "heading_valid": valid_heading, "speed_mps": speed[point_idx],
            "speed_valid": speed_valid[point_idx], "time_gap_s": dt[point_idx],
            "large_gap": dt[point_idx] > 10,
            "from_out_degree": degree[geometry.arcs[ids, 0]],
            "to_out_degree": degree[geometry.arcs[ids, 1]],
        }))
    result = pd.concat(blocks, ignore_index=True)
    result["trip_id"] = result["trip_id"].astype("string")
    for col in ("point_idx", "arc_id", "candidate_rank", "from_out_degree", "to_out_degree"):
        result[col] = result[col].astype("int64")
    for col in ("heading_valid", "speed_valid", "large_gap"):
        result[col] = result[col].astype("bool")
    return result


def build_inference_table(partition, trip_id, k=16, root=ROOT):
    """Return identifiers + features, without labels or route-file access.

    Pass the selected k from configs/data_stage.json after running the stage.
    The explicit default is the widest evaluated setting, not a fitted choice.
    """
    return features_for_geometry(load_geometry(partition, trip_id, root=root), k=k)
