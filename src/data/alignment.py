import numpy as np
import pandas as pd
from src.data.loader import read_track, read_nodes, read_arcs, read_route
from src.data.projection import make_projector

def load_trip_projected(partition, trip_id):
    track = read_track(partition, trip_id)
    nodes = read_nodes(partition, trip_id)
    arcs = read_arcs(partition, trip_id)
    route = read_route(partition, trip_id)["arc_id"].values

    project, crs = make_projector(track.lon, track.lat)
    px, py = project(track.lon.values, track.lat.values)
    nx, ny = project(nodes.lon.values, nodes.lat.values)

    pts = np.column_stack([px, py])                     
    node_xy = np.column_stack([nx, ny])                 

    ra = arcs.iloc[route]
    a = node_xy[ra.from_node.values]                    
    b = node_xy[ra.to_node.values]
    return pts, a, b, route

def point_segment_dist(pts, a, b):
    """Distance from each point to each segment. Returns (n_points, n_route)."""
    ab = b - a                                          
    ab2 = (ab ** 2).sum(axis=1)                         
    ab2 = np.where(ab2 == 0, 1e-12, ab2)                
    ap = pts[:, None, :] - a[None, :, :]                
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / ab2[None, :], 0.0, 1.0)   
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]                  
    return np.linalg.norm(pts[:, None, :] - closest, axis=2)

def align_nearest(D):
    """A Naive method where each point gets the closest route arc by position in the route list."""
    return D.argmin(axis=1)  

def align_monotonic(D, max_jump=30, jump_penalty=0.5):
    """
    Choose a route position for every point so that positions never go backwards, 
    positions advance by at most max_jump between consecutive points and 
    total cost = sum of distances + jump_penalty * (positions skipped).
    """
    N, R = D.shape
    cost = np.full((N, R), np.inf)
    back = np.zeros((N, R), dtype=np.int32)
    cost[0] = D[0]                                   # any starting position allowed

    for i in range(1, N):
        for step in range(0, max_jump + 1):
            prev = cost[i - 1, : R - step] + jump_penalty * step     # came from j - step
            cur = cost[i, step:]
            better = prev + D[i, step:] < cur
            cost[i, step:] = np.where(better, prev + D[i, step:], cur)
            back[i, step:] = np.where(better, np.arange(R - step), back[i, step:])

    pos = np.zeros(N, dtype=np.int32)
    pos[-1] = cost[-1].argmin()
    for i in range(N - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos

def point_segment_dist_chunked(pts, a, b, chunk=1000):
    return np.vstack([point_segment_dist(pts[i:i + chunk], a, b)
                      for i in range(0, len(pts), chunk)])

def align_trip(partition, trip_id, bad_dist_m=30.0):
    """Reference arc for every GPS point of a trip, using the monotonic alignment."""
    pts, a, b, route = load_trip_projected(partition, trip_id)
    D = point_segment_dist_chunked(pts, a, b)
    pos = align_monotonic(D)
    dist = D[np.arange(len(pos)), pos]
    return pd.DataFrame({
        "trip_id": trip_id,
        "point_idx": np.arange(len(pos)),
        "route_pos": pos,
        "ref_arc_id": route[pos],
        "dist_to_ref_m": dist,
        "label_reliable": dist <= bad_dist_m,
    })
    

