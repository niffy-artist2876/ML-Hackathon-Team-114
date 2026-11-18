import numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.data.loader import (DATA_ROOT, list_trips, read_track, read_nodes,
                             read_arcs, read_route)

def route_quality(partition, trip_id, root=DATA_ROOT):
    n_pts = len(read_track(partition, trip_id, root))
    n_nodes = len(read_nodes(partition, trip_id, root))
    arcs = read_arcs(partition, trip_id, root)
    r = read_route(partition, trip_id, root)["arc_id"].values
    m = len(arcs)
    
    in_range = (r >= 0) & (r < m)
    r_ok = r[in_range]
    ra = arcs.iloc[r_ok]
    f, t = ra["from_node"].values, ra["to_node"].values
    
    connected = (t[:-1] == f[1:]) if len(r_ok) > 1 else np.array([], dtype=bool)
    
    af, at = arcs["from_node"].values, arcs["to_node"].values
    g = coo_matrix((np.ones(m), (af, at)), shape=(n_nodes, n_nodes)).tocsr()
    _, lab = connected_components(g, directed=True, connection="weak")
    comps = lab[f] if len(f) else np.array([], dtype=int)
    largest = np.bincount(lab).argmax()
    
    key = arcs["from_node"].astype(np.int64) * n_nodes + arcs["to_node"]
    dup_counts = key.map(key.value_counts())
    n_route_arcs_with_dup = int((dup_counts.iloc[r_ok].values > 1).sum())
    
    return {
        "partition": partition, "trip_id": trip_id,
        "n_points": n_pts, "n_route": len(r),
        "route_per_point": len(r) / n_pts,
        "n_out_of_range": int((~in_range).sum()),
        "n_breaks": int((~connected).sum()),
        "frac_connected": float(connected.mean()) if len(connected) else np.nan,
        "n_repeated_arcs": int(len(r) - len(set(r.tolist()))),
        "n_route_components": int(len(set(comps.tolist()))),
        "route_in_largest_comp": bool((comps == largest).all()) if len(comps) else False,
        "n_route_arcs_with_duplicate": n_route_arcs_with_dup,
    }
    
def build_route_quality(root=DATA_ROOT, partitions=("train", "student_test")):
    return pd.DataFrame([route_quality(p, trip, root)
                         for p in partitions for trip in list_trips(p, root)])


if __name__ == "__main__":
    q = build_route_quality()
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    q.to_csv("results/tables/route_quality_per_trip.csv", index=False)
    print(q.drop(columns=["partition", "trip_id"]).agg(["min", "median", "max"]).T)