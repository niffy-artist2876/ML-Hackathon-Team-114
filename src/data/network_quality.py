import numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.data.loader import DATA_ROOT, list_trips, read_nodes, read_arcs

def network_quality(partition, trip_id, root=DATA_ROOT):
    nodes = read_nodes(partition, trip_id, root)
    arcs = read_arcs(partition, trip_id, root)
    n, m = len(nodes), len(arcs)
    f, t = arcs["from_node"].values, arcs["to_node"].values
    
    valid = (f>=0)&(f<n) & (t>=0)&(t<n)
    f, t = f[valid], t[valid]
    g = coo_matrix((np.ones(len(f)), (f, t)), shape=(n, n)).tocsr()
    n_weak, lab_weak = connected_components(g, directed=True, connection="weak")
    n_strong, lab_strong = connected_components(g, directed=True, connection="strong")

    used = np.zeros(n, dtype=bool)
    used[f]=True
    used[t]=True
    
    pairs = set(zip(f.tolist(), t.tolist()))
    n_rev = sum((b, a) in pairs for a, b in pairs)
    
    return {
        "partition": partition, "trip_id": trip_id,
        "n_nodes": n, "n_arcs": m,
        "n_invalid_arc_ids": int((~valid).sum()),
        "n_isolated_nodes": int((~used).sum()),
        "n_self_loops": int((f == t).sum()),
        "n_duplicate_arcs": int(len(f) - len(pairs)),
        "frac_arcs_with_reverse": n_rev / max(len(pairs), 1),
        "n_weak_components": int(n_weak),
        "largest_weak_frac": float(np.bincount(lab_weak).max() / n),
        "n_strong_components": int(n_strong),
        "largest_strong_frac": float(np.bincount(lab_strong).max() / n),
    }

def build_network_quality(root=DATA_ROOT, partitions=("train", "student_test")):
    return pd.DataFrame([network_quality(p, trip, root)
                         for p in partitions for trip in list_trips(p, root)])
    
if __name__=="__main__":
    q = build_network_quality()
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    q.to_csv("results/tables/network_quality_per_trip.csv", index=False)
    print(q.drop(columns=["partition", "trip_id"]).agg(["min", "median", "max"]).T)
    