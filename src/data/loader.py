import pandas as pd
from pathlib import Path

'''
This module loads the datasets.
'''


DATA_ROOT = Path("data/raw")

def read_track(partition, trip_id, root=DATA_ROOT):
    path = Path(root) / partition / trip_id / f"{trip_id}.track"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat", "t"])

def read_nodes(partition, trip_id, root=DATA_ROOT):
    path = Path(root) / partition / trip_id / f"{trip_id}.nodes"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["lon", "lat"])

def read_arcs(partition, trip_id, root=DATA_ROOT):
    path = Path(root) / partition / trip_id / f"{trip_id}.arcs"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["from_node", "to_node"])

def read_route(partition, trip_id, root=DATA_ROOT):
    path = Path(root) / partition / trip_id / f"{trip_id}.route"
    return pd.read_csv(path, sep=r"\s+", header=None, names=["arc_id"])

def list_trips(partition, root=DATA_ROOT):
    return sorted(p.name for p in (Path(root)/partition).iterdir() if p.is_dir())
