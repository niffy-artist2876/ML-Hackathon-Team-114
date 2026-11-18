"""Readers for the per-trip PathSnap files."""

from pathlib import Path

import pandas as pd


DATA_ROOT = Path("data/raw")


def _trip_file(partition: str, trip_id: str, suffix: str, root: Path | str) -> Path:
    trip = str(trip_id)
    return Path(root) / partition / trip / f"{trip}.{suffix}"


def read_track(partition: str, trip_id: str, root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Load GPS observations with columns ``lon``, ``lat`` and ``t``."""
    return pd.read_csv(_trip_file(partition, trip_id, "track", root), sep=r"\s+", header=None, names=["lon", "lat", "t"])


def read_nodes(partition: str, trip_id: str, root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Load network nodes with columns ``lon`` and ``lat``."""
    return pd.read_csv(_trip_file(partition, trip_id, "nodes", root), sep=r"\s+", header=None, names=["lon", "lat"])


def read_arcs(partition: str, trip_id: str, root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Load directed arcs with columns ``from_node`` and ``to_node``."""
    return pd.read_csv(_trip_file(partition, trip_id, "arcs", root), sep=r"\s+", header=None, names=["from_node", "to_node"])


def read_route(partition: str, trip_id: str, root: Path | str = DATA_ROOT) -> pd.DataFrame:
    """Load the ordered route arc IDs."""
    return pd.read_csv(_trip_file(partition, trip_id, "route", root), sep=r"\s+", header=None, names=["arc_id"])


def list_trips(partition: str, root: Path | str = DATA_ROOT) -> list[str]:
    """Return trip directory names in deterministic order."""
    partition_root = Path(root) / partition
    if not partition_root.exists():
        raise FileNotFoundError(f"Partition does not exist: {partition_root}")
    return sorted(path.name for path in partition_root.iterdir() if path.is_dir())
