"""Convert PathSnap longitude/latitude coordinates to metres.

The inventory places the data near Hesse, Germany (about 8.58 E, 49.85 N).
That location lies in UTM zone 32N, so EPSG:25832 (ETRS89 / UTM 32N) is used
for metric operations. ``always_xy=True`` is intentional: the input order is
longitude, latitude and the output order is easting, northing.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


WGS84_CRS = CRS.from_epsg(4326)
PROJECTED_CRS = CRS.from_epsg(25832)


def utm_crs_for_coordinates(lon: Iterable[float], lat: Iterable[float]) -> CRS:
    """Choose the local UTM CRS for a coordinate collection.

    The original inspected trip is in UTM 32N, but the full dataset contains
    trips worldwide. A per-trip UTM zone keeps distances in metres without
    producing invalid northings for southern-hemisphere trips.
    """
    lon_array = np.asarray(list(lon), dtype=float)
    lat_array = np.asarray(list(lat), dtype=float)
    valid = np.isfinite(lon_array) & np.isfinite(lat_array)
    if not valid.any():
        raise ValueError("At least one finite coordinate is required")
    median_lon = float(np.median(lon_array[valid]))
    median_lat = float(np.median(lat_array[valid]))
    zone = int(np.floor((median_lon + 180.0) / 6.0)) + 1
    zone = min(60, max(1, zone))
    epsg = (32600 if median_lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def project_lonlat(
    lon: Iterable[float], lat: Iterable[float], crs: CRS | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Project longitude/latitude arrays to easting/northing in metres."""
    lon_array = np.asarray(list(lon), dtype=float)
    lat_array = np.asarray(list(lat), dtype=float)
    if lon_array.shape != lat_array.shape:
        raise ValueError("lon and lat must have the same shape")
    if not np.isfinite(lon_array).all() or not np.isfinite(lat_array).all():
        raise ValueError("lon and lat must contain only finite values")
    target_crs = crs or utm_crs_for_coordinates(lon_array, lat_array)
    transformer = Transformer.from_crs(WGS84_CRS, target_crs, always_xy=True)
    x, y = transformer.transform(lon_array, lat_array)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def project_dataframe(
    frame: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    x_col: str = "x_m",
    y_col: str = "y_m",
    crs: CRS | None = None,
) -> pd.DataFrame:
    """Return a copy with projected metre columns, preserving input columns."""
    missing = {lon_col, lat_col} - set(frame.columns)
    if missing:
        raise KeyError(f"Missing coordinate columns: {sorted(missing)}")
    result = frame.copy()
    target_crs = crs or utm_crs_for_coordinates(result[lon_col], result[lat_col])
    result[x_col], result[y_col] = project_lonlat(result[lon_col], result[lat_col], target_crs)
    return result


def distance_m(
    lon_1: float, lat_1: float, lon_2: float, lat_2: float
) -> float:
    """Return the projected straight-line distance between two coordinates."""
    x, y = project_lonlat([lon_1, lon_2], [lat_1, lat_2])
    return float(np.hypot(x[1] - x[0], y[1] - y[0]))


def projection_sanity_check(
    frame: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat"
) -> dict[str, float | str]:
    """Summarise a metre-scale check using the first and last valid points."""
    target_crs = utm_crs_for_coordinates(frame[lon_col], frame[lat_col])
    projected = project_dataframe(frame, lon_col=lon_col, lat_col=lat_col, crs=target_crs)
    if len(projected) < 2:
        raise ValueError("At least two points are required for a distance check")
    distance = float(
        np.hypot(
            projected.iloc[-1]["x_m"] - projected.iloc[0]["x_m"],
            projected.iloc[-1]["y_m"] - projected.iloc[0]["y_m"],
        )
    )
    return {
        "crs": target_crs.to_string(),
        "input_order": "lon,lat",
        "output_order": "easting,northing",
        "first_to_last_distance_m": distance,
    }
