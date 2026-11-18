import numpy as np, pandas as pd
from pathlib import Path
from src.data.loader import DATA_ROOT, list_trips, read_track

def haversine_m(lon1, lat1, lon2, lat2):
    # gives the great-circle distance in metres
    # since lon1/lon2 and lat1/lat2 are angles, euclidean distance is not the correct distance metric 
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def gps_quality(partition, trip_id, root=DATA_ROOT):
    t = read_track(partition, trip_id, root)
    dt = np.diff(t["t"].values)
    step = haversine_m(t["lon"].values[:-1], t["lat"].values[:-1],
                       t["lon"].values[1:], t["lat"].values[1:])
    
    return {
        "partition": partition,
        "trip_id": trip_id,
        "n_points": len(t),
        # missing or invalid values
        "n_nan": int(t.isna().sum().sum()),
        "n_bad_range": int(((t["lon"].abs() > 180) | (t["lat"].abs() > 90)).sum()),
        "n_dup_coords": int(t.duplicated(subset=["lon", "lat"]).sum()),
        # timestamps
        "t_monotonic": bool((dt > 0).all()),
        "n_nonpositive_dt": int((dt <= 0).sum()),
        "frac_dt_not_1": float((dt != 1).mean()),
        "dt_max": float(dt.max()),
        "n_gaps_gt_1s": int((dt > 1).sum()),
        "n_gaps_gt_5s": int((dt > 5).sum()),
        # geography
        "lon_min": t["lon"].min(), "lon_max": t["lon"].max(),
        "lat_min": t["lat"].min(), "lat_max": t["lat"].max(),
        "extent_ew_m": float(haversine_m(t["lon"].min(), t["lat"].mean(), t["lon"].max(), t["lat"].mean())),
        "extent_ns_m": float(haversine_m(t["lon"].mean(), t["lat"].min(), t["lon"].mean(), t["lat"].max())),
        "path_length_m": float(step.sum()),
        "step_max_m": float(step.max()),
        "step_median_m": float(np.median(step)),
    }

def build_gps_quality(root=DATA_ROOT, partitions=("train", "student_test")):
    rows = [gps_quality(p, trip, root) for p in partitions for trip in list_trips(p, root)]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    q = build_gps_quality()
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    q.to_csv("results/tables/gps_quality_per_trip.csv", index=False)
    summary = q.drop(columns=["partition", "trip_id"]).agg(["min", "median", "max"]).T
    summary.to_csv("results/tables/gps_quality_summary.csv")
    print(summary)

