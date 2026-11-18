import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.loader import list_trips, read_track

T = Path("results/tables")
OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    

def plot_inventory():
    q = pd.read_csv(T / "gps_quality_per_trip.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for part, g in q.groupby("partition"):
        ax[0].hist(g["n_points"], bins=20, alpha=0.6, label=part)
    ax[0].set_xlabel("GPS points per trip"); ax[0].set_ylabel("trips"); ax[0].legend()
    ax[0].set_title("Trip length (partition: train + student_test)")
    ax[1].hist(q["path_length_m"] / 1000, bins=20)
    ax[1].set_xlabel("path length (km)"); ax[1].set_title("Trajectory length")
    save(fig, "eda_trip_sizes.png")



def plot_time_gaps():
    q = pd.read_csv(T / "gps_quality_per_trip.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(q["frac_dt_not_1"], bins=20)
    ax[0].set_xlabel("fraction of steps with dt != 1 s"); ax[0].set_ylabel("trips")
    ax[1].hist(np.log10(q["dt_max"]), bins=20)
    ax[1].set_xlabel("log10(largest time gap, s)"); ax[1].set_title("Largest gap per trip")
    save(fig, "eda_time_gaps.png")
    
def plot_trip_locations():
    q = pd.read_csv(T / "gps_quality_per_trip.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for part, g in q.groupby("partition"):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, label=part, s=25)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.legend()
    ax.set_title("Trip centres (all partitions used)")
    save(fig, "eda_trip_locations.png")

def plot_network_size():
    q = pd.read_csv(T / "network_quality_per_trip.csv")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(q.n_nodes, q.n_arcs, s=15); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].set_title("Network size")
    ax[1].hist(q.largest_weak_frac, bins=20); ax[1].set_xlabel("largest weak component (frac of nodes)")
    ax[2].hist(q.frac_arcs_with_reverse, bins=20); ax[2].set_xlabel("arcs with a reverse arc (frac)")
    save(fig, "eda_network.png")

def plot_route_vs_points():
    g = pd.read_csv(T / "gps_quality_per_trip.csv")
    r = pd.read_csv(T / "route_quality_per_trip.csv")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(r.n_points, r.n_route, s=15); ax[0].set_xlabel("GPS points"); ax[0].set_ylabel("route arcs")
    ax[1].hist(r.frac_connected.dropna(), bins=20); ax[1].set_xlabel("fraction of consecutive route arcs connected")
    save(fig, "eda_route.png")


def plot_trajectories(partition="train", n=6):
    trips = list_trips(partition)[:n]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, trip in zip(axes.ravel(), trips):
        t = read_track(partition, trip)
        ax.plot(t.lon, t.lat, lw=0.8); ax.scatter(t.lon.iloc[0], t.lat.iloc[0], c="g", s=20)
        ax.set_title(f"{partition} / {trip}", fontsize=9); ax.set_aspect(1 / np.cos(np.radians(t.lat.mean())))
    save(fig, f"eda_trajectories_{partition}.png")


if __name__ == "__main__":
    for f in (plot_inventory, plot_time_gaps, plot_trip_locations, plot_network_size,
              plot_route_vs_points, plot_trajectories):
        f()
    print("figures written to", OUT)