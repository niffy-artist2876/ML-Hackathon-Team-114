"""Development-only map helpers; reference layers never enter inference."""
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def review_map(g, route, alignment, predicted, output):
    ll = g.lonlat
    roads = g.node_lonlat[g.arcs]
    low, high = ll.min(axis=0) - .005, ll.max(axis=0) + .005
    mask = (roads.max(axis=1) >= low).all(axis=1) & (roads.min(axis=1) <= high).all(axis=1)
    visible = np.flatnonzero(mask)
    visible = visible[::max(1, int(np.ceil(len(visible) / 8000)))]
    view = folium.Map(location=ll.mean(axis=0)[::-1], tiles="OpenStreetMap")
    for name, ids, color, weight, show in [
        ("Network (display subset)", visible, "gray", 1, True),
        ("Reference route", route, "green", 5, True),
        ("Nearest-arc baseline", predicted, "orange", 3, False),
        ("Aligned reference arcs", alignment.reference_arc_id, "purple", 2, False)]:
        group = folium.FeatureGroup(name=name, show=show)
        features = [{"type": "Feature", "properties": {"arc_id": int(i)},
                     "geometry": {"type": "LineString", "coordinates": roads[i].tolist()}}
                    for i in np.unique(ids)]
        folium.GeoJson({"type": "FeatureCollection", "features": features},
                       style_function=lambda _, c=color, w=weight: {"color": c, "weight": w},
                       tooltip=folium.GeoJsonTooltip(fields=["arc_id"])).add_to(group)
        group.add_to(view)
    points = folium.FeatureGroup(name="GPS (sampled display)")
    for i in range(0, len(ll), max(1, int(np.ceil(len(ll) / 1500)))):
        row = alignment.iloc[i]
        folium.CircleMarker(ll[i, ::-1], radius=2, color="royalblue", fill=True,
                            tooltip=f"point {i}; occurrence {row.route_occurrence}; ref {row.reference_arc_id}; {row.reference_distance_m:.1f} m").add_to(points)
    points.add_to(view)
    view.fit_bounds([ll.min(axis=0)[::-1].tolist(), ll.max(axis=0)[::-1].tolist()])
    folium.LayerControl().add_to(view)
    title = (f"<div style='position:fixed;top:10px;left:55px;z-index:9999;background:white;padding:10px'>"
             f"{g.partition}/{g.trip_id} — development review<br>"
             "Blue GPS | Gray network | Green reference | Orange baseline | Purple aligned<br>"
             "Derived labels; not independently verified. Display may be sampled.</div>")
    view.get_root().html.add_child(folium.Element(title))
    view.save(str(output))


def review_figure(g, route, alignment, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    segments = np.stack((g.index.starts, g.index.ends), axis=1)
    xy = g.xy
    low, high = xy.min(axis=0) - 150, xy.max(axis=0) + 150
    visible = (segments.max(axis=1) >= low).all(axis=1) & (segments.min(axis=1) <= high).all(axis=1)
    ax = axes[0]
    ax.add_collection(LineCollection(segments[visible], colors="#cccccc", linewidths=.4))
    ax.add_collection(LineCollection(segments[np.unique(route)], colors="green", linewidths=2, label="Reference route"))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="royalblue", label="GPS")
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="Easting (m)", ylabel="Northing (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax = axes[1]
    ax.plot(alignment.point_idx, alignment.route_occurrence, color="purple", label="Aligned route occurrence")
    ax.set(xlabel="GPS observation index", ylabel="Reference-route occurrence")
    twin = ax.twinx()
    twin.plot(alignment.point_idx, alignment.reference_distance_m, color="darkorange", alpha=.6)
    twin.set_ylabel("Distance to aligned segment (m)", color="darkorange")
    ax.legend(loc="upper left")
    fig.suptitle(f"{g.partition}/{g.trip_id} | EPSG:{g.epsg} | derived reference alignment")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def inventory_figure(inventory, quality, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for partition, color in (("train", "royalblue"), ("student_test", "darkorange")):
        counts = inventory.loc[(inventory.partition == partition) & (inventory.file_type == "track"), "rows"]
        axes[0].hist(counts, bins=20, alpha=.6, label=partition, color=color)
        gaps = quality.loc[quality.partition == partition, "time_gap_max_s"]
        axes[1].scatter(range(len(gaps)), gaps, label=partition, color=color, s=12)
    axes[0].set(xlabel="GPS observations per trip", ylabel="Trips")
    axes[1].set(xlabel="Trip index within partition", ylabel="Largest time gap (s)", yscale="log")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
