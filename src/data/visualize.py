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
    low, high = ll.min(axis=0) - .008, ll.max(axis=0) + .008
    mask = (roads.max(axis=1) >= low).all(axis=1) & (roads.min(axis=1) <= high).all(axis=1)
    visible = np.flatnonzero(mask)
    visible = visible[::max(1, int(np.ceil(len(visible) / 8000)))]

    # Center map on the median GPS coordinates with rich interactive basemaps
    center_lat, center_lon = float(np.median(ll[:, 1])), float(np.median(ll[:, 0]))
    view = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)

    # Base tile layers: OpenStreetMap, Esri Satellite, Esri Street Map, CartoDB Light, CartoDB Dark
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True,
    ).add_to(view)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri WorldImagery",
        name="Esri Satellite (Photorealistic)",
        control=True,
    ).add_to(view)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri WorldStreetMap",
        name="Esri Street Map",
        control=True,
    ).add_to(view)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        subdomains="abcd",
        max_zoom=20,
        name="CartoDB Positron (Clean Light)",
        control=True,
    ).add_to(view)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap &copy; CARTO",
        subdomains="abcd",
        max_zoom=20,
        name="CartoDB Dark (Night Mode)",
        control=True,
    ).add_to(view)

    # 1. Road network layer (display subset for performance)
    net_group = folium.FeatureGroup(name="Road Network (Nearby Arcs)", show=True)
    net_features = [{"type": "Feature", "properties": {"arc_id": int(i)},
                     "geometry": {"type": "LineString", "coordinates": roads[i].tolist()}}
                    for i in np.unique(visible)]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": net_features},
        style_function=lambda _: {"color": "#607D8B", "weight": 1.5, "opacity": 0.55},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Road Arc ID:"])
    ).add_to(net_group)
    net_group.add_to(view)

    # 2. Reference route (ground-truth sequence)
    ref_group = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    ref_features = [{"type": "Feature", "properties": {"arc_id": int(i)},
                     "geometry": {"type": "LineString", "coordinates": roads[i].tolist()}}
                    for i in np.unique(route)]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": ref_features},
        style_function=lambda _: {"color": "#00C853", "weight": 5, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Reference Arc:"])
    ).add_to(ref_group)
    ref_group.add_to(view)

    # 3. Aligned reference arcs (assigned by dynamic programming)
    aligned_group = folium.FeatureGroup(name="Aligned Reference Arcs", show=True)
    aligned_ids = alignment.reference_arc_id.to_numpy(dtype=int)
    aligned_features = [{"type": "Feature", "properties": {"arc_id": int(i)},
                         "geometry": {"type": "LineString", "coordinates": roads[i].tolist()}}
                        for i in np.unique(aligned_ids)]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": aligned_features},
        style_function=lambda _: {"color": "#7C4DFF", "weight": 3.5, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Aligned Arc:"])
    ).add_to(aligned_group)
    aligned_group.add_to(view)

    # 4. Nearest-arc baseline predictions
    baseline_group = folium.FeatureGroup(name="Nearest-Arc Baseline Predictions", show=False)
    base_features = [{"type": "Feature", "properties": {"arc_id": int(i)},
                      "geometry": {"type": "LineString", "coordinates": roads[i].tolist()}}
                     for i in np.unique(predicted)]
    folium.GeoJson(
        {"type": "FeatureCollection", "features": base_features},
        style_function=lambda _: {"color": "#FF9100", "weight": 3, "opacity": 0.85, "dashArray": "6, 6"},
        tooltip=folium.GeoJsonTooltip(fields=["arc_id"], aliases=["Baseline Arc:"])
    ).add_to(baseline_group)
    baseline_group.add_to(view)

    # 5. Far reference matches (>50m warning links)
    links_group = folium.FeatureGroup(name="Far Reference Matches (>50m)", show=True)
    far_subset = alignment.loc[alignment.reference_far]
    for row in far_subset.itertuples():
        a, b = g.arcs[row.reference_arc_id]
        closest = g.node_lonlat[a] + row.reference_fraction * (g.node_lonlat[b] - g.node_lonlat[a])
        folium.PolyLine(
            [ll[row.point_idx, ::-1].tolist(), closest[::-1].tolist()],
            color="#FF1744", weight=2.5, opacity=0.9, dash_array="4, 4",
            tooltip=f"Point {row.point_idx}: Gap = {row.reference_distance_m:.1f} m to Arc {row.reference_arc_id}"
        ).add_to(links_group)
    links_group.add_to(view)

    # 6. GPS Observations (sampled markers with detailed tooltips)
    pts_group = folium.FeatureGroup(name="GPS Observations", show=True)
    step = max(1, int(np.ceil(len(ll) / 1500)))
    for i in range(0, len(ll), step):
        row = alignment.iloc[i]
        popup_html = (
            f"<div style='font-family:Arial,sans-serif;font-size:12px;min-width:180px;line-height:1.4'>"
            f"<b style='color:#1565C0'>GPS Observation #{i}</b><br>"
            f"<b>Timestamp:</b> {g.timestamps[i]:.1f} s<br>"
            f"<b>Assigned Arc:</b> {row.reference_arc_id}<br>"
            f"<b>Route Occurrence:</b> {row.route_occurrence}<br>"
            f"<b>Distance to Arc:</b> {row.reference_distance_m:.2f} m<br>"
            f"<b>Lon, Lat:</b> {ll[i, 0]:.6f}, {ll[i, 1]:.6f}"
            f"</div>"
        )
        pt_color = "#FF1744" if row.reference_far else "#2979FF"
        folium.CircleMarker(
            location=[ll[i, 1], ll[i, 0]],
            radius=3.5,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color=pt_color,
            fill_opacity=0.9,
            tooltip=f"Point #{i} | Arc {row.reference_arc_id} ({row.reference_distance_m:.1f}m)",
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(pts_group)
    pts_group.add_to(view)

    # 7. Start and Finish Milestone Markers
    folium.Marker(
        location=[ll[0, 1], ll[0, 0]],
        popup=f"<b>Trip Start: Point #0</b><br>Time: {g.timestamps[0]}s",
        tooltip="Trip Start Point #0",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(view)
    folium.Marker(
        location=[ll[-1, 1], ll[-1, 0]],
        popup=f"<b>Trip End: Point #{len(ll)-1}</b><br>Time: {g.timestamps[-1]}s",
        tooltip=f"Trip End Point #{len(ll)-1}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
    ).add_to(view)

    # Zoom bounds to GPS trajectory with a small margin
    view.fit_bounds([[float(ll[:, 1].min()), float(ll[:, 0].min())],
                     [float(ll[:, 1].max()), float(ll[:, 0].max())]])

    folium.LayerControl(position="topright", collapsed=False).add_to(view)

    # Floating rich dashboard legend overlay
    far_count = int(alignment.reference_far.sum())
    med_dist = float(alignment.reference_distance_m.median())
    p95_dist = float(alignment.reference_distance_m.quantile(0.95))
    overlay_html = f"""
    <div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:rgba(255,255,255,0.95);
                backdrop-filter:blur(6px);padding:14px 18px;border-radius:12px;
                box-shadow:0 8px 24px rgba(0,0,0,0.22);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
                font-size:12px;color:#212121;max-width:340px;line-height:1.5;border:1px solid rgba(0,0,0,0.08)">
      <div style="font-size:14px;font-weight:700;color:#0D47A1;margin-bottom:4px;display:flex;align-items:center;gap:6px">
        <span>📍</span> PathSnap Route Map: {g.partition}/{g.trip_id}
      </div>
      <div style="color:#555;font-size:11px;margin-bottom:10px">
        CRS: <b>EPSG:{g.epsg}</b> | Points: <b>{len(ll):,}</b> | Route: <b>{len(route)} arcs</b>
      </div>
      <div style="background:#F5F7FA;padding:8px 10px;border-radius:6px;margin-bottom:10px;font-size:11px">
        Median Distance: <b>{med_dist:.1f} m</b> &nbsp;|&nbsp; 95%: <b>{p95_dist:.1f} m</b><br>
        Points >50m: <b style="color:{'#D32F2F' if far_count > 0 else '#388E3C'}">{far_count} ({far_count/len(ll):.1%})</b>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;font-size:11px">
        <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:#2979FF;vertical-align:middle;margin-right:4px"></span>GPS Points</div>
        <div><span style="display:inline-block;width:14px;height:4px;background:#00C853;vertical-align:middle;margin-right:4px"></span>Ref Route</div>
        <div><span style="display:inline-block;width:14px;height:4px;background:#7C4DFF;vertical-align:middle;margin-right:4px"></span>Aligned Arcs</div>
        <div><span style="display:inline-block;width:14px;height:3px;background:#FF9100;vertical-align:middle;margin-right:4px;border-top:1px dashed #FF9100"></span>Baseline</div>
        <div><span style="display:inline-block;width:14px;height:2px;background:#607D8B;vertical-align:middle;margin-right:4px"></span>Road Network</div>
        <div><span style="display:inline-block;width:14px;height:2px;background:#FF1744;vertical-align:middle;margin-right:4px;border-top:1px dashed #FF1744"></span>Gap &gt;50m</div>
      </div>
    </div>
    """
    view.get_root().html.add_child(folium.Element(overlay_html))
    view.save(str(output))


def review_figure(g, route, alignment, output):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
    assigned = alignment.reference_arc_id.to_numpy(dtype=int)
    fraction = alignment.reference_fraction.to_numpy()
    snapped = g.index.starts[assigned] + fraction[:, None] * (g.index.ends[assigned] - g.index.starts[assigned])
    far = alignment.reference_far.to_numpy(dtype=bool)
    ax.add_collection(LineCollection(segments[visible], colors="#dddddd", linewidths=.4))
    ax.add_collection(LineCollection(segments[np.unique(assigned)], colors="purple", linewidths=2, label="Assigned arcs"))
    ax.add_collection(LineCollection(np.stack((xy[far], snapped[far]), axis=1), colors="red", linewidths=.4, alpha=.3, label="Matches >50m"))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="royalblue", label="GPS")
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel="Easting (m)", ylabel="Northing (m)")
    ax.set_aspect("equal")
    ax.legend()
    ax = axes[2]
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
