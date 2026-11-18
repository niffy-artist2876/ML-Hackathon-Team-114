import numpy as np
import folium
from pathlib import Path
from src.data.loader import read_track, read_nodes, read_arcs, read_route
from src.data.alignment import load_trip_projected, point_segment_dist, align_monotonic


def alignment_map(partition, trip_id, out_dir="results/maps", every=1):
    track = read_track(partition, trip_id)
    nodes = read_nodes(partition, trip_id)
    arcs = read_arcs(partition, trip_id)
    route = read_route(partition, trip_id)["arc_id"].values

    pts, a, b, _ = load_trip_projected(partition, trip_id)
    D = point_segment_dist(pts, a, b)
    pos = align_monotonic(D)
    dist = D[np.arange(len(pos)), pos]

    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15, control_scale=True)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])

    ref = folium.FeatureGroup(name="Reference route")
    ra = arcs.iloc[route]
    for f, t in zip(ra.from_node, ra.to_node):
        folium.PolyLine([(nodes.lat.iloc[f], nodes.lon.iloc[f]), (nodes.lat.iloc[t], nodes.lon.iloc[t])],
                        color="green", weight=4, opacity=0.7).add_to(ref)
    ref.add_to(m)

    gps = folium.FeatureGroup(name="GPS (colour = distance to assigned arc)")
    links = folium.FeatureGroup(name="Point -> assigned arc", show=False)
    for i in range(0, len(track), every):
        d = dist[i]
        col = "blue" if d < 15 else ("orange" if d < 30 else "red")
        folium.CircleMarker((track.lat.iloc[i], track.lon.iloc[i]), radius=3, color=col, fill=True,
                            tooltip=f"pt {i} | route pos {pos[i]} | arc {route[pos[i]]} | {d:.1f} m").add_to(gps)
        f, t = arcs.iloc[route[pos[i]]]
        mid = ((nodes.lat.iloc[f] + nodes.lat.iloc[t]) / 2, (nodes.lon.iloc[f] + nodes.lon.iloc[t]) / 2)
        folium.PolyLine([(track.lat.iloc[i], track.lon.iloc[i]), mid], color="gray", weight=1).add_to(links)
    gps.add_to(m); links.add_to(m)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
    padding:8px;border:1px solid grey;font-size:13px"><b>Alignment check</b><br>
    <span style="color:green">&#9473;</span> reference route<br>
    <span style="color:blue">&#9679;</span> GPS, &lt;15 m from assigned arc<br>
    <span style="color:orange">&#9679;</span> 15-30 m<br>
    <span style="color:red">&#9679;</span> &gt;30 m (suspect label)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"align_{partition}_{trip_id}.html"
    m.save(out)
    return out


if __name__ == "__main__":
    for part, trip, ev in [("train", "00000001", 1), ("student_test", "00000056", 3), ("train", "00000075", 1)]:
        print(alignment_map(part, trip, every=ev))
