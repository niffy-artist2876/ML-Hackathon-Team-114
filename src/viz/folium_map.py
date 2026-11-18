import folium
from pathlib import Path
from src.data.loader import read_track, read_nodes, read_arcs, read_route


def arc_coords(arc_ids, nodes, arcs):
    a = arcs.iloc[arc_ids]
    return [[(nodes.lat.iloc[f], nodes.lon.iloc[f]), (nodes.lat.iloc[t], nodes.lon.iloc[t])]
            for f, t in zip(a.from_node, a.to_node)]


def make_map(partition, trip_id, pred_route=None, out_dir="results/maps", max_net_arcs=4000):
    track = read_track(partition, trip_id)
    nodes = read_nodes(partition, trip_id)
    arcs = read_arcs(partition, trip_id)
    route = read_route(partition, trip_id)["arc_id"].values

    m = folium.Map(location=[track.lat.mean(), track.lon.mean()], zoom_start=15, control_scale=True)
    m.fit_bounds([[track.lat.min(), track.lon.min()], [track.lat.max(), track.lon.max()]])

    # road network near the trip (bounding box, capped so the HTML stays small)
    pad = 0.002
    inside = ((nodes.lat.iloc[arcs.from_node].values > track.lat.min() - pad) &
              (nodes.lat.iloc[arcs.from_node].values < track.lat.max() + pad) &
              (nodes.lon.iloc[arcs.from_node].values > track.lon.min() - pad) &
              (nodes.lon.iloc[arcs.from_node].values < track.lon.max() + pad))
    near = arcs.index[inside][:max_net_arcs]
    net = folium.FeatureGroup(name="Road network (nearby)", show=False)
    for seg in arc_coords(near, nodes, arcs):
        folium.PolyLine(seg, color="gray", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    ref = folium.FeatureGroup(name="Reference route")
    for seg in arc_coords(route, nodes, arcs):
        folium.PolyLine(seg, color="green", weight=4, opacity=0.8).add_to(ref)
    ref.add_to(m)

    if pred_route is not None:
        pr = folium.FeatureGroup(name="Predicted route")
        for seg in arc_coords(list(pred_route), nodes, arcs):
            folium.PolyLine(seg, color="red", weight=3, opacity=0.8, dash_array="6").add_to(pr)
        pr.add_to(m)

    gps = folium.FeatureGroup(name="GPS observations")
    for i, r in track.iterrows():
        folium.CircleMarker((r.lat, r.lon), radius=2, color="blue", fill=True,
                            tooltip=f"point {i}, t={r.t:.0f}s").add_to(gps)
    gps.add_to(m)

    legend = """<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;
    padding:8px;border:1px solid grey;font-size:13px">
    <b>Legend</b><br><span style="color:blue">&#9679;</span> GPS points<br>
    <span style="color:green">&#9473;</span> Reference route<br>
    <span style="color:red">&#9473;</span> Predicted route<br>
    <span style="color:gray">&#9473;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl().add_to(m)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(out_dir) / f"{partition}_{trip_id}.html"
    m.save(out)
    return out


if __name__ == "__main__":
    print(make_map("train", "00000001"))
