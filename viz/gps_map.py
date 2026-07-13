"""GPS activity heatmap from per-second records -> standalone HTML."""
from __future__ import annotations

from pathlib import Path

import folium
import polars as pl
from folium.plugins import HeatMap


def build_map(records: pl.DataFrame, out: Path) -> None:
    pts = records.select(["lat", "lon"]).drop_nulls()
    if not len(pts):
        return
    center = [pts["lat"].mean(), pts["lon"].mean()]
    m = folium.Map(location=center, zoom_start=13, tiles="CartoDB dark_matter")
    HeatMap(
        pts.to_numpy().tolist(),
        radius=6, blur=8, min_opacity=0.3,
        gradient={0.2: "#38BDF8", 0.5: "#4ADE80", 0.8: "#F5A524", 1.0: "#F87171"},
    ).add_to(m)
    m.save(str(out))
