"""
Chart builders. Each returns a Plotly figure themed to the instrument-panel
palette. build_site.py stitches them into the static page.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

# instrument-panel palette
INK = "#0E1116"
PANEL = "#161B22"
GRID = "#232A34"
TEXT = "#C7D0DB"
MUTE = "#7A8798"
GREEN = "#4ADE80"
AMBER = "#F5A524"
CYAN = "#38BDF8"
RED = "#F87171"
VIOLET = "#A78BFA"

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTE)),
    hoverlabel=dict(bgcolor=PANEL, font=dict(family="IBM Plex Mono", color=TEXT)),
)


def _fig(title: str) -> go.Figure:
    f = go.Figure()
    f.update_layout(**LAYOUT, title=dict(text=title, font=dict(color=TEXT, size=15)))
    return f


def hrv_readiness(d: pl.DataFrame) -> go.Figure:
    f = _fig("HRV readiness — nightly reading vs 60-day baseline band")
    x = d["date"].to_list()
    f.add_trace(go.Scatter(x=x, y=d["hrv_hi"], line=dict(width=0), showlegend=False,
                           hoverinfo="skip"))
    f.add_trace(go.Scatter(x=x, y=d["hrv_lo"], fill="tonexty", line=dict(width=0),
                           fillcolor="rgba(56,189,248,0.12)", name="baseline ±1σ",
                           hoverinfo="skip"))
    f.add_trace(go.Scatter(x=x, y=d["hrv_60d"], line=dict(color=CYAN, width=2, dash="dot"),
                           name="60-day"))
    f.add_trace(go.Scatter(x=x, y=d["hrv_last_night_avg"], mode="markers+lines",
                           line=dict(color=TEXT, width=1),
                           marker=dict(size=5, color=GREEN), name="nightly"))
    f.update_yaxes(title="HRV (ms)")
    return f


def acwr_load(d: pl.DataFrame) -> go.Figure:
    f = _fig("Training load & ACWR — the 0.8–1.3 'sweet spot' band")
    x = d["date"].to_list()
    f.add_trace(go.Bar(x=x, y=d["load"], marker_color="rgba(245,165,36,0.35)", name="daily load"))
    f.add_trace(go.Scatter(x=x, y=d["acute"], line=dict(color=AMBER, width=2), name="acute (7d)"))
    f.add_trace(go.Scatter(x=x, y=d["chronic"], line=dict(color=CYAN, width=2), name="chronic (28d)"))
    f.add_trace(go.Scatter(x=x, y=d["acwr"], line=dict(color=GREEN, width=2), name="ACWR",
                           yaxis="y2"))
    f.add_hrect(y0=0.8, y1=1.3, line_width=0, fillcolor="rgba(74,222,128,0.08)", yref="y2")
    f.update_layout(
        yaxis=dict(title="load", **{k: LAYOUT["yaxis"][k] for k in ("gridcolor",)}),
        yaxis2=dict(title="ACWR", overlaying="y", side="right", range=[0, 2.2],
                    gridcolor="rgba(0,0,0,0)", color=GREEN),
    )
    return f


def resting_hr_trend(d: pl.DataFrame) -> go.Figure:
    f = _fig("Resting heart rate — long-term fitness proxy")
    x = d["date"].to_list()
    roll = d["resting_hr"].rolling_mean(window_size=7, min_samples=2)
    f.add_trace(go.Scatter(x=x, y=d["resting_hr"], mode="markers",
                           marker=dict(size=4, color=MUTE), name="daily"))
    f.add_trace(go.Scatter(x=x, y=roll, line=dict(color=RED, width=2.5), name="7-day avg"))
    f.update_yaxes(title="bpm")
    return f


def sleep_stack(d: pl.DataFrame) -> go.Figure:
    f = _fig("Sleep architecture — stacked stages per night")
    x = d["date"].to_list()
    for name, col, color in [("deep", "deep_h", VIOLET), ("rem", "rem_h", CYAN),
                             ("light", "light_h", "rgba(122,135,152,0.6)")]:
        f.add_trace(go.Bar(x=x, y=d[col], name=name, marker_color=color))
    f.update_layout(barmode="stack", yaxis=dict(title="hours", gridcolor=GRID))
    return f


def correlation(cols, mat) -> go.Figure:
    labels = [c.replace("_", " ") for c in cols]
    f = _fig("What moves together — wellness correlation matrix")
    f.add_trace(go.Heatmap(z=mat, x=labels, y=labels, zmid=0,
                           colorscale=[[0, RED], [0.5, INK], [1, GREEN]],
                           text=np.round(mat, 2), texttemplate="%{text}",
                           textfont=dict(size=10, color=TEXT)))
    f.update_layout(margin=dict(l=110, r=20, t=40, b=90))
    return f


def clusters(a: pl.DataFrame) -> go.Figure:
    f = _fig("Runs auto-clustered by pace, HR & distance (k-means)")
    palette = {"Easy": GREEN, "Steady": CYAN, "Fast": AMBER, "Other": MUTE}
    for name in a["cluster_name"].unique().to_list():
        sub = a.filter(pl.col("cluster_name") == name)
        f.add_trace(go.Scatter(
            x=sub["avg_hr"], y=(sub["distance_m"] / sub["duration_s"] * 3.6),
            mode="markers", name=name,
            marker=dict(size=(sub["distance_m"] / 1000 + 5).to_list(),
                        color=palette.get(name, MUTE), opacity=0.75,
                        line=dict(width=0.5, color=INK)),
            text=[f"{d/1000:.1f} km" for d in sub["distance_m"]],
        ))
    f.update_xaxes(title="avg HR (bpm)")
    f.update_yaxes(title="avg speed (km/h)")
    return f


def drift_bars(d: pl.DataFrame) -> go.Figure:
    f = _fig("Aerobic decoupling per run — HR drift at fixed pace")
    d = d.sort("decoupling_pct")
    colors = [RED if v > 5 else AMBER if v > 2 else GREEN for v in d["decoupling_pct"]]
    f.add_trace(go.Bar(y=d["activity"], x=d["decoupling_pct"], orientation="h",
                       marker_color=colors))
    f.update_xaxes(title="decoupling %")
    f.update_yaxes(title="", showticklabels=False)
    f.update_layout(height=max(300, 14 * len(d)))
    return f


def hr_pace_trace(recs: pl.DataFrame, activity: str) -> go.Figure:
    r = recs.filter(pl.col("activity") == activity).sort("timestamp")
    f = _fig(f"Run telemetry — HR & speed over time")
    x = list(range(len(r)))
    f.add_trace(go.Scatter(x=x, y=r["hr"], line=dict(color=RED, width=1.5), name="HR"))
    f.add_trace(go.Scatter(x=x, y=(r["speed"] * 3.6), line=dict(color=GREEN, width=1.5),
                           name="km/h", yaxis="y2"))
    f.update_layout(
        xaxis=dict(title="sample", gridcolor=GRID),
        yaxis=dict(title="bpm", gridcolor=GRID, color=RED),
        yaxis2=dict(title="km/h", overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)", color=GREEN),
    )
    return f
