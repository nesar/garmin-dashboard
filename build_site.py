"""
Build the static dashboard into docs/ for GitHub Pages.

    python build_site.py

Reads store/*.parquet, runs the analysis layer, renders Plotly figures to
self-contained HTML fragments, and writes docs/index.html + docs/heatmap.html.
GitHub Pages serves docs/ as static files — no server needed.
"""
from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from analysis import features as F
from viz import charts as C
from viz.gps_map import build_map

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def fig_html(fig, div_id: str, height=360) -> str:
    fig.update_layout(height=height)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False,
                       div_id=div_id, config={"displayModeBar": False,
                                              "responsive": True})


def kpi(daily, acts, acwr_df):
    last = daily.tail(1)
    n_runs = len(acts)
    total_km = acts["distance_m"].sum() / 1000
    latest_acwr = acwr_df.tail(1)["acwr"][0]
    return [
        ("RESTING HR", f'{last["resting_hr"][0]:.0f}', "bpm"),
        ("HRV LAST NIGHT", f'{last["hrv_last_night_avg"][0]:.0f}', "ms"),
        ("SLEEP SCORE", f'{last["sleep_score"][0]:.0f}', "/100"),
        ("ACWR", f"{latest_acwr:.2f}", "ratio"),
        ("RUNS LOGGED", f"{n_runs}", "activities"),
        ("TOTAL DISTANCE", f"{total_km:.0f}", "km"),
    ]


def build():
    daily = F.load("daily")
    acts = F.load("activities")
    recs = F.load("records")

    hrv = F.hrv_baseline(daily)
    sleep = F.sleep_hours(daily)
    acwr_df = F.acwr(acts)
    cols, mat = F.correlation_matrix(daily)
    clustered = F.cluster_runs(acts)
    drift = F.cardiac_drift(recs)
    sample_act = recs["activity"].unique().to_list()[0]

    figs = {
        "hrv": fig_html(C.hrv_readiness(hrv), "c-hrv"),
        "acwr": fig_html(C.acwr_load(acwr_df), "c-acwr"),
        "rhr": fig_html(C.resting_hr_trend(daily), "c-rhr"),
        "sleep": fig_html(C.sleep_stack(sleep), "c-sleep"),
        "corr": fig_html(C.correlation(cols, mat), "c-corr", height=420),
        "clusters": fig_html(C.clusters(clustered), "c-clusters"),
        "drift": fig_html(C.drift_bars(drift), "c-drift", height=420),
        "trace": fig_html(C.hr_pace_trace(recs, sample_act), "c-trace"),
    }
    kpis = kpi(daily, acts, acwr_df)
    build_map(recs, DOCS / "heatmap.html")

    html = render(figs, kpis)
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print(f"Built site -> {DOCS/'index.html'} and {DOCS/'heatmap.html'}")


def render(figs, kpis) -> str:
    kpi_html = "".join(
        f'<div class="gauge"><span class="gauge-label">{lbl}</span>'
        f'<span class="gauge-val">{val}</span><span class="gauge-unit">{unit}</span></div>'
        for lbl, val, unit in kpis
    )
    return TEMPLATE.format(kpis=kpi_html, **figs)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Telemetry — Venu 3 personal dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{{
  --ink:#0E1116; --panel:#161B22; --panel2:#12161C; --grid:#232A34;
  --text:#C7D0DB; --mute:#7A8798; --green:#4ADE80; --amber:#F5A524;
  --cyan:#38BDF8; --red:#F87171; --violet:#A78BFA;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:
  radial-gradient(1200px 600px at 80% -10%, rgba(56,189,248,.06), transparent),
  var(--ink);color:var(--text);
  font-family:'Space Grotesk',system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 20px 80px}}

/* hero with animated ECG trace */
header{{position:relative;padding:56px 20px 30px;border-bottom:1px solid var(--grid);
  overflow:hidden}}
.ecg{{position:absolute;inset:0;width:100%;height:100%;opacity:.5}}
.eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.32em;
  color:var(--green);text-transform:uppercase;margin:0 0 10px}}
h1{{font-size:clamp(30px,5vw,52px);font-weight:700;margin:0;line-height:1.02;
  letter-spacing:-.02em}}
h1 .dim{{color:var(--mute)}}
.sub{{color:var(--mute);max-width:60ch;margin:14px 0 0;font-size:15px;line-height:1.5}}
.live-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--green);margin-right:8px;box-shadow:0 0 0 0 rgba(74,222,128,.6);
  animation:pulse 1.6s infinite}}
@keyframes pulse{{70%{{box-shadow:0 0 0 9px rgba(74,222,128,0)}}
  100%{{box-shadow:0 0 0 0 rgba(74,222,128,0)}}}}

/* gauges */
.gauges{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:30px 0 8px}}
.gauge{{background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--grid);border-radius:12px;padding:16px 14px;
  display:flex;flex-direction:column;gap:3px}}
.gauge-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.14em;
  color:var(--mute)}}
.gauge-val{{font-family:'IBM Plex Mono',monospace;font-size:30px;font-weight:500;
  color:var(--text);line-height:1}}
.gauge-unit{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--mute)}}

/* sections */
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:22px}}
.card{{background:linear-gradient(180deg,var(--panel),var(--panel2));
  border:1px solid var(--grid);border-radius:14px;padding:8px 8px 4px;overflow:hidden}}
.card.wide{{grid-column:1 / -1}}
.section-tag{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.22em;
  color:var(--cyan);text-transform:uppercase;margin:40px 0 4px;
  padding-top:20px;border-top:1px dashed var(--grid)}}
.maplink{{display:inline-flex;align-items:center;gap:8px;color:var(--ink);
  background:var(--green);text-decoration:none;font-family:'IBM Plex Mono',monospace;
  font-size:13px;font-weight:500;padding:11px 16px;border-radius:10px;margin-top:6px}}
.maplink:hover{{background:#6ee79a}}
footer{{margin-top:50px;color:var(--mute);font-family:'IBM Plex Mono',monospace;
  font-size:12px;line-height:1.7;border-top:1px solid var(--grid);padding-top:20px}}
@media(max-width:900px){{.gauges{{grid-template-columns:repeat(3,1fr)}}
  .grid{{grid-template-columns:1fr}}}}
@media(prefers-reduced-motion:reduce){{.ecg{{animation:none}}
  .live-dot{{animation:none}}}}
</style>
</head>
<body>
<header>
  <svg class="ecg" viewBox="0 0 1200 200" preserveAspectRatio="none">
    <polyline fill="none" stroke="#4ADE80" stroke-width="1.5"
      points="0,100 120,100 150,100 160,60 175,140 190,100 320,100 350,100 360,30 378,170 395,100 520,100 560,100 570,60 585,140 600,100 760,100 800,100 810,30 828,170 845,100 980,100 1020,100 1030,60 1045,140 1060,100 1200,100">
      <animate attributeName="stroke-dasharray" from="0,3000" to="3000,0" dur="4s" repeatCount="indefinite"/>
    </polyline>
  </svg>
  <div class="wrap" style="padding-bottom:0">
    <p class="eyebrow"><span class="live-dot"></span>Garmin Venu 3 · personal telemetry</p>
    <h1>Signals from the wrist.<br><span class="dim">Everything Connect won't show you.</span></h1>
    <p class="sub">Post-hoc analysis of workouts, sleep, HRV, recovery and training
    load — built from raw FIT files and the Garmin Connect API, rendered as a fully
    static site. Numbers below refresh whenever the ingest pipeline runs.</p>
  </div>
</header>

<div class="wrap">
  <div class="gauges">{kpis}</div>

  <div class="section-tag">Recovery & readiness</div>
  <div class="grid">
    <div class="card">{hrv}</div>
    <div class="card">{rhr}</div>
    <div class="card">{sleep}</div>
    <div class="card">{corr}</div>
  </div>

  <div class="section-tag">Training load</div>
  <div class="grid">
    <div class="card wide">{acwr}</div>
  </div>

  <div class="section-tag">Runs & efficiency</div>
  <div class="grid">
    <div class="card">{clusters}</div>
    <div class="card">{drift}</div>
    <div class="card wide">{trace}</div>
  </div>

  <div class="section-tag">Where you moved</div>
  <p class="sub" style="margin-bottom:2px">Every logged GPS point, heat-weighted.</p>
  <a class="maplink" href="heatmap.html">▸ open GPS activity heatmap</a>
  <p class="sub" style="margin-top:26px">Live heart-rate monitor (Web Bluetooth):</p>
  <a class="maplink" href="live.html" style="background:var(--red);color:#fff">♥ open live HR monitor</a>

  <footer>
    Built with garminconnect · fitdecode · polars · duckdb · plotly · folium · scikit-learn<br>
    Static build for GitHub Pages — data shown is synthetic sample data until you run the real ingest.<br>
    Rebuild: <span style="color:var(--green)">python build_site.py</span>
  </footer>
</div>
</body>
</html>"""


if __name__ == "__main__":
    build()
