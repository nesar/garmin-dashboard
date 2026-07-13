"""
Turn feature frames into short human-readable state sentences.

Each `n_*` function returns a small block of prose (2-4 sentences) describing
the *current* state of that feature, suitable for placement next to its chart.
All are defensive against missing data — if the input is empty or the latest
value is null they return a short "no data yet" note instead of crashing.
"""
from __future__ import annotations

import polars as pl


def _latest_nonnull(df: pl.DataFrame, col: str):
    if df.is_empty() or col not in df.columns:
        return None
    s = df.filter(df[col].is_not_null())
    return s[col][-1] if len(s) else None


def _fmt(val, spec=".0f"):
    return format(val, spec) if val is not None else "—"


# ---------- recovery & readiness ----------

def n_hrv(hrv: pl.DataFrame) -> str:
    last = _latest_nonnull(hrv, "hrv_last_night_avg")
    base = _latest_nonnull(hrv, "hrv_60d")
    if last is None or base is None:
        return ("Not enough nightly HRV data yet to compare against your 60-day "
                "baseline. Wear the watch overnight for at least a couple of weeks.")
    delta = last - base
    band = "above" if delta > 0 else "below" if delta < 0 else "at"
    verdict = ("Nervous system looking well-recovered."
               if delta > 3 else
               "Recovery is trending soft — extra sleep or an easy day pays off."
               if delta < -3 else
               "You're sitting right on your baseline — a normal recovery day.")
    return (f"Last night: {_fmt(last)} ms, {_fmt(abs(delta))} ms {band} your "
            f"60-day baseline of {_fmt(base)} ms. {verdict} The shaded band shows "
            "±1σ of your rolling baseline; readings inside it are typical for you.")


def n_rhr(daily: pl.DataFrame) -> str:
    last = _latest_nonnull(daily, "resting_hr")
    if last is None:
        return "No resting heart-rate readings recorded yet."
    d = daily.sort("date")
    roll = d["resting_hr"].rolling_mean(window_size=7, min_samples=2)
    r7 = None
    for v in reversed(roll.to_list()):
        if v is not None:
            r7 = v
            break
    if r7 is None:
        return f"Latest RHR: {_fmt(last)} bpm."
    trend = ("edging down (fitness signal — good)" if last < r7 - 1 else
             "edging up (fatigue or illness — watch it)" if last > r7 + 1 else
             "steady with the 7-day average")
    return (f"Most recent resting HR: {_fmt(last)} bpm; 7-day average sits at "
            f"{_fmt(r7)} bpm. You are {trend}. Big multi-day swings usually track "
            "sleep, stress, or an incoming cold before you notice symptoms.")


def n_sleep(sleep: pl.DataFrame) -> str:
    last = sleep.filter(sleep["sleep_h"].is_not_null()).sort("date").tail(7)
    if last.is_empty():
        return "No sleep sessions recorded yet."
    avg = last["sleep_h"].mean() or 0
    deep = last["deep_h"].mean() or 0
    rem = last["rem_h"].mean() or 0
    ok = 7.0 <= avg <= 9.0
    verdict = ("in the healthy 7-9 h window" if ok else
               "short of the 7 h recommended for adults" if avg < 7 else
               "on the long side, which can indicate under-recovery")
    return (f"Past 7 nights averaged {_fmt(avg, '.1f')} h total — {verdict}. "
            f"Roughly {_fmt(deep, '.1f')} h deep and {_fmt(rem, '.1f')} h REM per "
            "night. Deep sleep repairs the body; REM consolidates memory and mood.")


def n_correlation(cols, mat) -> str:
    """Highlight the strongest positive and negative wellness correlations."""
    import numpy as np
    n = len(cols)
    if n < 2 or mat is None:
        return "Not enough overlapping data to compute correlations."
    strongest_pos = (-1.0, None)
    strongest_neg = (1.0, None)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(mat[i, j])
            if np.isnan(v):
                continue
            if v > strongest_pos[0]:
                strongest_pos = (v, (cols[i], cols[j]))
            if v < strongest_neg[0]:
                strongest_neg = (v, (cols[i], cols[j]))
    parts = []
    if strongest_pos[1]:
        a, b = strongest_pos[1]
        parts.append(f"the strongest positive link is {a.replace('_',' ')} ↔ "
                     f"{b.replace('_',' ')} at r={strongest_pos[0]:.2f}")
    if strongest_neg[1] and strongest_neg[0] < 0:
        a, b = strongest_neg[1]
        parts.append(f"the strongest negative link is {a.replace('_',' ')} ↔ "
                     f"{b.replace('_',' ')} at r={strongest_neg[0]:.2f}")
    if not parts:
        return "No standout correlations across your daily wellness signals yet."
    return ("Across the daily wellness signals: " + "; ".join(parts) +
            ". Warm cells move together, cool cells push against each other.")


# ---------- training load ----------

def n_acwr(acwr_df: pl.DataFrame) -> str:
    last = _latest_nonnull(acwr_df, "acwr")
    acute = _latest_nonnull(acwr_df, "acute")
    chronic = _latest_nonnull(acwr_df, "chronic")
    if last is None:
        return "Not enough activity load logged yet to compute an ACWR."
    if last < 0.8:
        band = ("under-loading — the acute week is well below your rolling four-week "
                "average, and fitness will drift down if this persists")
    elif last <= 1.3:
        band = "inside the 0.8-1.3 sweet spot — a productive training load"
    elif last <= 1.5:
        band = "elevated — one more heavy week would push you into the injury-risk zone"
    else:
        band = "above 1.5 — this is the classical injury-risk spike"
    return (f"Current ACWR: {_fmt(last, '.2f')} (acute {_fmt(acute)} vs chronic "
            f"{_fmt(chronic)}). You are {band}. The green band on the chart is the "
            "sweet spot; short green line above bars = daily ratio.")


def n_fitness_fatigue(ff: pl.DataFrame) -> str:
    ctl = _latest_nonnull(ff, "ctl")
    atl = _latest_nonnull(ff, "atl")
    tsb = _latest_nonnull(ff, "tsb")
    if ctl is None:
        return "Not enough sessions logged for the CTL/ATL model yet."
    if tsb is None:
        return f"Fitness (CTL) at {_fmt(ctl, '.1f')}; fatigue (ATL) at {_fmt(atl, '.1f')}."
    if tsb > 10:
        state = "very fresh — race-ready if you have a session on the calendar"
    elif tsb > 0:
        state = "moderately fresh — sensible for a hard workout"
    elif tsb > -10:
        state = "productively fatigued — a normal training block"
    else:
        state = "deeply fatigued — schedule a recovery day before the next hard effort"
    return (f"Fitness (CTL): {_fmt(ctl, '.1f')}. Fatigue (ATL): {_fmt(atl, '.1f')}. "
            f"Form (TSB): {_fmt(tsb, '+.1f')} — you are {state}. CTL grows slowly "
            "with consistent training; ATL spikes with hard days; TSB (CTL−ATL) "
            "flips positive when the block is doing its job.")


# ---------- runs ----------

def n_clusters(clustered: pl.DataFrame) -> str:
    if clustered.is_empty():
        return "No runs to cluster yet."
    counts = clustered["cluster_name"].value_counts().sort("count", descending=True)
    parts = [f"{row['count']} {row['cluster_name'].lower()}" for row in counts.iter_rows(named=True)]
    return ("Your logged runs auto-group into three intensity buckets: "
            + ", ".join(parts) + ". Bubble size = distance; horizontal axis = "
            "average HR; vertical = average speed. A healthy training week clusters "
            "most sessions into 'Easy' with one or two harder points.")


def n_drift(drift: pl.DataFrame) -> str:
    if drift.is_empty():
        return "No FIT files parsed yet — cardiac drift needs per-second HR & pace."
    d = drift.filter(drift["decoupling_pct"].is_not_null())
    if d.is_empty():
        return "No decoupling values computed yet."
    med = float(d["decoupling_pct"].median() or 0)
    n_hi = int((d["decoupling_pct"] > 5).sum())
    verdict = ("Median decoupling under 2% — very aerobically efficient."
               if med < 2 else
               "Median around 5% — normal for tempo/threshold effort."
               if med < 5 else
               "Median above 5% — heat, dehydration or under-training may be at play.")
    return (f"Across your logged runs the median HR-vs-pace decoupling is "
            f"{med:.1f}%; {n_hi} runs exceed 5%. {verdict} Aerobic decoupling "
            "compares HR/speed in the first and second half of a session — a "
            "rising ratio at fixed effort is fatigue in the tank.")


def n_trace(recs: pl.DataFrame) -> str:
    if recs.is_empty():
        return "No per-second traces — parse FIT files to see this chart."
    acts = recs["activity"].unique().to_list()
    return (f"A representative run picked from {len(acts)} activities in the "
            "per-second store. HR (red) climbs quickly and settles; speed (green) "
            "usually mirrors terrain. Divergence between them tells you when the "
            "effort was harder than the pace looked.")


# ---------- weekly + sleep consistency + zones + recovery + body battery ----------

def n_weekly(weekly: pl.DataFrame) -> str:
    if weekly.is_empty():
        return "No weekly activity data yet."
    last = weekly.tail(4)
    total_min = float(last["minutes"].sum() or 0)
    total_km = float(last["km"].sum() or 0)
    sessions = int(last["sessions"].sum() or 0)
    return (f"Across the last four ISO weeks you logged {sessions} sessions "
            f"totalling {total_min:.0f} minutes and {total_km:.1f} km. Bars are "
            "weekly minutes (left axis); green line is kilometres (right axis). "
            "Consistency — not any single big week — is what compounds.")


def n_sleep_consistency(sc: pl.DataFrame) -> str:
    latest_sd = _latest_nonnull(sc, "sleep_h_sd_30d")
    latest_mean = _latest_nonnull(sc, "sleep_h_30d")
    if latest_sd is None or latest_mean is None:
        return "Need at least a couple of weeks of sleep data for the consistency band."
    verdict = ("very consistent — a strong recovery ally"
               if latest_sd < 0.6 else
               "moderately variable — try tightening bedtime by 30 min"
               if latest_sd < 1.0 else
               "swinging — irregular sleep is a bigger HRV drag than most people expect")
    return (f"30-day mean: {_fmt(latest_mean, '.1f')} h; standard deviation "
            f"{_fmt(latest_sd, '.2f')} h. Your recent sleep is {verdict}. The wider "
            "the violet band, the more your bedtimes and durations swing night-to-night.")


def n_hr_zones(z: pl.DataFrame) -> str:
    if z.is_empty():
        return "No activities with HR data yet."
    total = float(z["minutes"].sum() or 0)
    if total == 0:
        return "No HR-tagged minutes recorded yet."
    def pct(name):
        row = z.filter(z["zone"] == name)
        return 100 * float(row["minutes"].sum() or 0) / total
    z2 = pct("Z2"); z4_5 = pct("Z4") + pct("Z5")
    return (f"Z2 (aerobic base) accounts for {z2:.0f}% of logged time; Z4-Z5 "
            f"(threshold+) for {z4_5:.0f}%. The polarised-training rule of thumb "
            "is 80% easy (Z1-Z2) and ~20% hard (Z4+); too much middle-zone (Z3) "
            "quietly costs recovery without adding much fitness.")


def n_recovery(rec: pl.DataFrame) -> str:
    latest = _latest_nonnull(rec, "recovery")
    if latest is None:
        return ("Composite recovery score needs at least 10 days of overlapping HRV, "
                "resting HR, and sleep-score data.")
    if latest >= 70:
        band = "green — go train"
    elif latest >= 40:
        band = "amber — an easy or moderate day"
    else:
        band = "red — rest or sleep in"
    return (f"Today's composite readiness: {latest:.0f}/100 — {band}. Blends HRV "
            "(40%), resting HR (30%) and last night's sleep score (30%) against "
            "each metric's own 60-day baseline. Bands on the chart mark the "
            "green/amber/red zones.")


def n_body_battery(bb: pl.DataFrame) -> str:
    high = _latest_nonnull(bb, "body_battery_high")
    low = _latest_nonnull(bb, "body_battery_low")
    if high is None or low is None:
        return "No body-battery data yet."
    span = high - low
    verdict = ("You are draining efficiently and recharging fully — a good sign of "
               "matched training and recovery."
               if span > 60 else
               "Battery isn't swinging much day-to-day — either activity or sleep "
               "recovery is muted.")
    return (f"Most recent day: peaked at {high:.0f} and drained to {low:.0f} "
            f"(swing of {span:.0f}). {verdict} Green = daily peak, red = daily "
            "trough; the tinted band shows how much you both spent and refilled.")
