"""
Feature engineering + modeling on top of the parquet store.

Everything reads from store/*.parquet (real or synthetic) and returns polars/
pandas frames the viz layer consumes. Uses DuckDB where a SQL window is cleaner
than a dataframe op.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "store"


def load(name: str) -> pl.DataFrame:
    return pl.read_parquet(STORE / f"{name}.parquet")


# ---------- daily wellness ----------

def hrv_baseline(daily: pl.DataFrame) -> pl.DataFrame:
    """7-day and 60-day rolling HRV baseline (readiness band)."""
    d = daily.sort("date").with_columns(
        pl.col("hrv_last_night_avg").rolling_mean(window_size=7, min_samples=3).alias("hrv_7d"),
        pl.col("hrv_last_night_avg").rolling_mean(window_size=60, min_samples=10).alias("hrv_60d"),
        pl.col("hrv_last_night_avg").rolling_std(window_size=60, min_samples=10).alias("hrv_60d_sd"),
    )
    return d.with_columns(
        (pl.col("hrv_60d") - pl.col("hrv_60d_sd")).alias("hrv_lo"),
        (pl.col("hrv_60d") + pl.col("hrv_60d_sd")).alias("hrv_hi"),
    )


def sleep_hours(daily: pl.DataFrame) -> pl.DataFrame:
    return daily.with_columns(
        (pl.col("sleep_seconds") / 3600).alias("sleep_h"),
        (pl.col("deep_seconds") / 3600).alias("deep_h"),
        (pl.col("rem_seconds") / 3600).alias("rem_h"),
        (pl.col("light_seconds") / 3600).alias("light_h"),
    )


# ---------- training load ----------

def _session_load(a: pl.DataFrame) -> pl.DataFrame:
    """
    Per-activity 'load' number.

    Uses Garmin's `training_load` when populated. Otherwise falls back to a
    TRIMP-ish proxy: minutes × (avg_hr / 180)^2. The proxy is not TSS but
    behaves proportionally to intensity × duration, which is what CTL/ATL
    and ACWR care about.
    """
    return a.with_columns(
        pl.when(pl.col("training_load").is_not_null() & (pl.col("training_load") > 0))
        .then(pl.col("training_load"))
        .otherwise(
            (pl.col("duration_s").fill_null(0) / 60.0)
            * ((pl.col("avg_hr").fill_null(120) / 180.0) ** 2)
        )
        .alias("load")
    )


def _daily_load_spine(activities: pl.DataFrame) -> pl.DataFrame:
    """Continuous daily load series (zero-filled) covering the activity range."""
    a = _session_load(activities).with_columns(
        pl.col("start").str.slice(0, 10).alias("date")
    )
    daily_load = a.group_by("date").agg(pl.col("load").sum()).sort("date")
    if daily_load.is_empty():
        return daily_load
    spine = pl.DataFrame(
        {"date": pl.date_range(
            daily_load["date"].str.to_date().min(),
            daily_load["date"].str.to_date().max(),
            interval="1d", eager=True).cast(pl.Utf8)}
    )
    return spine.join(daily_load, on="date", how="left").with_columns(
        pl.col("load").fill_null(0)
    )


def acwr(activities: pl.DataFrame) -> pl.DataFrame:
    """
    Acute:Chronic Workload Ratio. Daily training load summed, then
    7-day (acute) vs 28-day (chronic) rolling means. The 0.8-1.3 band is the
    commonly cited 'sweet spot'; >1.5 flags injury-risk spikes.
    """
    dl = _daily_load_spine(activities)
    if dl.is_empty():
        return pl.DataFrame({"date": [], "load": [], "acute": [], "chronic": [], "acwr": []})
    dl = dl.with_columns(
        pl.col("load").rolling_mean(window_size=7, min_samples=1).alias("acute"),
        pl.col("load").rolling_mean(window_size=28, min_samples=1).alias("chronic"),
    )
    return dl.with_columns(
        (pl.col("acute") / pl.when(pl.col("chronic") > 0).then(pl.col("chronic")).otherwise(None))
        .alias("acwr")
    )


def fitness_fatigue(activities: pl.DataFrame,
                    ctl_tau: int = 42, atl_tau: int = 7) -> pl.DataFrame:
    """
    Banister-style Chronic Training Load (fitness), Acute Training Load (fatigue),
    and Training Stress Balance (form = CTL − ATL).

    CTL and ATL are exponentially-weighted moving averages of the daily load
    with time constants of 42 and 7 days respectively. Positive TSB = fresh,
    negative TSB = accumulating fatigue.
    """
    dl = _daily_load_spine(activities)
    if dl.is_empty():
        return pl.DataFrame({"date": [], "load": [], "ctl": [], "atl": [], "tsb": []})
    loads = dl["load"].to_numpy()
    ctl = np.zeros_like(loads, dtype=float)
    atl = np.zeros_like(loads, dtype=float)
    kc = 1 - np.exp(-1 / ctl_tau)
    ka = 1 - np.exp(-1 / atl_tau)
    for i, x in enumerate(loads):
        prev_c = ctl[i - 1] if i else 0.0
        prev_a = atl[i - 1] if i else 0.0
        ctl[i] = prev_c + kc * (x - prev_c)
        atl[i] = prev_a + ka * (x - prev_a)
    return dl.with_columns(
        pl.Series("ctl", ctl),
        pl.Series("atl", atl),
        pl.Series("tsb", ctl - atl),
    )


# ---------- weekly rollup ----------

def weekly_summary(activities: pl.DataFrame) -> pl.DataFrame:
    """Per-ISO-week: sessions, minutes, kilometres, avg HR (activity-weighted)."""
    if activities.is_empty():
        return pl.DataFrame({"week": [], "sessions": [], "minutes": [],
                             "km": [], "avg_hr": []})
    a = activities.with_columns(
        pl.col("start").str.slice(0, 10).str.to_date().alias("d"),
    ).with_columns(
        (pl.col("d").dt.year().cast(pl.Utf8) + "-W"
         + pl.col("d").dt.week().cast(pl.Utf8).str.zfill(2)).alias("week"),
    )
    return (
        a.group_by("week")
        .agg(
            pl.len().alias("sessions"),
            (pl.col("duration_s").sum() / 60).round(0).alias("minutes"),
            (pl.col("distance_m").sum() / 1000).round(1).alias("km"),
            pl.col("avg_hr").mean().round(0).alias("avg_hr"),
        )
        .sort("week")
    )


# ---------- sleep consistency ----------

def sleep_consistency(daily: pl.DataFrame) -> pl.DataFrame:
    """
    30-day rolling std of nightly sleep hours. Lower = more consistent bedtime
    and duration; larger = swinging schedule (a known HRV/recovery drag).
    """
    d = daily.sort("date").with_columns(
        (pl.col("sleep_seconds") / 3600).alias("sleep_h"),
    )
    return d.with_columns(
        pl.col("sleep_h").rolling_mean(window_size=30, min_samples=5).alias("sleep_h_30d"),
        pl.col("sleep_h").rolling_std(window_size=30, min_samples=5).alias("sleep_h_sd_30d"),
    )


# ---------- HR zone distribution ----------

# HR max estimate for zone bucketing. Uses avg_hr per activity, not per-second.
DEFAULT_MAX_HR = 190


def hr_zone_minutes(activities: pl.DataFrame, max_hr: int = DEFAULT_MAX_HR) -> pl.DataFrame:
    """
    Total minutes per HR zone across all activities, bucketed by session avg HR.
    Coarser than per-second zone time (which needs FIT data), but useful signal.
    """
    if activities.is_empty():
        return pl.DataFrame({"zone": ["Z1", "Z2", "Z3", "Z4", "Z5"],
                             "minutes": [0, 0, 0, 0, 0]})
    a = activities.filter(pl.col("avg_hr").is_not_null()
                          & pl.col("duration_s").is_not_null())
    zones = a.with_columns(
        ((pl.col("avg_hr") / max_hr) * 100).alias("pct"),
        (pl.col("duration_s") / 60).alias("minutes"),
    ).with_columns(
        pl.when(pl.col("pct") < 60).then(pl.lit("Z1"))
        .when(pl.col("pct") < 70).then(pl.lit("Z2"))
        .when(pl.col("pct") < 80).then(pl.lit("Z3"))
        .when(pl.col("pct") < 90).then(pl.lit("Z4"))
        .otherwise(pl.lit("Z5")).alias("zone")
    )
    out = zones.group_by("zone").agg(pl.col("minutes").sum().round(0)).sort("zone")
    # ensure all zones present (0 if missing)
    all_z = pl.DataFrame({"zone": ["Z1", "Z2", "Z3", "Z4", "Z5"]})
    return all_z.join(out, on="zone", how="left").with_columns(
        pl.col("minutes").fill_null(0)
    )


# ---------- recovery composite ----------

def recovery_score(daily: pl.DataFrame) -> pl.DataFrame:
    """
    Composite 0-100 daily readiness score.

    Blends three normalised sub-scores:
      • HRV vs 60-day baseline  (higher good)
      • Resting HR vs 60-day baseline (lower good)
      • Sleep score (already 0-100)

    Missing components are dropped from the weighted average of that day so
    the result is defined whenever at least one component is present.
    """
    d = daily.sort("date").with_columns(
        pl.col("hrv_last_night_avg").rolling_mean(window_size=60, min_samples=10).alias("_hrv_b"),
        pl.col("hrv_last_night_avg").rolling_std(window_size=60, min_samples=10).alias("_hrv_s"),
        pl.col("resting_hr").rolling_mean(window_size=60, min_samples=10).alias("_rhr_b"),
        pl.col("resting_hr").rolling_std(window_size=60, min_samples=10).alias("_rhr_s"),
    )
    # z-scores → 0-100 (50 = baseline, ±2σ = 0/100)
    hrv_z = (pl.col("hrv_last_night_avg") - pl.col("_hrv_b")) / pl.col("_hrv_s")
    rhr_z = -(pl.col("resting_hr") - pl.col("_rhr_b")) / pl.col("_rhr_s")  # lower RHR = better
    to_score = lambda z: (50 + 25 * z).clip(0, 100)
    d = d.with_columns(
        to_score(hrv_z).alias("_hrv_score"),
        to_score(rhr_z).alias("_rhr_score"),
        pl.col("sleep_score").cast(pl.Float64).alias("_sleep_score"),
    )
    # weighted average of whichever components are present per row
    parts = ["_hrv_score", "_rhr_score", "_sleep_score"]
    weights = [0.4, 0.3, 0.3]
    num = sum(pl.col(p).fill_null(0) * w for p, w in zip(parts, weights))
    den = sum(pl.when(pl.col(p).is_not_null()).then(w).otherwise(0)
              for p, w in zip(parts, weights))
    return d.with_columns(
        pl.when(den > 0).then(num / den).otherwise(None).alias("recovery")
    ).drop(["_hrv_b", "_hrv_s", "_rhr_b", "_rhr_s",
            "_hrv_score", "_rhr_score", "_sleep_score"])


# ---------- body battery ----------

def body_battery_series(daily: pl.DataFrame) -> pl.DataFrame:
    """Convenience: high/low band per day for the body-battery chart."""
    return daily.sort("date").select(
        ["date", "body_battery_high", "body_battery_low"]
    )


# ---------- per-second workout metrics ----------

def cardiac_drift(records: pl.DataFrame) -> pl.DataFrame:
    """
    Aerobic decoupling per activity: compares HR/speed in the first vs second
    half of each run. Positive % = cardiovascular drift (HR rising for the same
    pace), a fatigue / heat / dehydration signal.
    """
    con = duckdb.connect()
    con.register("r", records.to_pandas())
    q = """
    WITH ordered AS (
        SELECT activity, hr, speed,
               row_number() OVER (PARTITION BY activity ORDER BY timestamp) AS rn,
               count(*)     OVER (PARTITION BY activity)                    AS n
        FROM r WHERE hr IS NOT NULL AND speed > 0.5
    ),
    halves AS (
        SELECT activity,
               CASE WHEN rn <= n/2 THEN 'first' ELSE 'second' END AS half,
               avg(hr/speed) AS hr_per_speed
        FROM ordered GROUP BY activity, half
    )
    SELECT activity,
           max(CASE WHEN half='first'  THEN hr_per_speed END) AS first_half,
           max(CASE WHEN half='second' THEN hr_per_speed END) AS second_half
    FROM halves GROUP BY activity
    """
    out = con.execute(q).pl()
    return out.with_columns(
        ((pl.col("second_half") - pl.col("first_half")) / pl.col("first_half") * 100)
        .round(2)
        .alias("decoupling_pct")
    )


def cluster_runs(activities: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    """
    K-means on pace / HR / distance to auto-label run 'types' (easy/tempo/
    interval-ish) without you tagging them.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    a = activities.filter(pl.col("type") == "running").with_columns(
        (pl.col("distance_m") / pl.col("duration_s")).alias("speed_ms")
    )
    feats = a.select(["speed_ms", "avg_hr", "distance_m"]).drop_nulls()
    if len(feats) < k:
        return a.with_columns(pl.lit(0).alias("cluster"))
    X = StandardScaler().fit_transform(feats.to_numpy())
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    a = a.with_columns(pl.Series("cluster", labels))
    # Name clusters by mean speed: fastest -> "Fast/Intervals"
    order = (
        a.group_by("cluster").agg(pl.col("speed_ms").mean().alias("ms")).sort("ms")
    )
    names = ["Easy", "Steady", "Fast"][: order.height]
    mapping = {c: n for c, n in zip(order["cluster"].to_list(), names)}
    return a.with_columns(
        pl.col("cluster").replace_strict(mapping, default="Other").alias("cluster_name")
    )


def correlation_matrix(daily: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    cols = ["resting_hr", "hrv_last_night_avg", "sleep_score",
            "stress_avg", "steps", "body_battery_high"]
    df = daily.select(cols).drop_nulls().to_pandas()
    return cols, df.corr().to_numpy()


if __name__ == "__main__":
    daily, acts, recs = load("daily"), load("activities"), load("records")
    print("HRV baseline tail:\n", hrv_baseline(daily).tail(3).select(["date", "hrv_7d", "hrv_60d"]))
    print("\nACWR tail:\n", acwr(acts).tail(3).select(["date", "acute", "chronic", "acwr"]))
    print("\nCardiac drift head:\n", cardiac_drift(recs).head(3))
    print("\nClusters:\n", cluster_runs(acts)["cluster_name"].value_counts())
