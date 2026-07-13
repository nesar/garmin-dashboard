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

def acwr(activities: pl.DataFrame) -> pl.DataFrame:
    """
    Acute:Chronic Workload Ratio. Daily training load summed, then
    7-day (acute) vs 28-day (chronic) rolling means. The 0.8-1.3 band is the
    commonly cited 'sweet spot'; >1.5 flags injury-risk spikes.
    """
    a = activities.with_columns(pl.col("start").str.slice(0, 10).alias("date"))
    daily_load = (
        a.group_by("date").agg(pl.col("training_load").sum().alias("load")).sort("date")
    )
    # Reindex to a continuous date spine so rolling windows are calendar-correct.
    spine = pl.DataFrame(
        {"date": pl.date_range(
            daily_load["date"].str.to_date().min(),
            daily_load["date"].str.to_date().max(),
            interval="1d", eager=True).cast(pl.Utf8)}
    )
    dl = spine.join(daily_load, on="date", how="left").with_columns(
        pl.col("load").fill_null(0)
    )
    dl = dl.with_columns(
        pl.col("load").rolling_mean(window_size=7, min_samples=1).alias("acute"),
        pl.col("load").rolling_mean(window_size=28, min_samples=1).alias("chronic"),
    )
    return dl.with_columns(
        (pl.col("acute") / pl.when(pl.col("chronic") > 0).then(pl.col("chronic")).otherwise(None))
        .alias("acwr")
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
