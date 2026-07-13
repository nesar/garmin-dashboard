"""
Generate realistic synthetic Venu 3 data so the dashboard renders end-to-end
before you wire in your real Garmin credentials. Produces the same parquet
schema that connect_pull.py and fit_parse.py write, so the rest of the
pipeline is identical whether data is real or synthetic.

    python ingest/make_sample.py
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "store"
STORE.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

N_DAYS = 120


def daily_frame() -> pl.DataFrame:
    today = date.today()
    rows = []
    # A slow fitness uptrend: resting HR drifts down, HRV drifts up over the window.
    for i in range(N_DAYS):
        d = today - timedelta(days=N_DAYS - 1 - i)
        t = i / N_DAYS
        dow = d.weekday()
        trained_hard = dow in (1, 3, 5)  # Tue/Thu/Sat harder
        base_rhr = 54 - 4 * t + (2 if trained_hard else 0)
        rhr = base_rhr + rng.normal(0, 1.5)
        hrv = 45 + 18 * t - (6 if trained_hard else 0) + rng.normal(0, 5)
        # Sleep: weekends longer, occasional bad night
        base_sleep = 7.2 + (0.6 if dow in (4, 5) else 0) + rng.normal(0, 0.6)
        base_sleep = max(4.5, base_sleep)
        deep = base_sleep * rng.uniform(0.13, 0.20)
        rem = base_sleep * rng.uniform(0.18, 0.25)
        awake = base_sleep * rng.uniform(0.04, 0.09)
        light = base_sleep - deep - rem - awake
        sleep_score = int(np.clip(60 + (base_sleep - 6) * 12 + rng.normal(0, 6), 30, 98))
        stress = int(np.clip(35 - 8 * t + (10 if trained_hard else 0) + rng.normal(0, 6), 10, 80))
        rows.append(
            {
                "date": d.isoformat(),
                "resting_hr": round(rhr, 1),
                "hrv_last_night_avg": round(max(15, hrv), 1),
                "hrv_status": rng.choice(["balanced", "balanced", "unbalanced", "low"]),
                "steps": int(rng.normal(9000, 2500) + (3000 if trained_hard else 0)),
                "stress_avg": stress,
                "body_battery_high": int(np.clip(rng.normal(82, 8), 40, 100)),
                "body_battery_low": int(np.clip(rng.normal(20, 8), 5, 45)),
                "sleep_seconds": int(base_sleep * 3600),
                "deep_seconds": int(deep * 3600),
                "rem_seconds": int(rem * 3600),
                "light_seconds": int(light * 3600),
                "awake_seconds": int(awake * 3600),
                "sleep_score": sleep_score,
            }
        )
    return pl.DataFrame(rows).sort("date")


def activities_frame() -> pl.DataFrame:
    today = date.today()
    rows = []
    aid = 1_000_000
    for i in range(N_DAYS):
        d = today - timedelta(days=N_DAYS - 1 - i)
        dow = d.weekday()
        if dow not in (1, 3, 5, 6):
            continue
        t = i / N_DAYS
        aid += 1
        if dow == 6:  # long easy run
            dist = rng.uniform(11000, 16000)
            pace = rng.uniform(5.9, 6.4)  # min/km
            kind = "long_run"
        elif dow == 3:  # intervals
            dist = rng.uniform(6000, 9000)
            pace = rng.uniform(4.6, 5.1)
            kind = "intervals"
        else:  # easy / tempo
            dist = rng.uniform(5000, 8000)
            pace = rng.uniform(5.3, 5.8)
            kind = "easy"
        speed = 1000 / (pace * 60)  # m/s
        dur = dist / speed
        avg_hr = int(np.clip(150 + (pace - 5.5) * -12 + rng.normal(0, 5) - 6 * t, 120, 185))
        rows.append(
            {
                "activity_id": aid,
                "name": f"{kind.replace('_', ' ').title()}",
                "type": "running",
                "start": datetime.combine(d, datetime.min.time())
                .replace(hour=7)
                .isoformat(),
                "distance_m": round(dist, 1),
                "duration_s": round(dur, 1),
                "avg_hr": avg_hr,
                "max_hr": int(avg_hr + rng.uniform(8, 20)),
                "avg_speed": round(speed, 3),
                "elevation_gain": round(rng.uniform(20, 180), 1),
                "calories": int(dist / 1000 * rng.uniform(60, 75)),
                "avg_cadence": int(rng.normal(168, 5)),
                "training_load": round(dur / 60 * rng.uniform(1.5, 3.0), 1),
                "kind_true": kind,
            }
        )
    return pl.DataFrame(rows)


def records_frame(acts: pl.DataFrame) -> pl.DataFrame:
    """Per-second GPS+HR tracks. Loops around a park near Lemont, IL area."""
    center_lat, center_lon = 41.673, -87.99
    all_rows = []
    for a in acts.iter_rows(named=True):
        n = int(a["duration_s"])
        n = min(n, 5400)
        avg_hr = a["avg_hr"]
        speed = a["avg_speed"]
        radius = 0.006 + rng.uniform(0, 0.004)
        laps = a["distance_m"] / (2 * math.pi * radius * 111000)
        t0 = datetime.fromisoformat(a["start"])
        hr_drift = np.linspace(0, rng.uniform(4, 12), n)  # cardiac drift
        noise = rng.normal(0, 2, n)
        dist_cum = 0.0
        for s in range(0, n, 2):  # 0.5 Hz to keep files light
            frac = s / n
            ang = laps * 2 * math.pi * frac
            lat = center_lat + radius * math.sin(ang) + rng.normal(0, 0.00008)
            lon = center_lon + radius * math.cos(ang) * 1.3 + rng.normal(0, 0.00008)
            dist_cum += speed * 2
            hr = int(np.clip(avg_hr + hr_drift[s] + noise[s]
                             + 8 * math.sin(frac * math.pi * 6), 90, 195))
            all_rows.append(
                {
                    "timestamp": (t0 + timedelta(seconds=s)).isoformat(),
                    "lat": lat,
                    "lon": lon,
                    "hr": hr,
                    "cadence": int(np.clip(a["avg_cadence"] + rng.normal(0, 4), 150, 190)),
                    "speed": speed + rng.normal(0, 0.3),
                    "altitude": 200 + 15 * math.sin(ang * 2) + rng.normal(0, 1),
                    "distance": dist_cum,
                    "power": None,
                    "activity": str(a["activity_id"]),
                }
            )
    return pl.DataFrame(all_rows)


def main():
    daily = daily_frame()
    acts = activities_frame()
    recs = records_frame(acts)
    daily.write_parquet(STORE / "daily.parquet")
    acts.write_parquet(STORE / "activities.parquet")
    recs.write_parquet(STORE / "records.parquet")
    print(f"Sample data written to {STORE}:")
    print(f"  daily.parquet      {len(daily)} rows")
    print(f"  activities.parquet {len(acts)} rows")
    print(f"  records.parquet    {len(recs)} rows")


if __name__ == "__main__":
    main()
