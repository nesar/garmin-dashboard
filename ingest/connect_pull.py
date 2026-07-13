"""
Pull wellness + activity data from Garmin Connect into local parquet/DuckDB.

Auth: set GARMIN_EMAIL / GARMIN_PASSWORD env vars, or pass to GarminIngest().
Nothing is hard-coded and no password is stored. A token session is cached by
garminconnect in ~/.garminconnect so you only log in once.

Usage:
    export GARMIN_EMAIL="you@example.com"
    export GARMIN_PASSWORD="..."          # or use a .env you don't commit
    python ingest/connect_pull.py --days 120
"""
from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

STORE = REPO_ROOT / "store"
STORE.mkdir(exist_ok=True)


TOKENSTORE = str(
    Path(os.environ.get("GARMINTOKENS", Path.home() / ".garminconnect")).expanduser()
)


def _prompt_mfa() -> str:
    return input("Garmin MFA code: ").strip()


def _client():
    from garminconnect import Garmin

    email = os.environ.get("GARMIN_EMAIL")
    password = os.environ.get("GARMIN_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "Set GARMIN_EMAIL and GARMIN_PASSWORD env vars first "
            "(do NOT commit them). See module docstring."
        )
    g = Garmin(email=email, password=password, prompt_mfa=_prompt_mfa)
    g.login(TOKENSTORE)
    return g


def pull_daily(g, days: int) -> pl.DataFrame:
    """One row per day: resting HR, HRV, body battery, stress, sleep, steps, SpO2."""
    rows = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        ds = d.isoformat()
        row = {"date": ds}
        try:
            stats = g.get_stats(ds)
            row["resting_hr"] = stats.get("restingHeartRate")
            row["steps"] = stats.get("totalSteps")
            row["stress_avg"] = stats.get("averageStressLevel")
            row["body_battery_high"] = stats.get("bodyBatteryHighestValue")
            row["body_battery_low"] = stats.get("bodyBatteryLowestValue")
        except Exception:
            pass
        try:
            hrv = g.get_hrv_data(ds)
            if hrv and hrv.get("hrvSummary"):
                row["hrv_last_night_avg"] = hrv["hrvSummary"].get("lastNightAvg")
                row["hrv_status"] = hrv["hrvSummary"].get("status")
        except Exception:
            pass
        try:
            sleep = g.get_sleep_data(ds)
            dto = (sleep or {}).get("dailySleepDTO", {}) or {}
            row["sleep_seconds"] = dto.get("sleepTimeSeconds")
            row["deep_seconds"] = dto.get("deepSleepSeconds")
            row["rem_seconds"] = dto.get("remSleepSeconds")
            row["light_seconds"] = dto.get("lightSleepSeconds")
            row["awake_seconds"] = dto.get("awakeSleepSeconds")
            scores = dto.get("sleepScores", {}) or {}
            row["sleep_score"] = (scores.get("overall") or {}).get("value")
        except Exception:
            pass
        rows.append(row)
    return pl.DataFrame(rows).sort("date")


def pull_activities(g, limit: int = 200) -> pl.DataFrame:
    """One row per activity summary."""
    acts = g.get_activities(0, limit)
    rows = []
    for a in acts:
        rows.append(
            {
                "activity_id": a.get("activityId"),
                "name": a.get("activityName"),
                "type": (a.get("activityType") or {}).get("typeKey"),
                "start": a.get("startTimeLocal"),
                "distance_m": a.get("distance"),
                "duration_s": a.get("duration"),
                "avg_hr": a.get("averageHR"),
                "max_hr": a.get("maxHR"),
                "avg_speed": a.get("averageSpeed"),
                "elevation_gain": a.get("elevationGain"),
                "calories": a.get("calories"),
                "avg_cadence": a.get("averageRunningCadenceInStepsPerMinute"),
                "training_load": a.get("activityTrainingLoad"),
            }
        )
    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--activities", type=int, default=200)
    args = ap.parse_args()

    g = _client()
    daily = pull_daily(g, args.days)
    acts = pull_activities(g, args.activities)

    daily.write_parquet(STORE / "daily.parquet")
    acts.write_parquet(STORE / "activities.parquet")
    print(f"Wrote {len(daily)} daily rows, {len(acts)} activities to {STORE}")


if __name__ == "__main__":
    main()
