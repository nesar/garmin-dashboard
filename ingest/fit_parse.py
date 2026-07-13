"""
Parse .FIT files into per-second record frames (GPS, HR, cadence, altitude, speed).

Drop exported .FIT files into data/fit/ and run:
    python ingest/fit_parse.py

Produces store/records.parquet (all activities stacked, keyed by activity file).
"""
from __future__ import annotations

from pathlib import Path

import fitdecode
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = ROOT / "data" / "fit"
STORE = ROOT / "store"
STORE.mkdir(exist_ok=True)

# semicircles -> degrees (Garmin stores lat/lon as semicircles in raw FIT)
SEMI = 180.0 / 2**31


def parse_fit(path: Path) -> pl.DataFrame:
    recs = []
    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if not isinstance(frame, fitdecode.FitDataMessage):
                continue
            if frame.name != "record":
                continue
            g = {f.name: f.value for f in frame.fields}
            lat = g.get("position_lat")
            lon = g.get("position_long")
            recs.append(
                {
                    "timestamp": g.get("timestamp"),
                    "lat": lat * SEMI if isinstance(lat, (int, float)) else None,
                    "lon": lon * SEMI if isinstance(lon, (int, float)) else None,
                    "hr": g.get("heart_rate"),
                    "cadence": g.get("cadence"),
                    "speed": g.get("speed"),
                    "altitude": g.get("altitude") or g.get("enhanced_altitude"),
                    "distance": g.get("distance"),
                    "power": g.get("power"),
                }
            )
    df = pl.DataFrame(recs)
    if len(df):
        df = df.with_columns(pl.lit(path.stem).alias("activity"))
    return df


def main():
    files = sorted(FIT_DIR.glob("*.fit")) + sorted(FIT_DIR.glob("*.FIT"))
    if not files:
        print(f"No .fit files in {FIT_DIR}. Export them from Garmin Connect first.")
        return
    frames = [parse_fit(f) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        print("No record messages found.")
        return
    out = pl.concat(frames, how="diagonal")
    out.write_parquet(STORE / "records.parquet")
    print(f"Parsed {len(files)} files -> {len(out)} records at {STORE/'records.parquet'}")


if __name__ == "__main__":
    main()
