"""
Live heart-rate monitor over BLE using bleak (runs on your laptop).

The Venu 3 must be broadcasting HR: on the watch, open the activity or use
Settings > Sensors > Broadcast Heart Rate (or 'Broadcast During Activity').
Then run this on the laptop with Bluetooth on:

    python live/hr_monitor.py            # discover + stream, print + log CSV
    python live/hr_monitor.py --scan     # just list nearby BLE HR devices

It subscribes to the standard BLE Heart Rate Service (0x180D) / Measurement
characteristic (0x2A37), parses the flag byte per the BLE spec, and appends
readings to live/hr_log.csv so you can replay/plot them later.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
LOG = Path(__file__).resolve().parent / "hr_log.csv"


def parse_hr(data: bytearray) -> dict:
    """Decode the Heart Rate Measurement characteristic (BLE spec 3.113)."""
    flags = data[0]
    hr16 = flags & 0x01
    idx = 1
    if hr16:
        hr = int.from_bytes(data[idx:idx + 2], "little"); idx += 2
    else:
        hr = data[idx]; idx += 1
    out = {"hr": hr}
    # optional energy-expended field
    if flags & 0x08:
        idx += 2
    # optional RR intervals (for HRV) — 1/1024 s units
    rr = []
    if flags & 0x10:
        while idx + 1 < len(data):
            rr.append(int.from_bytes(data[idx:idx + 2], "little") / 1024.0)
            idx += 2
    if rr:
        out["rr"] = rr
    return out


async def scan():
    print("Scanning 8s for BLE devices advertising Heart Rate service...")
    devices = await BleakScanner.discover(timeout=8.0, service_uuids=[HR_SERVICE])
    if not devices:
        print("None found. Make sure the watch is broadcasting HR and BT is on.")
    for d in devices:
        print(f"  {d.name or '(unknown)'}  [{d.address}]")
    return devices


async def stream(address: str | None):
    if address is None:
        devices = await scan()
        if not devices:
            return
        address = devices[0].address
        print(f"Connecting to {address} ...")

    new = not LOG.exists()
    f = LOG.open("a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["epoch", "hr", "rr_ms"])

    def handler(_char, data: bytearray):
        p = parse_hr(data)
        rr = ";".join(f"{x*1000:.0f}" for x in p.get("rr", []))
        w.writerow([f"{time.time():.3f}", p["hr"], rr]); f.flush()
        bar = "█" * min(40, p["hr"] // 5)
        print(f"\r♥ {p['hr']:3d} bpm  {bar:<40}", end="", flush=True)

    async with BleakClient(address) as client:
        print("Connected. Streaming — Ctrl+C to stop.")
        await client.start_notify(HR_MEASUREMENT, handler)
        try:
            while client.is_connected:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await client.stop_notify(HR_MEASUREMENT)
            f.close()
            print(f"\nSaved to {LOG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="list HR devices and exit")
    ap.add_argument("--address", help="BLE address/UUID to connect to directly")
    args = ap.parse_args()
    try:
        if args.scan:
            asyncio.run(scan())
        else:
            asyncio.run(stream(args.address))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
