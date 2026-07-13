# Garmin Venu 3 — personal telemetry dashboard

Pulls everything your Venu 3 records (workouts, sleep, HRV, Body Battery, stress,
resting HR, SpO2, per-second GPS/HR from FIT files), runs analysis Garmin Connect
doesn't (ACWR training load, HRV readiness bands, aerobic decoupling, run
clustering), and renders it as a **fully static site** you can host on GitHub
Pages — plus a **live heart-rate monitor** that streams over Bluetooth.

## What's inside

```
garmin-dashboard/
├── ingest/
│   ├── connect_pull.py   # garminconnect API → parquet (daily wellness + activities)
│   ├── fit_parse.py      # raw .FIT files → per-second records parquet
│   └── make_sample.py    # synthetic data so the site renders before you wire in real data
├── analysis/features.py  # ACWR, HRV baseline, cardiac drift, k-means run clustering
├── viz/                  # plotly charts + folium GPS heatmap
├── build_site.py         # assembles docs/index.html + heatmap.html (GitHub Pages)
├── live/hr_monitor.py    # bleak local live-HR monitor (Python)
├── docs/                 # ← published static site
│   ├── index.html        # the dashboard
│   ├── heatmap.html      # GPS activity heatmap
│   └── live.html         # Web Bluetooth live HR (works ON GitHub Pages)
└── .github/workflows/build.yml  # auto-rebuild + deploy
```

## Quick start

```bash
pip install -r requirements.txt
python ingest/make_sample.py   # generates realistic sample data
python build_site.py           # builds docs/
# open docs/index.html in a browser
```

## Using your real data

1. **API pull** — put your credentials in a local `.env` (git-ignored):
   ```env
   GARMIN_EMAIL=you@example.com
   GARMIN_PASSWORD=...
   ```
   Then:
   ```bash
   python ingest/connect_pull.py --days 180 --fit --fit-limit 60
   ```
   `--fit` auto-downloads original `.fit` files for the most recent N activities
   into `data/fit/`. No manual export needed. MFA is handled interactively on
   first login; the session token is cached in `~/.garminconnect` so subsequent
   runs are silent.
2. **Parse FIT** (for GPS heatmap + per-second HR/pace charts):
   ```bash
   python ingest/fit_parse.py
   ```
3. **Rebuild** the site: `python build_site.py`.

### Automate it

`scripts/nightly_refresh.sh` + a launchd plist run the full pipeline daily and
push updated `docs/` to GitHub Pages. See [`scripts/README.md`](scripts/README.md).

> Credentials are read from env vars and never written to disk or committed
> (`.env` and `store/` are git-ignored). Don't paste your password into any file
> that gets pushed.

## Live heart rate

**Locally (Python / bleak):**
```bash
python live/hr_monitor.py --scan   # find the watch
python live/hr_monitor.py          # stream + log to live/hr_log.csv
```
Enable **Broadcast Heart Rate** on the watch first.

**In the browser (Web Bluetooth):** open `docs/live.html` (or the live page on
your published site) in Chrome/Edge/Opera and click connect. This works directly
on GitHub Pages because it runs in the browser — no server needed. (Safari/iOS
don't support Web Bluetooth.)

## Deploy to GitHub Pages

1. Push this folder to a GitHub repo.
2. Repo **Settings → Pages → Source: Deploy from a branch**, branch `main`,
   folder **`/docs`**. (Or use the included Actions workflow for auto-deploy.)
3. Your site goes live at `https://<user>.github.io/<repo>/`.

To refresh: rerun ingest + `build_site.py` locally and commit `docs/`, or let the
weekly Actions workflow rebuild it.

## The analyses, briefly

- **Recovery composite** — daily 0-100 readiness score blending HRV, resting HR
  and sleep score against their own 60-day baselines.
- **HRV readiness** — nightly HRV against a rolling 60-day baseline ±1σ band.
- **Fitness / fatigue / form (CTL/ATL/TSB)** — Banister EWMA of daily load;
  positive TSB = fresh, negative = accumulating fatigue.
- **ACWR** — acute (7d) vs chronic (28d) training load; the 0.8–1.3 band is the
  commonly cited low-injury-risk zone.
- **Weekly volume** — minutes + km + sessions by ISO week.
- **HR zone minutes** — cumulative time by zone across all activities.
- **Sleep architecture + consistency** — nightly stacked stages plus a 30-day
  mean-and-spread band that flags irregular schedules.
- **Body battery** — daily high/low envelope.
- **Aerobic decoupling** — HR-per-pace in the first vs second half of each run; a
  fatigue/heat/fitness signal (needs FIT files).
- **Run clustering** — k-means on pace/HR/distance auto-labels easy/steady/fast.
- **Correlation matrix** — how sleep, HRV, stress, steps, resting HR move together.

Built with garminconnect · fitdecode · polars · duckdb · plotly · folium · scikit-learn · bleak.
