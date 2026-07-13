# Nightly refresh via launchd

Runs `ingest → parse → build → commit → push` at 06:15 every day.

## Install

```bash
# 1) copy the plist into your user LaunchAgents dir
cp scripts/com.nesar.garmin-dashboard.refresh.plist \
   ~/Library/LaunchAgents/

# 2) load it
launchctl load ~/Library/LaunchAgents/com.nesar.garmin-dashboard.refresh.plist

# 3) verify it's registered
launchctl list | grep garmin-dashboard
```

## Run it right now (test)

```bash
launchctl start com.nesar.garmin-dashboard.refresh
tail -f logs/nightly.log
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.nesar.garmin-dashboard.refresh.plist
rm ~/Library/LaunchAgents/com.nesar.garmin-dashboard.refresh.plist
```

## Notes

- **MFA:** the first interactive login (via `python ingest/connect_pull.py`) caches
  a Garmin session in `~/.garminconnect`. After that the nightly job runs with no
  MFA prompt. Tokens usually last months; when they expire the job will fail and
  the log will say so — just run the ingest by hand once to refresh.
- **git push:** needs a cached credential (SSH key or gh-cli token) so it can push
  headless. If you haven't set that up, the script logs `git push failed` and moves
  on — the commit still lands locally.
- **conda env path** is hard-coded to `~/anaconda3/envs/llm_env/bin/python` in
  `nightly_refresh.sh`. Edit it if you rename or move the env.
- **Laptop asleep at 06:15?** launchd catches up on next wake, fires once. Fine
  for a MacBook.
