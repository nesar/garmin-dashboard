#!/usr/bin/env bash
# Nightly refresh: pull Garmin data, parse FIT files, rebuild static site,
# commit + push if anything changed.
#
# Wired up via launchd (scripts/com.nesar.garmin-dashboard.refresh.plist).
# All output goes to logs/nightly.log so you can inspect what happened.
#
# Runs unattended, so it exits fast on any error rather than hanging.
set -euo pipefail

REPO="/Users/nesar/Projects/Private/garmin-dashboard"
PY="/Users/nesar/anaconda3/envs/llm_env/bin/python"
LOG="$REPO/logs/nightly.log"
mkdir -p "$REPO/logs"

exec >> "$LOG" 2>&1
echo ""
echo "==== $(date '+%Y-%m-%d %H:%M:%S %Z') refresh start ===="

cd "$REPO"

# 1) pull wellness + activity summaries + fit files (last 30 activities)
"$PY" ingest/connect_pull.py --days 30 --activities 60 --fit --fit-limit 30

# 2) reparse FIT files if any new ones landed
"$PY" ingest/fit_parse.py || true

# 3) rebuild static site into docs/
"$PY" build_site.py

# 4) commit + push if anything actually changed
if [[ -n "$(git status --porcelain docs store)" ]]; then
  git add docs store
  git commit -m "nightly refresh $(date '+%Y-%m-%d')" || true
  # Do NOT push if there's no remote or credentials aren't cached — just log it.
  if git remote get-url origin >/dev/null 2>&1; then
    git push || echo "git push failed — check credentials / SSH key"
  else
    echo "no origin remote configured — skipping push"
  fi
else
  echo "no changes to commit"
fi

echo "==== $(date '+%Y-%m-%d %H:%M:%S %Z') refresh done ===="
