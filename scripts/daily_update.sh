#!/usr/bin/env bash
# Daily resumable update for the conference-program scrape.
#
# Re-runs the scrape (which skips pages already in data/programs.csv and throttles
# to the free-tier rate limit), then commits and pushes any new rows. Safe to run
# repeatedly: when everything is scraped it is a no-op. Intended for a daily
# scheduled trigger so the denominator fills in a little more each day until the
# whole source list is covered.
#
# Requirements in the environment:
#   - GEMINI_API_KEY set (free tier is fine; the scrape throttles itself)
#   - config.yaml present with rate_limit_rpm set (e.g. 4)
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${PA_BRANCH:-claude/youthful-archimedes-xr4avp}"
SOURCES="${PA_SOURCES:-data/sources.csv}"
PROGRAMS="${PA_PROGRAMS:-data/programs.csv}"

echo "== $(date -u +%FT%TZ) resuming scrape =="
python -m publication_analyzer -c config.yaml scrape "$SOURCES" -o "$PROGRAMS"

PROGRESS="${PROGRAMS}.progress.json"
if [[ -n "$(git status --porcelain "$PROGRAMS" "$PROGRESS")" ]]; then
  git add "$PROGRAMS" "$PROGRESS"
  git commit -m "Update scraped programs ($(date -u +%F))"
  git push origin "$BRANCH"
  echo "== pushed updated $PROGRAMS (+ resume state) =="
else
  echo "== no new rows; nothing to commit =="
fi
