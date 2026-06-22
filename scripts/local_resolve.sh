#!/usr/bin/env bash
# Resolve publications via OpenAlex from a NON-rate-limited IP (e.g. your laptop)
# and update the shared cache, so FT50 rates compute instantly everywhere after.
#
# Why: OpenAlex hard-throttles the cloud IPs this project's automation runs on.
# Running the OpenAlex step once from a normal IP fills data/openalex_cache.json;
# once committed, every later run (local or CI) reuses it with no API calls.
#
# Usage:
#   PA_MAILTO="you@example.com" ./scripts/local_resolve.sh
# (PA_MAILTO is your email for OpenAlex's faster "polite pool" — recommended.)
set -euo pipefail
cd "$(dirname "$0")/.."

export PA_MAILTO="${PA_MAILTO:-movingsaleapril2017@gmail.com}"

echo "== Resolving $(($(wc -l < data/programs.csv)-1)) papers via OpenAlex (no per-run cap) =="
# No --max-lookups here: a normal IP isn't throttled, so resolve everything at once.
python -m publication_analyzer -c config.ci.yaml analyze \
  --programs data/programs.csv \
  --output data/rates.csv \
  --details data/paper_details.csv \
  --cache data/openalex_cache.json

echo "== Committing cache + results =="
git add data/openalex_cache.json data/rates.csv data/paper_details.csv
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "Local OpenAlex resolution: cache + rates ($(date -u +%F))"
  git push origin "$(git rev-parse --abbrev-ref HEAD)"
  echo "== Pushed. Rates now resolve instantly from the cache everywhere. =="
fi
