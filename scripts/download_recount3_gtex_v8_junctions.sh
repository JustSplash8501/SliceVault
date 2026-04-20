#!/usr/bin/env bash
set -u

MANIFEST="data/raw/recount3/gtex_v8/files/manifest.tsv"
LOG="data/raw/recount3/gtex_v8/download.log"
FAIL_LOG="data/raw/recount3/gtex_v8/download_failures.log"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

: > "$FAIL_LOG"

while IFS=$'\t' read -r project asset url local_path; do
  if [[ "$project" == "project" ]]; then
    continue
  fi
  mkdir -p "$(dirname "$local_path")"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $project $asset" | tee -a "$LOG"
  if curl -L --fail --retry 5 -C - -o "$local_path" "$url" >> "$LOG" 2>&1 < /dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $project $asset" | tee -a "$LOG"
  else
    rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL  $project $asset rc=$rc" | tee -a "$LOG" "$FAIL_LOG"
  fi

done < "$MANIFEST"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DOWNLOAD_LOOP_COMPLETE" | tee -a "$LOG"
