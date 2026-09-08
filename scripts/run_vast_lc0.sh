#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.cargo/bin:/venv/main/bin:$PATH"
export PYTHONUNBUFFERED=1
REPO_ROOT="${REPO_ROOT:-/workspace/piebot_lc0_repo}"
OUT_ROOT="${OUT_ROOT:-/workspace/piebot_lc0_20260907}"
HOURS="${HOURS:-720}"
EVICT_RAW="${EVICT_RAW-0}"
DISK_CAPACITY_GB="${DISK_CAPACITY_GB-0}"
case "$EVICT_RAW" in
  0) evict_raw_flag="--no-evict-raw" ;;
  1) evict_raw_flag="--evict-raw" ;;
  *) echo 'EVICT_RAW must be 0 or 1' >&2; exit 2 ;;
esac
cd "$REPO_ROOT"
exec python3 -m training.nnue.lc0_deploy --repo "$REPO_ROOT" \
  --out-root "$OUT_ROOT" --hours "$HOURS" \
  --disk-capacity-gb "$DISK_CAPACITY_GB" "$evict_raw_flag" "$@"
