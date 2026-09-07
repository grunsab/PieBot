#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/.cargo/bin:/venv/main/bin:$PATH"
export PYTHONUNBUFFERED=1
REPO_ROOT="${REPO_ROOT:-/workspace/piebot_lc0_repo}"
OUT_ROOT="${OUT_ROOT:-/workspace/piebot_lc0_20260907}"
HOURS="${HOURS:-336}"
cd "$REPO_ROOT"
exec python3 -m training.nnue.lc0_deploy --repo "$REPO_ROOT" \
  --out-root "$OUT_ROOT" --hours "$HOURS" "$@"
