#!/bin/bash
# Deploy campaign_v5 (h128 fresh student, depth-9 node-capped teacher) on the
# Threadripper box. Fetches the pinned commit from origin, so the box needs
# GitHub access but no side-channel file transfer.
set -uo pipefail
export PATH="/root/.cargo/bin:/venv/main/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
log(){ echo "[deploy_v5 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

NEW_SHA="${NEW_SHA:?export NEW_SHA to the 40-char commit to deploy}"

log "step 1: graceful stop of piebot_campaign_v2 (v4 lineage)"
supervisorctl stop piebot_campaign_v2
for i in $(seq 1 72); do
  status=$(supervisorctl status piebot_campaign_v2 | awk '{print $2}')
  case "$status" in STOPPED|EXITED|FATAL) break;; esac
  sleep 5
done
supervisorctl status piebot_campaign_v2

if pgrep -f '[t]raining.nnue.autopilot' >/dev/null; then
  log "FATAL: autopilot orphan survived group stop"; exit 1
fi
if pgrep -f '[s]elfplay --|[r]elabel_jsonl|[c]ompare_play' >/dev/null; then
  log "workers still winding down; waiting 90s"
  sleep 90
  if pgrep -f '[s]elfplay --|[r]elabel_jsonl|[c]ompare_play' >/dev/null; then
    log "FATAL: worker processes persist after stop"; exit 1
  fi
fi
log "v4 stopped cleanly; state preserved at /workspace/piebot_campaign_v4"

log "step 2: update checkout to $NEW_SHA"
cd /workspace/piebot_rust || exit 1
git fetch origin campaign-v2 || { log "FATAL: origin fetch failed"; exit 1; }
git checkout -B campaign-v2 "$NEW_SHA" || { log "FATAL: checkout failed"; exit 1; }
actual=$(git rev-parse HEAD)
if [ "$actual" != "$NEW_SHA" ]; then
  log "FATAL: SHA mismatch: $actual"; exit 1
fi
log "checkout verified at $actual"

log "step 3: install v5 supervisor conf and start"
cp deploy/vast/piebot_campaign_v2.conf /etc/supervisor/conf.d/piebot_campaign_v2.conf || exit 1
supervisorctl reread
supervisorctl update
sleep 5
supervisorctl start piebot_campaign_v2 2>/dev/null || true
sleep 30
supervisorctl status piebot_campaign_v2
log "launcher log tail:"
tail -n 25 /workspace/piebot_campaign_v2_supervisor.log
log "DEPLOY-V5-DONE"
