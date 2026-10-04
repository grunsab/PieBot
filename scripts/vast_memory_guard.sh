#!/usr/bin/env bash
# Log this container's memory once a minute and stop the training program if
# non-reclaimable memory stays near the cgroup limit. Page cache is excluded:
# it is counted against the limit but the kernel reclaims it on demand.
set -uo pipefail
PROGRAM="${GUARD_PROGRAM:-piebot_lc0v2}"
LOG="${GUARD_LOG:-/workspace/memory_guard.log}"
LIMIT_FRACTION_PCT="${GUARD_LIMIT_PCT:-80}"
STRIKES_NEEDED="${GUARD_STRIKES:-3}"
INTERVAL="${GUARD_INTERVAL_S:-60}"
CG=/sys/fs/cgroup
strikes=0
while true; do
  max=$(cat "$CG/memory.max" 2>/dev/null || echo max)
  [ "$max" = max ] && max=$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
  anon=$(awk '$1=="anon" {print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
  shmem=$(awk '$1=="shmem" {print $2}' "$CG/memory.stat" 2>/dev/null || echo 0)
  current=$(cat "$CG/memory.current" 2>/dev/null || echo 0)
  held=$((anon + shmem))
  pct=$((held * 100 / max))
  echo "$(date -u +%FT%TZ) held_gb=$((held / 1073741824)) held_pct=$pct current_gb=$((current / 1073741824)) limit_gb=$((max / 1073741824))" >> "$LOG"
  if [ "$pct" -ge "$LIMIT_FRACTION_PCT" ]; then
    strikes=$((strikes + 1))
  else
    strikes=0
  fi
  if [ "$strikes" -ge "$STRIKES_NEEDED" ]; then
    echo "$(date -u +%FT%TZ) STOPPING $PROGRAM: held memory at ${pct}% of the limit for $strikes samples" >> "$LOG"
    supervisorctl stop "$PROGRAM" >> "$LOG" 2>&1
    strikes=0
  fi
  sleep "$INTERVAL"
done
