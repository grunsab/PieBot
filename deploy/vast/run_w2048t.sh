#!/bin/bash
# Lineage w2048t: the w2048 net plus threat inputs (PIENNQ03 version 6), the
# second run of evidence/nnue_w2048_20261030 (see threats_design.md). Same
# data, schedule and length as run_w2048.sh; only --threats differs, so the
# two nets compare directly. Width is HIDDEN (2048 by default; 1024 only with
# the owner's approval, per the plan's NPS decision gate).
#
# Not installed by default. When the plain run has finished and the GPU is free:
#   install deploy/vast/piebot_w2048t.conf into /etc/supervisor/conf.d.
#
# Like run_w2048.sh it refuses a partially verified corpus, resumes from the
# newest checkpoint that passes the watcher's check, and pins the trainer to
# the E-cores. INIT_FROM=<plain checkpoint dir> starts from a plain pairwise
# checkpoint of the same width with zero threat weights instead of from scratch.
# LR and FINAL_LR_FRAC override the schedule (a fine-tune wants a lower LR).
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ulimit -n 65536

REPO=${REPO:-/workspace/piebot}
DATA=${DATA:-/workspace/bt4}
OUT=${OUT:-/workspace/w2048t}
NET_ID=${NET_ID:-w2048t}
HIDDEN=${HIDDEN:-2048}
SUPERBATCHES=${SUPERBATCHES:-3000}
LR=${LR:-0.001}
FINAL_LR_FRAC=${FINAL_LR_FRAC:-0.03}
CPUS=${CPUS:-16-31}
B=$REPO/training/bullet/target/release/piebot-bullet

ARGS=()
missing=0
while IFS=$'\t' read -r ds path size sha; do
  f="$DATA/${ds//\//__}__$(basename "$path")"
  if [ -f "$f" ] && [ -f "$f.sha256ok" ]; then ARGS+=(--data "$f"); else echo "not verified: $f" >&2; missing=1; fi
done < "$DATA/files.tsv"
[ "$missing" = 0 ] || { echo "refusing to train on a partial corpus" >&2; exit 2; }
echo "corpus: $(( ${#ARGS[@]} / 2 )) files; threat inputs; hidden $HIDDEN"

mkdir -p "$OUT"
START=()
latest=$(python3 "$REPO/scripts/experiments/w2048_watch.py" latest-good --out "$OUT" --net-id "$NET_ID" \
  --threats --hidden "$HIDDEN" || true)
if [ -n "$latest" ]; then
  n=${latest##*-}
  START=(--resume "$latest" --start-superbatch $((n + 1)))
  echo "resuming from $latest at superbatch $((n + 1))"
elif [ -n "${INIT_FROM:-}" ]; then
  START=(--init-from "$INIT_FROM")
  echo "starting from $INIT_FROM with zero threat weights"
fi

exec taskset -c "$CPUS" "$B" train "${ARGS[@]}" --out "$OUT" --net-id "$NET_ID" \
  --hidden "$HIDDEN" --pairwise --l1 16 --l2 32 --threats --act-l1 0.1 --buckets 16 \
  --wdl 0.2 --loss bce --lr "$LR" --final-lr-frac "$FINAL_LR_FRAC" --seed 2048 \
  --superbatches "$SUPERBATCHES" --save-rate 20 --threads 8 "${START[@]}"
