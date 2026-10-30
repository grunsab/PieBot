#!/bin/bash
# Lineage w2048 (2026-10-30): 2048-wide pairwise arch-v3 net trained with
# bullet on BT4-relabelled binpacks (Leela games plus Stockfish-game positions
# relabelled by BT4; no Stockfish scores, no DFRC). See the campaign plan in
# evidence/nnue_w2048_20261030/README.md.
#
# Run by Supervisor (deploy/vast/piebot_w2048.conf). On start it:
#   - refuses to train unless every file in $DATA/files.tsv is present and
#     sha256-verified (the download script leaves FILE.sha256ok),
#   - resumes from the newest checkpoint whose float weights pass a sanity
#     check (finite, plausible magnitude), else starts fresh,
#   - pins the trainer to the E-cores so the P-cores stay free for matches.
set -euo pipefail
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ulimit -n 65536

REPO=${REPO:-/workspace/piebot}
DATA=${DATA:-/workspace/bt4}
OUT=${OUT:-/workspace/w2048}
NET_ID=${NET_ID:-w2048}
SUPERBATCHES=${SUPERBATCHES:-3000}
CPUS=${CPUS:-16-31}
B=$REPO/training/bullet/target/release/piebot-bullet

ARGS=()
missing=0
while IFS=$'\t' read -r ds path size sha; do
  f="$DATA/${ds//\//__}__$(basename "$path")"
  if [ -f "$f" ] && [ -f "$f.sha256ok" ]; then ARGS+=(--data "$f"); else echo "not verified: $f" >&2; missing=1; fi
done < "$DATA/files.tsv"
[ "$missing" = 0 ] || { echo "refusing to train on a partial corpus" >&2; exit 2; }
echo "corpus: ${#ARGS[@]} args ($(( ${#ARGS[@]} / 2 )) files)"

mkdir -p "$OUT"
RESUME=()
latest=$(python3 "$REPO/scripts/experiments/w2048_watch.py" latest-good --out "$OUT" --net-id "$NET_ID" || true)
if [ -n "$latest" ]; then
  n=${latest##*-}
  RESUME=(--resume "$latest" --start-superbatch $((n + 1)))
  echo "resuming from $latest at superbatch $((n + 1))"
fi

exec taskset -c "$CPUS" "$B" train "${ARGS[@]}" --out "$OUT" --net-id "$NET_ID" \
  --hidden 2048 --pairwise --l1 16 --l2 32 --act-l1 0.1 --buckets 16 \
  --wdl 0.2 --loss bce --lr 0.001 --final-lr-frac 0.03 --seed 2048 \
  --superbatches "$SUPERBATCHES" --save-rate 20 --threads 8 "${RESUME[@]}"
