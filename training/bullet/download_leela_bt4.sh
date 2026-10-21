#!/bin/bash
# Download the Leela-only, BT4-relabelled Stockfish binpacks used to train
# PieBot arch-v3 nets with training/bullet (565 GB in 36 files, checked 2026-10-19).
#
#   training/bullet/download_leela_bt4.sh /workspace/bt4            # download
#   DRY_RUN=1 training/bullet/download_leela_bt4.sh /workspace/bt4  # list only
#
# Files are resolved through the Hugging Face API. For each dataset it takes
# the plain `.binpack` files (Q mapped to centipawns by the relabeller) and
# skips:
#   - the `.q` variants (the same positions with Q * 32000 as the score; the
#     plain files carry the same target and are ~20% smaller),
#   - a whole file when it also exists split into `.part_N` files,
#   - files under 5 GB (MIN_GB), as Triumviratus' recipe does.
# Downloads are resumable (`curl -C -`), run JOBS at a time, and each file is
# checked against the size the API reports; a finished file is never fetched
# again. Datasets with Stockfish self-play positions are deliberately absent.
set -euo pipefail
OUT=${1:?usage: download_leela_bt4.sh OUT_DIR}
JOBS=${JOBS:-4}
MIN_GB=${MIN_GB:-5}
DATASETS=${DATASETS:-"vondele/linrock_relabel_1 vondele/linrock_relabel_2 vondele/from_kaggle_1_relabel vondele/from_kaggle_2_relabel xushawn/test80-bt4-relabel"}
mkdir -p "$OUT"
LIST="$OUT/files.tsv"

: > "$LIST.tmp"
for ds in $DATASETS; do
  curl -sf --retry 5 "https://huggingface.co/api/datasets/$ds/tree/main?recursive=1" |
    python3 -c '
import json, re, sys
ds, min_gb = sys.argv[1], float(sys.argv[2])
files = [f for f in json.load(sys.stdin) if f.get("type") == "file"]
plain = [f for f in files if f["path"].endswith(".binpack") and ".q." not in f["path"]]
names = {f["path"] for f in plain}
for f in plain:
    p = f["path"]
    if p.replace(".binpack", ".part_0.binpack") in names:   # whole file also split: keep the parts
        continue
    if f["size"] < min_gb * 1e9:
        continue
    size = f["size"]
    print(f"{ds}\t{p}\t{size}")
' "$ds" "$MIN_GB" >> "$LIST.tmp"
done
mv "$LIST.tmp" "$LIST"
awk -F'\t' '{n++; s+=$3} END {printf "%d files, %.1f GB\n", n, s/1e9}' "$LIST"
[ -n "${DRY_RUN:-}" ] && { cat "$LIST"; exit 0; }

fetch() {
  ds=$1; path=$2; size=$3; out=$4
  dst="$out/${ds//\//__}__$(basename "$path")"
  if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$size" ]; then return 0; fi
  for try in 1 2 3 4 5; do
    curl -sfL -C - --retry 5 -o "$dst.part" "https://huggingface.co/datasets/$ds/resolve/main/$path" || true
    if [ -f "$dst.part" ] && [ "$(stat -c %s "$dst.part")" = "$size" ]; then mv "$dst.part" "$dst"; echo "done $dst"; return 0; fi
    sleep $((try * 10))
  done
  echo "FAILED $ds/$path" | tee -a "$out/failed.txt"
}
export -f fetch
tr '\t' ' ' < "$LIST" | xargs -P "$JOBS" -L 1 bash -c 'fetch "$0" "$1" "$2" "'"$OUT"'"'
echo ALL_DONE >> "$OUT/download.log"
