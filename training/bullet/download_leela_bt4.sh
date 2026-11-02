#!/bin/bash
# Download the BT4-relabelled Stockfish-format binpacks used to train PieBot
# arch-v3 nets with training/bullet.
#
#   training/bullet/download_leela_bt4.sh /workspace/bt4            # download
#   DRY_RUN=1 training/bullet/download_leela_bt4.sh /workspace/bt4  # list only
#
# Every file's score is Leela BT4's Q (BT4-tf13tune), never a Stockfish
# evaluation. Positions come from Leela games and, in
# vondele/master-binpacks_relabel, from Stockfish games relabelled by BT4.
#
# Files are resolved through the Hugging Face API. For each dataset it takes
# the plain `.binpack` files (Q mapped to centipawns by the relabeller) and
# skips:
#   - files whose name does not say BT4 (the label source must be explicit),
#   - the `.q` variants and `.zst` copies (same positions, same target),
#   - a whole file when it also exists split into `.part_N` files,
#   - a non-`-histfix` file when a `-histfix` relabel of the same games exists,
#   - names matching EXCLUDE (default: DFRC / Chess960 data),
#   - files under MIN_GB.
# Downloads are resumable (`curl -C -`), run JOBS at a time, and each file is
# checked against the size and LFS sha256 the API reports; a verified file is
# never fetched again.
set -euo pipefail
OUT=${1:?usage: download_leela_bt4.sh OUT_DIR}
JOBS=${JOBS:-4}
MIN_GB=${MIN_GB:-3}
EXCLUDE=${EXCLUDE:-dfrc}
DATASETS=${DATASETS:-"vondele/linrock_relabel_1 vondele/linrock_relabel_2 vondele/from_kaggle_1_relabel vondele/from_kaggle_2_relabel xushawn/test80-bt4-relabel vondele/master-binpacks_relabel"}
mkdir -p "$OUT"
LIST="$OUT/files.tsv"

: > "$LIST.tmp"
for ds in $DATASETS; do
  curl -sf --retry 5 "https://huggingface.co/api/datasets/$ds/tree/main?recursive=1" |
    python3 -c '
import json, re, sys
ds, min_gb, exclude = sys.argv[1], float(sys.argv[2]), sys.argv[3]
files = [f for f in json.load(sys.stdin) if f.get("type") == "file"]
plain = [f for f in files if f["path"].endswith(".binpack") and ".q." not in f["path"]
         and "BT4" in f["path"]]
names = {f["path"] for f in plain}
for f in sorted(plain, key=lambda f: f["path"]):
    p = f["path"]
    if exclude and re.search(exclude, p, re.I):
        continue
    if p.replace(".binpack", ".part_0.binpack") in names:   # whole file also split: keep the parts
        continue
    if "-histfix" not in p and p.replace(".binpack", "-histfix.binpack") in names:
        continue                                             # prefer the history-aware relabel
    if f["size"] < min_gb * 1e9:
        continue
    sha = (f.get("lfs") or {}).get("oid", "")
    size = f["size"]
    print(f"{ds}\t{p}\t{size}\t{sha}")
' "$ds" "$MIN_GB" "$EXCLUDE" >> "$LIST.tmp"
done
mv "$LIST.tmp" "$LIST"
awk -F'\t' '{n++; s+=$3} END {printf "%d files, %.1f GB\n", n, s/1e9}' "$LIST"
[ -n "${DRY_RUN:-}" ] && { cat "$LIST"; exit 0; }

fetch() {
  ds=$1; path=$2; size=$3; sha=$4; out=$5
  dst="$out/${ds//\//__}__$(basename "$path")"
  if [ -f "$dst" ] && [ -f "$dst.sha256ok" ]; then return 0; fi
  for try in 1 2 3 4 5; do
    if [ ! -f "$dst" ]; then
      curl -sfL -C - --retry 5 -o "$dst.part" "https://huggingface.co/datasets/$ds/resolve/main/$path" || true
      [ -f "$dst.part" ] && [ "$(stat -c %s "$dst.part")" = "$size" ] && mv "$dst.part" "$dst"
    fi
    if [ -f "$dst" ]; then
      if [ -z "$sha" ] || [ "$(sha256sum "$dst" | cut -d' ' -f1)" = "$sha" ]; then
        touch "$dst.sha256ok"; echo "done $dst"; return 0
      fi
      echo "sha256 mismatch $dst (try $try)" | tee -a "$out/failed.txt"
      rm -f "$dst"
    fi
    sleep $((try * 10))
  done
  echo "FAILED $ds/$path" | tee -a "$out/failed.txt"
}
export -f fetch
tr '\t' ' ' < "$LIST" | xargs -P "$JOBS" -L 1 bash -c 'fetch "$0" "$1" "$2" "$3" "'"$OUT"'"'
echo ALL_DONE >> "$OUT/download.log"
