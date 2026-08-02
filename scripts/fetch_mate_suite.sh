#!/usr/bin/env bash
set -euo pipefail

# Fetch and prepare Lichess puzzle dataset, then build Mate-in-N suite files.
#
# Outputs:
# - PieBot/data/lichess_db_puzzle.csv.zst (download)
# - PieBot/data/lichess_db_puzzle.csv (decompressed)
# - PieBot/src/suites/matein7.txt (JSONL: {"fen":"...","best":"..."})
# - PieBot/src/suites/matein6.txt, matein5.txt, ... as needed to total 1000 positions
#
# Usage:
#   scripts/fetch_mate_suite.sh

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/PieBot/data"
OUT_DIR="$ROOT_DIR/PieBot/src/suites"
URL="https://database.lichess.org/lichess_db_puzzle.csv.zst"
ZST_FILE="$DATA_DIR/lichess_db_puzzle.csv.zst"
CSV_FILE="$DATA_DIR/lichess_db_puzzle.csv"

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "[fetch] Downloading Lichess puzzle database..."
if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$ZST_FILE"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ZST_FILE" "$URL"
else
  echo "Error: neither curl nor wget is installed. Please install one and retry." >&2
  exit 1
fi

echo "[fetch] Decompressing (zstd)..."
if command -v unzstd >/dev/null 2>&1; then
  unzstd -f "$ZST_FILE" -o "$CSV_FILE"
elif command -v zstd >/dev/null 2>&1; then
  zstd -d -f "$ZST_FILE" -o "$CSV_FILE"
else
  echo "Error: zstd not installed. Please install zstd (brew install zstd, apt-get install zstd, etc.)." >&2
  exit 1
fi

echo "[build] Generating Mate-in-N suite files (total=1000, prefer mateIn7)..."
(
  cd "$ROOT_DIR/PieBot"
  cargo run --locked --release --bin build_mate_suite -- \
    --input "$CSV_FILE" \
    --out "$OUT_DIR" \
    --total 1000
)

echo "[done] Suite files written under $OUT_DIR"
