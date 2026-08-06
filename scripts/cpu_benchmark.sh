#!/usr/bin/env bash
set -Eeuo pipefail

# Portable CPU benchmark using PieBot's actual training workload, so machine
# offers can be compared on the numbers that set campaign cycle time:
#   1. single-thread search (deterministic matein3 suite, NPS)
#   2. self-play generation throughput (book openings, adjudication, NNUE)
#   3. teacher relabel throughput at depth 5 (the dominant cycle cost)
# Deterministic seeds: the same work runs on every machine.
#
# Usage: scripts/cpu_benchmark.sh [output.json]
# Env: REPO_ROOT (default: repo containing this script), THREADS_LIST
#      (default "16 32 max"), BENCH_GAMES (default 60).

export PATH="/root/.cargo/bin:/venv/main/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
OUT_JSON="${1:-$REPO_ROOT/cpu_benchmark_result.json}"
THREADS_LIST="${THREADS_LIST:-16 32 max}"
BENCH_GAMES="${BENCH_GAMES:-60}"
MODEL="$REPO_ROOT/models/cycle_000098_quant.nnue"
BOOK="$REPO_ROOT/books/openings_v1.fen"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

[ -f "$MODEL" ] || { echo "missing committed model: $MODEL" >&2; exit 1; }
[ -f "$BOOK" ] || { echo "missing opening book: $BOOK" >&2; exit 1; }

EFFECTIVE=$(python3 - <<'PY' 2>/dev/null || nproc
import math, os
from pathlib import Path
counts = [os.cpu_count() or 1]
if hasattr(os, "sched_getaffinity"):
    counts.append(len(os.sched_getaffinity(0)))
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.is_file():
    quota_raw, period_raw = cpu_max.read_text().split()[:2]
    if quota_raw != "max":
        counts.append(max(1, math.floor(int(quota_raw) / int(period_raw))))
print(min(counts))
PY
)
CPU_MODEL=$( (lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//') || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)

echo "[bench] cpu='$CPU_MODEL' effective_cores=$EFFECTIVE"
echo "[bench] building release binaries"
cargo build --locked --release --quiet \
  --bin accept --bin selfplay --bin relabel_jsonl \
  --manifest-path "$REPO_ROOT/PieBot/Cargo.toml"

echo "[bench] 1/3 single-thread search (matein3 depth 7)"
cd "$REPO_ROOT/PieBot"
ST_LINE=$(PIEBOT_SUITE_FILE=src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 \
  ./target/release/accept 2>/dev/null | tail -1)
echo "  $ST_LINE"
ST_NPS=$(echo "$ST_LINE" | grep -o 'nps=[0-9.]*' | cut -d= -f2)
ST_NODES=$(echo "$ST_LINE" | grep -o 'nodes=[0-9]*' | cut -d= -f2)
cd "$REPO_ROOT"

echo "[bench] 2/3 self-play throughput ($BENCH_GAMES games, all cores)"
SP_START=$(date +%s.%N)
"$REPO_ROOT/PieBot/target/release/selfplay" \
  --games "$BENCH_GAMES" --depth 2 --max-plies 160 --threads 1 \
  --parallel-games "$EFFECTIVE" --seed 777 \
  --openings "$BOOK" \
  --nnue-quant-file "$MODEL" --nnue-blend-percent 25 \
  --jsonl-out "$WORK/shard" --skip-bin >/dev/null 2>&1
SP_SECONDS=$(python3 -c "import time; print(f'{time.time() - $SP_START:.2f}')")
SP_ROWS=$(wc -l < "$WORK/shard/"*.jsonl | tail -1 | tr -d ' ')
echo "  ${SP_SECONDS}s for $SP_ROWS positions"

RELABEL_JSON="{}"
for T in $THREADS_LIST; do
  [ "$T" = max ] && T="$EFFECTIVE"
  echo "[bench] 3/3 relabel depth 5, threads=$T (1500 records)"
  RL_START=$(date +%s.%N)
  "$REPO_ROOT/PieBot/target/release/relabel_jsonl" \
    --input "$WORK/shard" --output "$WORK/relabel_$T" \
    --depth 5 --every 2 --threads "$T" --hash-mb 2048 --max-records 1500 \
    --nnue-quant-file "$MODEL" --nnue-blend-percent 25 >/dev/null 2>&1
  RL_SECONDS=$(python3 -c "import time; print(f'{time.time() - $RL_START:.2f}')")
  RATE=$(python3 -c "print(f'{1500 / $RL_SECONDS:.1f}')")
  echo "  threads=$T: ${RL_SECONDS}s (${RATE} positions/s)"
  RELABEL_JSON=$(python3 -c "
import json
d = json.loads('$RELABEL_JSON')
d['threads_$T'] = {'seconds': $RL_SECONDS, 'positions_per_second': $RATE}
print(json.dumps(d))")
done

python3 - "$OUT_JSON" <<PY
import json, platform, sys
result = {
    "schema": "piebot-cpu-benchmark-v1",
    "cpu_model": """$CPU_MODEL""".strip(),
    "effective_cores": int("$EFFECTIVE"),
    "platform": platform.platform(),
    "single_thread": {"matein3_nps": float("$ST_NPS"), "nodes": int("$ST_NODES")},
    "selfplay": {"games": int("$BENCH_GAMES"), "seconds": float("$SP_SECONDS"),
                 "positions": int("$SP_ROWS")},
    "relabel_depth5": json.loads('$RELABEL_JSON'),
    "seeds": {"selfplay": 777},
}
with open(sys.argv[1], "w") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
PY
echo "[bench] wrote $OUT_JSON"
