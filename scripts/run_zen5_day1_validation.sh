#!/usr/bin/env bash
set -euo pipefail

# 1-day Zen5+RTX3090 validation run.
# Purpose: verify end-to-end pipeline reliability and produce a first trained NNUE.
# Note: this is not a direct Elo calibration against a fixed external engine pool.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT_DEFAULT="/opt/piebot_runs/day1_validation_$(date +%Y%m%d_%H%M%S)"

OUT_ROOT="${OUT_ROOT:-$OUT_ROOT_DEFAULT}"
HOURS="${HOURS:-24}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-8000}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-2}"
SELFPLAY_THREADS="${SELFPLAY_THREADS:-1}"
SELFPLAY_PARALLEL_GAMES="${SELFPLAY_PARALLEL_GAMES:-128}"
RELABEL_DEPTH="${RELABEL_DEPTH:-9}"
RELABEL_EVERY="${RELABEL_EVERY:-8}"
RELABEL_THREADS="${RELABEL_THREADS:-48}"
RELABEL_HASH_MB="${RELABEL_HASH_MB:-4096}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-350000}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
RETAIN_FULL_CYCLES="${RETAIN_FULL_CYCLES:-8}"
TRAINER_BACKEND="${TRAINER_BACKEND:-torch}"
TRAINER_DEVICE="${TRAINER_DEVICE:-cuda}"

COMPARE_GAMES="${COMPARE_GAMES:-24}"
COMPARE_MOVETIME_MS="${COMPARE_MOVETIME_MS:-150}"
COMPARE_THREADS="${COMPARE_THREADS:-1}"

log() {
  printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd python3
require_cmd cargo
require_cmd jq

if [[ "$TRAINER_DEVICE" == "cuda" && "$TRAINER_BACKEND" == "auto" ]]; then
  log "CUDA requested; resolving trainer backend auto to torch (no stub fallback)"
  TRAINER_BACKEND="torch"
fi

if [[ "$TRAINER_DEVICE" == "cuda" ]]; then
  require_cmd nvidia-smi
fi

log "repo root: $ROOT_DIR"
log "output root: $OUT_ROOT"
if ! mkdir -p "$OUT_ROOT"; then
  echo "cannot create $OUT_ROOT; create /opt/piebot_runs and grant this user ownership first" >&2
  exit 1
fi
if [[ ! -w "$OUT_ROOT" ]]; then
  echo "output root is not writable: $OUT_ROOT" >&2
  exit 1
fi
cd "$ROOT_DIR"

log "checking trainer backend/device availability"
python3 - "$TRAINER_BACKEND" "$TRAINER_DEVICE" <<'PY'
import sys

backend, device = sys.argv[1:]
if backend == "stub":
    if device == "cuda":
        raise SystemExit("CUDA requested but trainer backend is explicitly stub")
    print("trainer_backend", backend)
    print("trainer_device", device)
    raise SystemExit(0)

try:
    import torch
except Exception as exc:
    if backend == "torch" or device == "cuda":
        raise SystemExit(f"PyTorch is required for backend={backend}, device={device}: {exc}")
    print(f"PyTorch unavailable; backend=auto will use the CPU stub: {exc}", file=sys.stderr)
    raise SystemExit(0)

cuda_available = bool(torch.cuda.is_available())
print("torch_version", torch.__version__)
print("cuda_available", cuda_available)
print("cuda_devices", torch.cuda.device_count())
if device == "cuda" and not cuda_available:
    raise SystemExit("CUDA requested but torch CUDA is unavailable; refusing CPU stub fallback")
PY

log "building required binaries"
cargo build --locked --release --manifest-path "$ROOT_DIR/PieBot/Cargo.toml" \
  --bin selfplay --bin relabel_jsonl --bin compare_play \
  --bin accept --bin accept_temp

log "starting 24h autopilot run"
python3 -m training.nnue.autopilot \
  --out-root "$OUT_ROOT" \
  --hours "$HOURS" \
  --profile zen5_9755_7d \
  --selfplay-games "$SELFPLAY_GAMES" \
  --selfplay-depth "$SELFPLAY_DEPTH" \
  --selfplay-threads "$SELFPLAY_THREADS" \
  --selfplay-parallel-games "$SELFPLAY_PARALLEL_GAMES" \
  --teacher-relabel-depth "$RELABEL_DEPTH" \
  --teacher-relabel-every "$RELABEL_EVERY" \
  --teacher-relabel-threads "$RELABEL_THREADS" \
  --teacher-relabel-hash-mb "$RELABEL_HASH_MB" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --hidden-dim "$HIDDEN_DIM" \
  --retain-full-cycles "$RETAIN_FULL_CYCLES" \
  --trainer-backend "$TRAINER_BACKEND" \
  --trainer-device "$TRAINER_DEVICE"

STATE_JSON="$OUT_ROOT/autopilot_state.json"
if [[ ! -f "$STATE_JSON" ]]; then
  echo "missing state file: $STATE_JSON" >&2
  exit 2
fi

COMPLETED="$(jq -r '.completed_cycles | length' "$STATE_JSON")"
if [[ "$COMPLETED" -lt 1 ]]; then
  echo "no completed cycles in $STATE_JSON" >&2
  exit 3
fi

LAST_CYCLE="$(printf 'cycle_%06d' "$COMPLETED")"
CYCLE_DIR="$OUT_ROOT/cycles/$LAST_CYCLE"
SUMMARY_JSON="$CYCLE_DIR/pipeline_summary.json"
NNUE_QUANT="$CYCLE_DIR/nnue_quant.nnue"

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "missing cycle summary: $SUMMARY_JSON" >&2
  exit 4
fi
if [[ ! -f "$NNUE_QUANT" ]]; then
  echo "missing quantized NNUE artifact: $NNUE_QUANT" >&2
  exit 5
fi

log "running acceptance sanity suite (matein3 depth7)"
PIEBOT_SUITE_FILE="$ROOT_DIR/PieBot/src/suites/matein3.txt" \
PIEBOT_TEST_THREADS=1 \
PIEBOT_TEST_START_DEPTH=7 \
PIEBOT_TEST_MAX_DEPTH=7 \
cargo run --locked --release --manifest-path "$ROOT_DIR/PieBot/Cargo.toml" --bin accept

log "running experimental-search acceptance sanity suite (matein3 depth7)"
PIEBOT_SUITE_FILE="$ROOT_DIR/PieBot/src/suites/matein3.txt" \
PIEBOT_TEST_THREADS=1 \
PIEBOT_TEST_START_DEPTH=7 \
PIEBOT_TEST_MAX_DEPTH=7 \
cargo run --locked --release --manifest-path "$ROOT_DIR/PieBot/Cargo.toml" --bin accept_temp

log "comparing trained NNUE against the PST baseline with identical search"
cargo run --locked --release --manifest-path "$ROOT_DIR/PieBot/Cargo.toml" --bin compare_play -- \
  --same-search \
  --games "$COMPARE_GAMES" \
  --movetime "$COMPARE_MOVETIME_MS" \
  --noise-plies 12 \
  --noise-topk 5 \
  --threads "$COMPARE_THREADS" \
  --base-eval pst \
  --base-use-nnue false \
  --base-blend 0 \
  --exp-eval nnue \
  --exp-use-nnue true \
  --exp-blend 100 \
  --exp-nnue-quant-file "$NNUE_QUANT" \
  --json-out "$OUT_ROOT/post_compare.json" \
  --csv-out "$OUT_ROOT/post_compare.csv" \
  --pgn-out "$OUT_ROOT/post_compare.pgn"

log "validation complete"
log "state file: $STATE_JSON"
log "last cycle summary: $SUMMARY_JSON"
log "trained NNUE: $NNUE_QUANT"
log "compare output: $OUT_ROOT/post_compare.json"
echo
echo "NOTE: this confirms pipeline correctness and basic playing sanity."
echo "It does not prove a calibrated Elo target without an external rating match framework."
