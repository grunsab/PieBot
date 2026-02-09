#!/usr/bin/env bash
set -euo pipefail

# 7-day Zen5+RTX3090 production autopilot run.
# Uses game-level parallel self-play by default for high-core throughput.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT_DEFAULT="/opt/piebot_runs/zen5_7d_$(date +%Y%m%d_%H%M%S)"

OUT_ROOT="${OUT_ROOT:-$OUT_ROOT_DEFAULT}"
HOURS="${HOURS:-168}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-12000}"
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
TRAINER_BACKEND="${TRAINER_BACKEND:-auto}"
TRAINER_DEVICE="${TRAINER_DEVICE:-cuda}"

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

if [[ "$TRAINER_DEVICE" == "cuda" ]]; then
  require_cmd nvidia-smi
fi

log "repo root: $ROOT_DIR"
log "output root: $OUT_ROOT"
mkdir -p "$OUT_ROOT"

if [[ "$TRAINER_BACKEND" == "torch" || "$TRAINER_BACKEND" == "auto" ]]; then
  log "checking torch/cuda visibility"
  python3 - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}", file=sys.stderr)
    raise
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_devices", torch.cuda.device_count())
PY
fi

log "building required binaries"
cargo build --release --manifest-path "$ROOT_DIR/PieBot/Cargo.toml" \
  --bin selfplay --bin relabel_jsonl --bin compare_play

log "starting 7-day autopilot run"
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

ACTIVE_MODEL="$(jq -r '.active_model_path // empty' "$STATE_JSON")"
LAST_CYCLE="$(printf 'cycle_%06d' "$COMPLETED")"
SUMMARY_JSON="$OUT_ROOT/cycles/$LAST_CYCLE/pipeline_summary.json"

log "run complete"
log "completed cycles: $COMPLETED"
log "active model: ${ACTIVE_MODEL:-<none>}"
log "state file: $STATE_JSON"
log "last summary: $SUMMARY_JSON"

