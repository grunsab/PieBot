#!/usr/bin/env bash
set -Eeuo pipefail

# campaign_v2: the long super-GM training lineage (CampaignPlan_SuperGM_v1 s4).
# Deadline is minted ONCE at root creation and covers the campaign plus the
# 15-day reserve. Gate knobs (including SPRT) are frozen here for the
# lineage's entire life: this launcher is never edited mid-lineage.

REPO_ROOT="${REPO_ROOT:-/workspace/piebot_rust}"
PIEBOT_DIR="${PIEBOT_DIR:-$REPO_ROOT/PieBot}"
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
OUT_ROOT="${OUT_ROOT:-/workspace/piebot_campaign_v2}"
BOOTSTRAP_DIR="$OUT_ROOT/bootstrap"

PRIOR_RUN_ROOT="/workspace/piebot_runs/main_72h_self_teacher_repair_v1"
# Weights-only dense bootstrap: the protected cycle-86 artifact (the accepted
# cycle-98 DENSE checkpoint was retention-pruned; quants are not checkpoints).
INITIAL_CHECKPOINT_SOURCE="${INITIAL_CHECKPOINT_SOURCE:-$PRIOR_RUN_ROOT/bootstrap/cycle_000086_checkpoint.json}"
INITIAL_CHECKPOINT="$BOOTSTRAP_DIR/cycle_000086_checkpoint.json"
INITIAL_CHECKPOINT_SHA256="${INITIAL_CHECKPOINT_SHA256:-0ce48cc1299d5750bd43512793e843d8363e1e09a5c4a72c3b22e024951f367c}"
# Actor/teacher incumbent: the last gameplay-accepted quant (cycle 98).
INITIAL_ACTIVE_MODEL_SOURCE="${INITIAL_ACTIVE_MODEL_SOURCE:-$PRIOR_RUN_ROOT/cycles/cycle_000098/nnue_quant.nnue}"
INITIAL_ACTIVE_MODEL="$BOOTSTRAP_DIR/cycle_000098_nnue_quant.nnue"
INITIAL_ACTIVE_MODEL_SHA256="${INITIAL_ACTIVE_MODEL_SHA256:-3fa9bae3127319930ec16ebb1ee3117656abe7001984f6c8655108a08d278c3a}"
# Incumbent blend comes from the Phase 0 ladder measurement, not docs.
INITIAL_ACTIVE_MODEL_BLEND_PERCENT="${INITIAL_ACTIVE_MODEL_BLEND_PERCENT:-25}"

VALIDATION_SHARD_SOURCE="${VALIDATION_SHARD_SOURCE:-$PRIOR_RUN_ROOT/bootstrap/validation/shard_000000.jsonl}"
VALIDATION_JSONL_DIR="$BOOTSTRAP_DIR/validation"
VALIDATION_SHARD="$VALIDATION_JSONL_DIR/shard_000000.jsonl"
VALIDATION_SHARD_SHA256="${VALIDATION_SHARD_SHA256:-d6f4a72a356bb516f62f76488b89c4c70519acca93c625f55709e958485bc8d8}"
VALIDATION_PROVENANCE_SOURCE="$REPO_ROOT/deploy/vast/piebot_fixed_validation_provenance.json"
VALIDATION_PROVENANCE="$BOOTSTRAP_DIR/piebot_fixed_validation_provenance.json"
VALIDATION_PROVENANCE_SHA256="d7c0cf91b113b45ba8cabfe6e891f6643e4973b8c97010b7632baf61b86aab1e"

# Curated opening suite (positions only; staged by content hash).
OPENINGS_SOURCE="${OPENINGS_SOURCE:-$REPO_ROOT/books/openings_v1.fen}"
SELFPLAY_OPENINGS="$BOOTSTRAP_DIR/openings_v1.fen"
OPENINGS_SHA256="${OPENINGS_SHA256:-d35b81a1a75d03d6172c40f94c9e8626e3f3b6ed8995f935f5bce1e1c5550294}"

HOURS="${HOURS:-1440}"
SELFPLAY_GAMES="${SELFPLAY_GAMES:-8000}"
SELFPLAY_DEPTH="${SELFPLAY_DEPTH:-2}"
SELFPLAY_THREADS="${SELFPLAY_THREADS:-1}"
# Slot partition: 32 autopilot lane + reserved arena/A-B lanes (plan s4).
SELFPLAY_PARALLEL_GAMES="${SELFPLAY_PARALLEL_GAMES:-32}"
RELABEL_DEPTH="${RELABEL_DEPTH:-7}"
RELABEL_EVERY="${RELABEL_EVERY:-2}"
RELABEL_THREADS="${RELABEL_THREADS:-32}"
RELABEL_HASH_MB="${RELABEL_HASH_MB:-4096}"
# Node cap sized from the measured p95 of depth-5 node cost; no default so an
# unmeasured launch refuses to start.
RELABEL_MAX_NODES="${RELABEL_MAX_NODES:-}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_SAMPLES="${MAX_SAMPLES:-700000}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.002}"
WARM_START_LEARNING_RATE="${WARM_START_LEARNING_RATE:-0.001}"
RETAIN_FULL_CYCLES="${RETAIN_FULL_CYCLES:-8}"
REPLAY_WINDOW_CYCLES="${REPLAY_WINDOW_CYCLES:-6}"
GATE_GAMES="${GATE_GAMES:-24}"
GATE_SEARCH_THREADS="${GATE_SEARCH_THREADS:-1}"
GATE_PARALLEL_GAMES="${GATE_PARALLEL_GAMES:-12}"
# Adjudication (plan WP5 Pilot B): resign 900cp x 8 plies with a 15%
# no-resign fraction; draw-adjudicate |10cp| x 40 plies past ply 80.
RESIGN_CP="${RESIGN_CP:-900}"
RESIGN_PLIES="${RESIGN_PLIES:-8}"
NO_RESIGN_FRACTION="${NO_RESIGN_FRACTION:-0.15}"
DRAW_ADJ_CP="${DRAW_ADJ_CP:-10}"
DRAW_ADJ_PLIES="${DRAW_ADJ_PLIES:-40}"
DRAW_ADJ_MIN_PLY="${DRAW_ADJ_MIN_PLY:-80}"

export PATH="/root/.cargo/bin:/venv/main/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer, got: $value"
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got: $value"
}

require_positive_number() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "$name must be numeric, got: $value"
  "$PYTHON_BIN" - "$name" "$value" <<'PY'
import sys

name, raw = sys.argv[1:]
if float(raw) <= 0.0:
    raise SystemExit(f"{name} must be greater than zero, got: {raw}")
PY
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  "$PYTHON_BIN" - "$path" "$expected" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2].lower()
digest = hashlib.sha256()
with path.open("rb") as handle:
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(
        f"artifact SHA-256 mismatch: expected {expected}, got {actual}"
    )
print("verified_artifact_sha256", actual)
PY
}

stage_verified_file() {
  local source="$1"
  local destination="$2"
  local expected_sha256="$3"
  if [[ -f "$destination" ]]; then
    verify_sha256 "$destination" "$expected_sha256"
    return
  fi
  require_file "$source"
  verify_sha256 "$source" "$expected_sha256"
  local temporary="$destination.tmp.$$"
  cp -- "$source" "$temporary"
  verify_sha256 "$temporary" "$expected_sha256"
  mv -f -- "$temporary" "$destination"
  verify_sha256 "$destination" "$expected_sha256"
}

require_autopilot_flag() {
  local required_flag="$1"
  local help_text
  help_text="$("$PYTHON_BIN" -m training.nnue.autopilot --help)"
  [[ "$help_text" == *"$required_flag"* ]] \
    || die "autopilot does not support required safety flag: $required_flag"
}

effective_cpu_count() {
  "$PYTHON_BIN" <<'PY'
import math
import os
from pathlib import Path

counts = [os.cpu_count() or 1]
if hasattr(os, "sched_getaffinity"):
    counts.append(len(os.sched_getaffinity(0)))
cpu_max = Path("/sys/fs/cgroup/cpu.max")
if cpu_max.is_file():
    quota_raw, period_raw = cpu_max.read_text(encoding="utf-8").split()[:2]
    if quota_raw != "max":
        counts.append(max(1, math.floor(int(quota_raw) / int(period_raw))))
print(min(counts))
PY
}

require_cmd cargo
require_cmd cp
require_cmd git
require_cmd nvidia-smi
require_file "$PYTHON_BIN"
require_dir "$REPO_ROOT"
require_file "$PIEBOT_DIR/Cargo.toml"

require_positive_number HOURS "$HOURS"
require_positive_int SELFPLAY_GAMES "$SELFPLAY_GAMES"
require_positive_int SELFPLAY_DEPTH "$SELFPLAY_DEPTH"
require_positive_int SELFPLAY_THREADS "$SELFPLAY_THREADS"
require_positive_int SELFPLAY_PARALLEL_GAMES "$SELFPLAY_PARALLEL_GAMES"
require_positive_int RELABEL_DEPTH "$RELABEL_DEPTH"
require_positive_int RELABEL_EVERY "$RELABEL_EVERY"
require_positive_int RELABEL_THREADS "$RELABEL_THREADS"
require_positive_int RELABEL_HASH_MB "$RELABEL_HASH_MB"
[[ -n "$RELABEL_MAX_NODES" ]] \
  || die "RELABEL_MAX_NODES must be set to the measured p95 depth-5 node cost (battery step 5); refusing an unmeasured depth-7 teacher"
require_positive_int RELABEL_MAX_NODES "$RELABEL_MAX_NODES"
require_positive_int EPOCHS "$EPOCHS"
require_positive_int BATCH_SIZE "$BATCH_SIZE"
require_positive_int MAX_SAMPLES "$MAX_SAMPLES"
require_positive_int HIDDEN_DIM "$HIDDEN_DIM"
require_positive_number LEARNING_RATE "$LEARNING_RATE"
require_positive_number WARM_START_LEARNING_RATE "$WARM_START_LEARNING_RATE"
require_nonnegative_int RETAIN_FULL_CYCLES "$RETAIN_FULL_CYCLES"
require_nonnegative_int REPLAY_WINDOW_CYCLES "$REPLAY_WINDOW_CYCLES"
require_positive_int GATE_GAMES "$GATE_GAMES"
require_positive_int GATE_SEARCH_THREADS "$GATE_SEARCH_THREADS"
require_positive_int GATE_PARALLEL_GAMES "$GATE_PARALLEL_GAMES"
require_nonnegative_int INITIAL_ACTIVE_MODEL_BLEND_PERCENT "$INITIAL_ACTIVE_MODEL_BLEND_PERCENT"

[[ "$RELABEL_DEPTH" -eq 7 ]] || die "this deployment is pinned to the node-capped PieBot depth-7 teacher"
(( REPLAY_WINDOW_CYCLES <= RETAIN_FULL_CYCLES )) \
  || die "REPLAY_WINDOW_CYCLES must not exceed RETAIN_FULL_CYCLES: replay silently shrinks when retention deletes cycles"
(( RETAIN_FULL_CYCLES >= 1 )) \
  || die "RETAIN_FULL_CYCLES=0 disables all cleanup and will fill the disk"
(( GATE_PARALLEL_GAMES == 1 || GATE_SEARCH_THREADS == 1 )) \
  || die "parallel promotion matches require GATE_SEARCH_THREADS=1"
(( INITIAL_ACTIVE_MODEL_BLEND_PERCENT <= 100 )) \
  || die "INITIAL_ACTIVE_MODEL_BLEND_PERCENT must be between 0 and 100"
[[ "$INITIAL_ACTIVE_MODEL_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "INITIAL_ACTIVE_MODEL_SHA256 must contain exactly 64 hexadecimal characters"
[[ "$INITIAL_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "INITIAL_CHECKPOINT_SHA256 must contain exactly 64 hexadecimal characters"
[[ "$VALIDATION_SHARD_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "VALIDATION_SHARD_SHA256 must contain exactly 64 hexadecimal characters"
[[ "$OPENINGS_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "OPENINGS_SHA256 must contain exactly 64 hexadecimal characters"
[[ "$OUT_ROOT" != "/workspace/piebot_runs/main_48h_20260802T081500Z" ]] \
  || die "OUT_ROOT must not overwrite a previous training run"
[[ "$OUT_ROOT" != "/workspace/piebot_runs/main_72h_self_teacher_repair_v1" ]] \
  || die "OUT_ROOT must not overwrite a previous training run"

cd "$REPO_ROOT"
[[ -z "$(git status --porcelain)" ]] || die "repository worktree is not clean"
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
[[ "$SOURCE_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "cannot resolve source Git commit"

mkdir -p "$OUT_ROOT"
[[ -w "$OUT_ROOT" ]] || die "output root is not writable: $OUT_ROOT"
mkdir -p "$BOOTSTRAP_DIR" "$VALIDATION_JSONL_DIR"
SOURCE_COMMIT_FILE="$OUT_ROOT/source_git_commit"
if [[ -f "$SOURCE_COMMIT_FILE" ]]; then
  read -r PINNED_SOURCE_GIT_COMMIT < "$SOURCE_COMMIT_FILE"
  [[ "$PINNED_SOURCE_GIT_COMMIT" == "$SOURCE_GIT_COMMIT" ]] \
    || die "refusing source commit change: pinned $PINNED_SOURCE_GIT_COMMIT, current $SOURCE_GIT_COMMIT"
elif [[ -f "$OUT_ROOT/autopilot_state.json" ]]; then
  die "existing training state has no pinned source_git_commit"
else
  SOURCE_COMMIT_TMP="$SOURCE_COMMIT_FILE.tmp.$$"
  printf '%s\n' "$SOURCE_GIT_COMMIT" > "$SOURCE_COMMIT_TMP"
  mv -f -- "$SOURCE_COMMIT_TMP" "$SOURCE_COMMIT_FILE"
fi

stage_verified_file "$INITIAL_CHECKPOINT_SOURCE" "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"
stage_verified_file "$INITIAL_ACTIVE_MODEL_SOURCE" "$INITIAL_ACTIVE_MODEL" "$INITIAL_ACTIVE_MODEL_SHA256"
stage_verified_file "$VALIDATION_SHARD_SOURCE" "$VALIDATION_SHARD" "$VALIDATION_SHARD_SHA256"
stage_verified_file "$VALIDATION_PROVENANCE_SOURCE" "$VALIDATION_PROVENANCE" "$VALIDATION_PROVENANCE_SHA256"
stage_verified_file "$OPENINGS_SOURCE" "$SELFPLAY_OPENINGS" "$OPENINGS_SHA256"

require_autopilot_flag "--initial-checkpoint-weights-only"
require_autopilot_flag "--initial-active-model"
require_autopilot_flag "--initial-active-model-blend-percent"
require_autopilot_flag "--gate-parallel-games"
require_autopilot_flag "--gate-incremental-pst-policy"
require_autopilot_flag "--gate-pst-veto-margin"
require_autopilot_flag "--selfplay-openings"
require_autopilot_flag "--teacher-relabel-max-nodes"
require_autopilot_flag "--gate-sprt"
require_autopilot_flag "--selfplay-resign-cp"
verify_sha256 "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"
verify_sha256 "$INITIAL_ACTIVE_MODEL" "$INITIAL_ACTIVE_MODEL_SHA256"
verify_sha256 "$SELFPLAY_OPENINGS" "$OPENINGS_SHA256"

EFFECTIVE_CPUS="$(effective_cpu_count)"
SELFPLAY_CPU_SLOTS=$((SELFPLAY_THREADS * SELFPLAY_PARALLEL_GAMES))
GATE_CPU_SLOTS=$((GATE_SEARCH_THREADS * GATE_PARALLEL_GAMES))
REQUIRED_CPUS="$SELFPLAY_CPU_SLOTS"
if (( RELABEL_THREADS > REQUIRED_CPUS )); then
  REQUIRED_CPUS="$RELABEL_THREADS"
fi
if (( GATE_CPU_SLOTS > REQUIRED_CPUS )); then
  REQUIRED_CPUS="$GATE_CPU_SLOTS"
fi
(( EFFECTIVE_CPUS >= REQUIRED_CPUS )) \
  || die "only $EFFECTIVE_CPUS effective CPUs are available; $REQUIRED_CPUS are configured"

log "hardware preflight"
log "source Git commit: $SOURCE_GIT_COMMIT"
log "effective CPUs: $EFFECTIVE_CPUS (selfplay slots: $SELFPLAY_CPU_SLOTS; relabel slots: $RELABEL_THREADS; gate slots: $GATE_CPU_SLOTS)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" - <<'PY'
import shutil
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
props = torch.cuda.get_device_properties(0)
available = shutil.disk_usage(Path("/workspace")).free
print("torch_version", torch.__version__)
print("cuda_device", props.name)
print("cuda_memory_bytes", props.total_memory)
print("training_disk_free_bytes", available)
if props.total_memory < 24 * 1024**3:
    raise SystemExit("GPU memory is below the 24 GiB production minimum")
if available < 30 * 1024**3:
    raise SystemExit("training disk has less than 30 GiB free")
PY

log "building optimized production binaries"
cargo build --locked --release --manifest-path "$PIEBOT_DIR/Cargo.toml" \
  --bin selfplay --bin relabel_jsonl --bin compare_play

AUTOPILOT_ARGS=(
  "--out-root" "$OUT_ROOT"
  "--piebot-dir" "$PIEBOT_DIR"
  "--hours" "$HOURS"
  "--profile" "zen5_9755_7d"
  "--retry-limit" "5"
  "--retry-backoff-sec" "30"
  "--selfplay-games" "$SELFPLAY_GAMES"
  "--selfplay-depth" "$SELFPLAY_DEPTH"
  "--selfplay-threads" "$SELFPLAY_THREADS"
  "--selfplay-parallel-games" "$SELFPLAY_PARALLEL_GAMES"
  "--selfplay-openings" "$SELFPLAY_OPENINGS"
  "--selfplay-resign-cp" "$RESIGN_CP"
  "--selfplay-resign-plies" "$RESIGN_PLIES"
  "--selfplay-no-resign-fraction" "$NO_RESIGN_FRACTION"
  "--selfplay-draw-adj-cp" "$DRAW_ADJ_CP"
  "--selfplay-draw-adj-plies" "$DRAW_ADJ_PLIES"
  "--selfplay-draw-adj-min-ply" "$DRAW_ADJ_MIN_PLY"
  "--teacher-relabel-depth" "$RELABEL_DEPTH"
  "--teacher-relabel-every" "$RELABEL_EVERY"
  "--teacher-relabel-threads" "$RELABEL_THREADS"
  "--teacher-relabel-hash-mb" "$RELABEL_HASH_MB"
  "--teacher-relabel-max-nodes" "$RELABEL_MAX_NODES"
  "--teacher-sample-fraction" "0.5"
  "--min-teacher-depth" "5"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--max-samples" "$MAX_SAMPLES"
  "--hidden-dim" "$HIDDEN_DIM"
  "--learning-rate" "$LEARNING_RATE"
  "--warm-start-learning-rate" "$WARM_START_LEARNING_RATE"
  "--warm-start"
  "--initial-checkpoint" "$INITIAL_CHECKPOINT"
  "--initial-checkpoint-weights-only"
  "--initial-active-model" "$INITIAL_ACTIVE_MODEL"
  "--initial-active-model-blend-percent" "$INITIAL_ACTIVE_MODEL_BLEND_PERCENT"
  "--continue-optimizer-state"
  "--validation-jsonl-dir" "$VALIDATION_JSONL_DIR"
  "--validation-provenance-json" "$VALIDATION_PROVENANCE"
  "--validation-require-teacher"
  "--retain-full-cycles" "$RETAIN_FULL_CYCLES"
  "--replay-window-cycles" "$REPLAY_WINDOW_CYCLES"
  "--teacher-lag-cycles" "0"
  "--gate-games" "$GATE_GAMES"
  "--gate-threads" "$GATE_SEARCH_THREADS"
  "--gate-parallel-games" "$GATE_PARALLEL_GAMES"
  "--gate-min-score-delta" "0.0"
  "--gate-incremental-pst-policy" "regression-veto"
  "--gate-pst-veto-margin" "0.0"
  "--gate-paired-openings"
  "--gate-sprt"
  "--gate-sprt-delta1" "0.25"
  "--gate-sprt-alpha" "0.05"
  "--gate-sprt-beta" "0.05"
  "--gate-sprt-min-pairs" "48"
  "--gate-sprt-batch-pairs" "24"
  "--gate-sprt-max-pairs" "300"
  "--trainer-backend" "torch"
  "--trainer-device" "cuda"
)

log "starting campaign_v2: PieBot-only node-capped depth-$RELABEL_DEPTH self-relabel training"
log "output root: $OUT_ROOT"
log "deadline budget: $HOURS hours (persisted once in autopilot_state.json)"
log "opening suite: $SELFPLAY_OPENINGS (sha256 $OPENINGS_SHA256)"
log "teacher node cap: $RELABEL_MAX_NODES nodes/position (measured p95 of depth-5 cost)"
"$PYTHON_BIN" -m training.nnue.autopilot "${AUTOPILOT_ARGS[@]}"

require_file "$OUT_ROOT/autopilot_state.json"
log "campaign deadline reached cleanly; state: $OUT_ROOT/autopilot_state.json"
