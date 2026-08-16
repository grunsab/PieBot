#!/usr/bin/env bash
set -Eeuo pipefail

# campaign_v2: the long super-GM training lineage (CampaignPlan_SuperGM_v1 s4).
# Deadline is minted ONCE at root creation and covers the campaign plus the
# 15-day reserve. Gate knobs (including SPRT) are frozen here for the
# lineage's entire life: this launcher is never edited mid-lineage.
#
# ONE SANCTIONED EXCEPTION (2026-08-14, campaign_v7). The freeze protects the
# lineage from having its promotion bar moved to manufacture an acceptance. It
# is not a licence to keep a measuring instrument that cannot see the quantity
# it measures. The gate shipped with delta1 = 0.25 -- an H1 of +43.7 Elo and an
# indifference point of +21.8 Elo -- while the trainer produces ~+0.65 Elo per
# cycle. Over cycles 30-68 that rejected every candidate, and a 1000-game paired
# match then measured the "rejected" cycle-68 net at +25.4 Elo [+11.3, +39.6]
# over the frozen active model. The gate was wrong, not the nets.
# See evidence/gate_power_and_unpromoted_progress_20260814.json.
# delta1 and max-pairs MUST move together: an inconclusive SPRT is recorded as
# a REJECT, so a tighter H1 with an unchanged cap is an unconditional reject.
#
# EXCEPTION 2 (2026-08-15, campaign_v7): SCHEDULING ONLY. batch-pairs 24 -> 180.
# alpha, beta, delta1, min-pairs, max-pairs, screen size and screen threshold are
# ALL UNCHANGED -- the promotion bar is not touched, only how the same games are
# dispatched.
#
# compare_play bounds match workers at min(parallel_games, cores, work_units)
# with work_units = games/2 (compare_play.rs:600-620), so the BATCH SIZE caps
# the worker count. A 24-pair batch pinned the gate to 24 of 184 cores no
# matter what GATE_PARALLEL_GAMES said -- the conf's 48 was already inert.
#
# Measured throughput, same net, same box, 150 ms movetime, CPU lane free:
#     24 pairs / 24 workers   48 games   48.2 s   0.996 games/s
#     90 pairs / 45 workers  180 games   90.8 s   1.981 games/s
#    180 pairs / 90 workers  360 games   95.7 s   3.762 games/s   <- 3.78x
# Per-worker throughput is flat (0.0415 / 0.0440 / 0.0418), i.e. aggregate
# throughput scales with WORKER COUNT, and the batch size is what makes workers
# reachable. See evidence/gate_sprt_work_granularity_20260815.json.
#
# 90 is the largest worker count actually measured; it is not proven optimal.
# For a fixed batch, wall-clock is non-increasing in workers, so 180 pairs on
# ~180 workers may well be faster still -- but that is unmeasured, sits at 1.9x
# SMT oversubscription on 96 physical cores, and would push peak gate RSS to
# roughly 180 x 296 MB ~ 53 GB. Measure before raising it.
#
# Statistical effect of the coarser look: the first SPRT look moves from 48 to
# 180 pairs. No observed confirmation has ever resolved that early -- c122 took
# 480 pairs, c118 600, c120 1128 -- so early stopping is not surrendered. Fewer
# looks at a Wald boundary is the CONSERVATIVE direction (error rates stay at or
# below nominal); the cost is extra games past the decision point, and those run
# on cores that were idle anyway.

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

# FRESH_INIT=1 starts a new-width lineage from fresh random weights: no
# checkpoint is staged and the warm-start source flags are omitted, so
# train_torch initializes at --hidden-dim. The active model (actor/teacher/
# gate incumbent) is still staged and verified. An empty
# INITIAL_CHECKPOINT_SOURCE cannot express this because :- defaults swallow
# empty strings.
FRESH_INIT="${FRESH_INIT:-0}"

# Training architecture: v1 = merged-perspective ReLU (PIENNQ01), v2 =
# dual-perspective SCReLU per the standard NNUE design (PIENNQ02). v2
# requires a fresh lineage (FRESH_INIT=1) because the feature set changes.
TRAIN_ARCH="${TRAIN_ARCH:-v1}"

# Teacher/actor label separation. Self-play stamps every row with
# teacher_depth = actor depth, so MIN_TEACHER_DEPTH must EXCEED the actor
# depth or actor self-labels masquerade as teacher labels (discovered
# 2026-08-08; in v4 this silently made the actor its own 0.8-mix teacher on
# non-relabeled rows). TEACHER_SAMPLE_FRACTION must match the relabel
# cadence: every-Nth-ply relabeling yields roughly 1/N teacher rows.
MIN_TEACHER_DEPTH="${MIN_TEACHER_DEPTH:-5}"
TEACHER_SAMPLE_FRACTION="${TEACHER_SAMPLE_FRACTION:-0.5}"

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
# Screen size is a statistical knob, so it lives here with the other frozen
# gate parameters rather than in the supervisor conf. 24 games (12 pairs) could
# not distinguish a real candidate from noise, so the screen leaked nearly every
# null cycle into the expensive confirmation while the threshold sat at 0.0.
GATE_GAMES="${GATE_GAMES:-96}"
GATE_SEARCH_THREADS="${GATE_SEARCH_THREADS:-1}"
GATE_PARALLEL_GAMES="${GATE_PARALLEL_GAMES:-12}"
# Confirmation batch size, in opening pairs. FROZEN and deliberately NOT
# env-overridable: it is a statistical knob (it sets when the SPRT boundary is
# consulted), so it lives here with the rest of the frozen gate parameters and
# not in the host-tunable supervisor conf. See EXCEPTION 2 in the header and
# evidence/gate_sprt_work_granularity_20260815.json.
GATE_SPRT_BATCH_PAIRS=180
# Adjudication (plan WP5 Pilot B): resign 900cp x 8 plies with a 15%
# no-resign fraction; draw-adjudicate |10cp| x 40 plies past ply 80.
RESIGN_CP="${RESIGN_CP:-900}"
RESIGN_PLIES="${RESIGN_PLIES:-8}"
NO_RESIGN_FRACTION="${NO_RESIGN_FRACTION:-0.15}"
DRAW_ADJ_CP="${DRAW_ADJ_CP:-10}"
DRAW_ADJ_PLIES="${DRAW_ADJ_PLIES:-40}"
DRAW_ADJ_MIN_PLY="${DRAW_ADJ_MIN_PLY:-80}"
# Actor budget (Pilot C): real TT + 4x node caps; relabel dominates cycle
# time so this raise is cheap, and it attacks the threefold repetition share.
ACTOR_TT_MB="${ACTOR_TT_MB:-128}"
POLICY_NODE_CAP="${POLICY_NODE_CAP:-40000}"
BESTMOVE_NODE_CAP="${BESTMOVE_NODE_CAP:-80000}"
# Tightened exploration window (cycle-18 tripwire response): noise plies
# halved so the stronger actor's choices stop being diluted into repetition.
TEMPERATURE_MOVES="${TEMPERATURE_MOVES:-12}"
# C8 external teacher (dormant when empty): a PieBot-lineage quant that
# teaches in place of the state-derived active model. Staged by content hash.
TEACHER_EXTERNAL_QUANT_FILE="${TEACHER_EXTERNAL_QUANT_FILE:-}"
TEACHER_EXTERNAL_QUANT_SHA256="${TEACHER_EXTERNAL_QUANT_SHA256:-}"
# P5: decisive-outcome target magnitude (objective-identity field; changing it
# requires a fresh out_root with a weights-only bootstrap).
TARGET_CP="${TARGET_CP:-100}"
# campaign_v8 objective term: weight on Huber over the wdl_scale-normalised cp
# error, added to the WDL/BCE loss. 0 reproduces the v7 objective exactly.
# Non-zero is a DIFFERENT target -- objective-identity field, so it requires a
# fresh OUT_ROOT with a weights-only bootstrap and fresh Adam.
# Why it exists: BCE through sigmoid(cp/400) keeps only 28% of its gradient at
# 1000 cp and 18% at 1200, while 28% of this lineage's labels sit above |600|.
# A depth-7 search still disagrees with its own net by 617 cp on average and on
# 57.5%% of best moves, yet that signal shrank 26.5%% through the WDL target.
# See evidence/objective_saturation_20260816.json.
CP_LOSS_WEIGHT="${CP_LOSS_WEIGHT:-0}"
# Teacher/outcome blend on rows that carry a teacher label (objective-identity
# field; a change requires a fresh out_root). At 0.8 even depth-9 relabeled
# rows were 20% game-outcome noise; campaign_v7 trains at 1.0 so teacher rows
# carry undiluted search evaluations. Rows WITHOUT a teacher label are
# unaffected - they always target the outcome alone.
TEACHER_MIX="${TEACHER_MIX:-0.8}"
# Blend percent a candidate is gated at when its architecture differs from
# the active model's. Empty keeps autopilot's default of restarting such a
# candidate at the ramp's first rung (25), which deadlocks a new
# architecture: the ramp only advances on an acceptance, so the candidate
# must win while contributing a quarter of the eval and paying all of its
# cost. Set to 100 for a lineage whose net is trained to stand alone.
CROSS_ARCH_GATE_BLEND_PERCENT="${CROSS_ARCH_GATE_BLEND_PERCENT:-}"

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
[[ "$CP_LOSS_WEIGHT" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "CP_LOSS_WEIGHT must be a non-negative number, got: $CP_LOSS_WEIGHT"
[[ "$TEACHER_MIX" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "TEACHER_MIX must be numeric, got: $TEACHER_MIX"
"$PYTHON_BIN" - "$TEACHER_MIX" <<'PY'
import sys

value = float(sys.argv[1])
if not 0.0 <= value <= 1.0:
    raise SystemExit(f"TEACHER_MIX must lie in [0, 1], got: {value}")
PY
require_nonnegative_int RETAIN_FULL_CYCLES "$RETAIN_FULL_CYCLES"
require_nonnegative_int REPLAY_WINDOW_CYCLES "$REPLAY_WINDOW_CYCLES"
require_positive_int GATE_GAMES "$GATE_GAMES"
require_positive_int GATE_SEARCH_THREADS "$GATE_SEARCH_THREADS"
require_positive_int GATE_PARALLEL_GAMES "$GATE_PARALLEL_GAMES"
require_nonnegative_int INITIAL_ACTIVE_MODEL_BLEND_PERCENT "$INITIAL_ACTIVE_MODEL_BLEND_PERCENT"

# Two calibrated teacher shapes: depth 7 capped at depth-5's p95 (144k), and
# depth 9 capped at depth-7's p95 (2.5M). Any other depth is unmeasured.
(( MIN_TEACHER_DEPTH > SELFPLAY_DEPTH )) \
  || die "MIN_TEACHER_DEPTH must exceed SELFPLAY_DEPTH: actor rows stamp teacher_depth = actor depth and would masquerade as teacher labels"
[[ "$RELABEL_DEPTH" -eq 7 || "$RELABEL_DEPTH" -eq 9 ]] \
  || die "this deployment supports only the measured node-capped PieBot depth-7 or depth-9 teacher"
(( REPLAY_WINDOW_CYCLES <= RETAIN_FULL_CYCLES )) \
  || die "REPLAY_WINDOW_CYCLES must not exceed RETAIN_FULL_CYCLES: replay silently shrinks when retention deletes cycles"
(( RETAIN_FULL_CYCLES >= 1 )) \
  || die "RETAIN_FULL_CYCLES=0 disables all cleanup and will fill the disk"
(( GATE_PARALLEL_GAMES == 1 || GATE_SEARCH_THREADS == 1 )) \
  || die "parallel promotion matches require GATE_SEARCH_THREADS=1"
# compare_play bounds match workers at min(parallel_games, cores, work_units)
# with work_units = games/2 (compare_play.rs:600-620). A batch carrying fewer
# pairs than GATE_PARALLEL_GAMES therefore silently runs FEWER workers than
# configured -- that is the original defect: 24-pair batches pinned the gate to
# 24 workers while the conf asked for 48. Warn, never die: this is a throughput
# preference, and a performance heuristic must not be able to halt a live
# lineage. EFFECTIVE_CPUS >= REQUIRED_CPUS below is the real safety check.
if (( GATE_SPRT_BATCH_PAIRS < GATE_PARALLEL_GAMES )); then
  log "WARNING: gate batch carries $GATE_SPRT_BATCH_PAIRS pairs but GATE_PARALLEL_GAMES=$GATE_PARALLEL_GAMES; compare_play will clamp to $GATE_SPRT_BATCH_PAIRS workers"
fi
(( INITIAL_ACTIVE_MODEL_BLEND_PERCENT <= 100 )) \
  || die "INITIAL_ACTIVE_MODEL_BLEND_PERCENT must be between 0 and 100"
[[ "$INITIAL_ACTIVE_MODEL_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
  || die "INITIAL_ACTIVE_MODEL_SHA256 must contain exactly 64 hexadecimal characters"
if [[ "$FRESH_INIT" != "1" ]]; then
  [[ "$INITIAL_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
    || die "INITIAL_CHECKPOINT_SHA256 must contain exactly 64 hexadecimal characters"
fi
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
fi

if [[ "$FRESH_INIT" != "1" ]]; then
  stage_verified_file "$INITIAL_CHECKPOINT_SOURCE" "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"
fi
stage_verified_file "$INITIAL_ACTIVE_MODEL_SOURCE" "$INITIAL_ACTIVE_MODEL" "$INITIAL_ACTIVE_MODEL_SHA256"
stage_verified_file "$VALIDATION_SHARD_SOURCE" "$VALIDATION_SHARD" "$VALIDATION_SHARD_SHA256"
stage_verified_file "$VALIDATION_PROVENANCE_SOURCE" "$VALIDATION_PROVENANCE" "$VALIDATION_PROVENANCE_SHA256"
stage_verified_file "$OPENINGS_SOURCE" "$SELFPLAY_OPENINGS" "$OPENINGS_SHA256"
if [[ -n "$TEACHER_EXTERNAL_QUANT_FILE" ]]; then
  [[ "$TEACHER_EXTERNAL_QUANT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] \
    || die "TEACHER_EXTERNAL_QUANT_SHA256 must be set with the external teacher"
  EXTERNAL_TEACHER_STAGED="$BOOTSTRAP_DIR/external_teacher.nnue"
  stage_verified_file "$TEACHER_EXTERNAL_QUANT_FILE" "$EXTERNAL_TEACHER_STAGED" "$TEACHER_EXTERNAL_QUANT_SHA256"
fi

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
require_autopilot_flag "--selfplay-actor-tt-mb"
require_autopilot_flag "--selfplay-temperature-moves"
require_autopilot_flag "--teacher-external-quant-file"
require_autopilot_flag "--target-cp"
require_autopilot_flag "--cp-loss-weight"
require_autopilot_flag "--teacher-mix"
if [[ -n "$CROSS_ARCH_GATE_BLEND_PERCENT" ]]; then
  require_autopilot_flag "--cross-arch-gate-blend-percent"
fi
require_autopilot_flag "--train-arch"
if [[ "$FRESH_INIT" != "1" ]]; then
  verify_sha256 "$INITIAL_CHECKPOINT" "$INITIAL_CHECKPOINT_SHA256"
fi
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
if props.total_memory < 20 * 1024**3:
    raise SystemExit("GPU memory is below the 20 GiB production minimum")
if available < 30 * 1024**3:
    raise SystemExit("training disk has less than 30 GiB free")
PY

log "building optimized production binaries"
cargo build --locked --release --manifest-path "$PIEBOT_DIR/Cargo.toml" \
  --bin selfplay --bin relabel_jsonl --bin compare_play

# The pin is written only now, after every preflight and the build have
# passed: a failed launch must never leave a poisoned root behind.
if [[ ! -f "$SOURCE_COMMIT_FILE" ]]; then
  SOURCE_COMMIT_TMP="$SOURCE_COMMIT_FILE.tmp.$$"
  printf '%s\n' "$SOURCE_GIT_COMMIT" > "$SOURCE_COMMIT_TMP"
  mv -f -- "$SOURCE_COMMIT_TMP" "$SOURCE_COMMIT_FILE"
fi

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
  "--selfplay-actor-tt-mb" "$ACTOR_TT_MB"
  "--selfplay-policy-node-cap" "$POLICY_NODE_CAP"
  "--selfplay-bestmove-node-cap" "$BESTMOVE_NODE_CAP"
  "--selfplay-temperature-moves" "$TEMPERATURE_MOVES"
)
if [[ -n "$TEACHER_EXTERNAL_QUANT_FILE" ]]; then
  AUTOPILOT_ARGS+=("--teacher-external-quant-file" "$EXTERNAL_TEACHER_STAGED")
fi
AUTOPILOT_ARGS+=(
  "--teacher-relabel-depth" "$RELABEL_DEPTH"
  "--teacher-relabel-every" "$RELABEL_EVERY"
  "--teacher-relabel-threads" "$RELABEL_THREADS"
  "--teacher-relabel-hash-mb" "$RELABEL_HASH_MB"
  "--teacher-relabel-max-nodes" "$RELABEL_MAX_NODES"
  "--teacher-sample-fraction" "$TEACHER_SAMPLE_FRACTION"
  "--min-teacher-depth" "$MIN_TEACHER_DEPTH"
  "--target-cp" "$TARGET_CP"
  "--cp-loss-weight" "$CP_LOSS_WEIGHT"
  "--teacher-mix" "$TEACHER_MIX"
  "--train-arch" "$TRAIN_ARCH"
  "--epochs" "$EPOCHS"
  "--batch-size" "$BATCH_SIZE"
  "--max-samples" "$MAX_SAMPLES"
  "--hidden-dim" "$HIDDEN_DIM"
  "--learning-rate" "$LEARNING_RATE"
  "--warm-start-learning-rate" "$WARM_START_LEARNING_RATE"
  "--warm-start"
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
  "--gate-min-score-delta" "0.05"
  "--gate-incremental-pst-policy" "regression-veto"
  "--gate-pst-veto-margin" "0.0"
  "--gate-paired-openings"
  "--gate-sprt"
  "--gate-sprt-delta1" "0.0575"
  "--gate-sprt-alpha" "0.05"
  "--gate-sprt-beta" "0.05"
  "--gate-sprt-min-pairs" "48"
  "--gate-sprt-batch-pairs" "$GATE_SPRT_BATCH_PAIRS"
  "--gate-sprt-max-pairs" "1600"
  "--trainer-backend" "torch"
  "--trainer-device" "cuda"
)
if [[ "$FRESH_INIT" != "1" ]]; then
  AUTOPILOT_ARGS+=(
    "--initial-checkpoint" "$INITIAL_CHECKPOINT"
    "--initial-checkpoint-weights-only"
  )
fi
if [[ -n "$CROSS_ARCH_GATE_BLEND_PERCENT" ]]; then
  AUTOPILOT_ARGS+=(
    "--cross-arch-gate-blend-percent" "$CROSS_ARCH_GATE_BLEND_PERCENT"
  )
fi

log "starting campaign_v2: PieBot-only node-capped depth-$RELABEL_DEPTH self-relabel training"
log "output root: $OUT_ROOT"
log "deadline budget: $HOURS hours (persisted once in autopilot_state.json)"
log "opening suite: $SELFPLAY_OPENINGS (sha256 $OPENINGS_SHA256)"
log "teacher node cap: $RELABEL_MAX_NODES nodes/position (measured node-cost calibration)"
if [[ "$FRESH_INIT" == "1" ]]; then
  log "fresh-init lineage: hidden-dim $HIDDEN_DIM from random weights, no warm-start checkpoint"
fi
"$PYTHON_BIN" -m training.nnue.autopilot "${AUTOPILOT_ARGS[@]}"

require_file "$OUT_ROOT/autopilot_state.json"
log "campaign deadline reached cleanly; state: $OUT_ROOT/autopilot_state.json"
