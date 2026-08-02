# PieBot (Rust Engine + NNUE Training Pipeline)

PieBot combines a Rust chess engine (`PieBot/`) with a Python NNUE data/training stack (`training/nnue/`) for continuous self-play, relabeling, training, and model-gated promotion.

## Repository Layout

- Engine crate: `PieBot/`
- Training pipeline: `training/nnue/`
- Setup and host docs: `documents/`
- Automation and helper scripts: `scripts/`
- Team workflow and gates: `AGENTS.md`

## Latest Changes (2026-08)

- Closed-loop model handoff is active: cycle `N+1` self-play/relabel uses the accepted model from cycle `N`.
- Replay-window training is active: each cycle can train on fresh + recent-cycle JSONL shards.
- Lagged teacher support is active: relabel can use an older accepted model to reduce coupling.
- Self-play now supports game-level parallel fan-out (`--parallel-games`, with `0` = auto by available cores / per-game threads).
- Autopilot now gates promotion via engine A/B; candidate model is promoted only if `compare_play` passes in `--same-search` mode.
- `compare_play` now applies per-side configs correctly (eval mode, blend, NNUE files, hash, threads).
- Noise opening sampling now uses engine-ordered top-K, not raw legal-move order.
- Self-play distinguishes real chess outcomes from truncation, and invalid outcomes are excluded
  from the legacy binary format rather than mislabeled as draws.
- Pipeline resume markers bind shard hashes, generator/relabel settings, seeds, input/model hashes,
  training configuration, checkpoints, and exported model checksums.
- Cargo dependencies are locked for production builds and pipeline-spawned engine commands.
- Full default and all-feature Rust target tests are green in the current tree.

## Current Status

- Baseline search: `PieBot/src/search/alphabeta.rs`
- Experimental search: `PieBot/src/search/alphabeta_temp.rs`
- Acceptance binaries: `accept`, `accept_temp`
- A/B runner: `compare_play`
- Training orchestrator: `training.nnue.autopilot`

Both baseline and experimental searches pass all 91 `matein3` cases at deterministic depth 7.

The default UCI path supports fixed-depth, movetime, and clock-managed searches. Search is still
synchronous, so asynchronous `stop`, pondering, and `go infinite` are not yet supported; the
self-play/relabel/training pipeline does not depend on those commands.

## Quick Start

Build key binaries:
```bash
cargo build --locked --release --manifest-path PieBot/Cargo.toml \
  --bin uci --bin selfplay --bin relabel_jsonl --bin compare_play \
  --bin accept --bin accept_temp
```

The default `uci` binary is the production Cozy Chess engine with the NNUE UCI
path. The legacy Pleco adapter is compatibility-only and requires the explicit
`--features board-pleco` opt-in.

Run a one-cycle smoke autopilot run:
```bash
python3 -m training.nnue.autopilot \
  --out-root /tmp/piebot_smoke \
  --max-cycles 1 \
  --selfplay-games 4 \
  --selfplay-depth 2 \
  --selfplay-threads 1 \
  --selfplay-parallel-games 0 \
  --teacher-relabel-depth 4 \
  --epochs 1 \
  --batch-size 128 \
  --trainer-backend auto \
  --trainer-device auto
```

Run the 7-day Zen5 profile:
```bash
python3 -m training.nnue.autopilot \
  --out-root /opt/piebot_runs/zen5_7d \
  --profile zen5_9755_7d \
  --hours 168
```

## Validation Commands

Python pipeline tests:
```bash
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
```

Rust full test gate:
```bash
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

Acceptance sanity (single thread):
```bash
PIEBOT_TEST_THREADS=1 PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 \
  cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin accept
PIEBOT_TEST_THREADS=1 PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 \
  cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin accept_temp
```

Game-level A/B sanity:
```bash
cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --games 40 --movetime 200 --noise-plies 12 --noise-topk 5 --threads 1
```

Model-only gate-style A/B (same search, different models):
```bash
cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --same-search --games 40 --movetime 200 --threads 1 \
  --base-eval nnue --base-nnue-quant-file /path/base.nnue \
  --exp-eval nnue --exp-nnue-quant-file /path/candidate.nnue
```

## License

AGPL-3.0. See `PieBot/LICENSE`.
