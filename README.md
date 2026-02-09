# PieBot (Rust Engine + NNUE Training Pipeline)

PieBot combines a Rust chess engine (`PieBot/`) with a Python NNUE data/training stack (`training/nnue/`) for continuous self-play, relabeling, training, and model-gated promotion.

## Repository Layout

- Engine crate: `PieBot/`
- Training pipeline: `training/nnue/`
- Setup and host docs: `documents/`
- Automation and helper scripts: `scripts/`
- Team workflow and gates: `AGENTS.md`

## Latest Changes (2026-02)

- Closed-loop model handoff is active: cycle `N+1` self-play/relabel uses the accepted model from cycle `N`.
- Replay-window training is active: each cycle can train on fresh + recent-cycle JSONL shards.
- Lagged teacher support is active: relabel can use an older accepted model to reduce coupling.
- Self-play now supports game-level parallel fan-out (`--parallel-games`, with `0` = auto by available cores / per-game threads).
- Autopilot now gates promotion via engine A/B; candidate model is promoted only if `compare_play` passes in `--same-search` mode.
- `compare_play` now applies per-side configs correctly (eval mode, blend, NNUE files, hash, threads).
- Noise opening sampling now uses engine-ordered top-K, not raw legal-move order.
- `bench.rs` compile drift against Pleco APIs has been fixed.
- Full `cargo test -q` is green in current tree.

## Current Status

- Baseline search: `PieBot/src/search/alphabeta.rs`
- Experimental search: `PieBot/src/search/alphabeta_temp.rs`
- Acceptance binaries: `accept`, `accept_temp`
- A/B runner: `compare_play`
- Training orchestrator: `training.nnue.autopilot`

Known gap to keep iterating:
- Experimental acceptance (`accept_temp`) still misses one `matein3` case in default depth-7 settings (index 15).

## Quick Start

Build key binaries:
```bash
cargo build --release --manifest-path PieBot/Cargo.toml \
  --bin selfplay --bin relabel_jsonl --bin compare_play --bin accept --bin accept_temp
```

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
python3 -m unittest -q \
  training.nnue.tests.test_run_pipeline \
  training.nnue.tests.test_autopilot
```

Rust full test gate:
```bash
cargo test -q --manifest-path PieBot/Cargo.toml
```

Acceptance sanity (single thread):
```bash
cargo run --quiet --manifest-path PieBot/Cargo.toml --bin accept
cargo run --quiet --manifest-path PieBot/Cargo.toml --bin accept_temp
```

Game-level A/B sanity:
```bash
cargo run --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --games 40 --movetime 200 --noise-plies 12 --noise-topk 5 --threads 1
```

Model-only gate-style A/B (same search, different models):
```bash
cargo run --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --same-search --games 40 --movetime 200 --threads 1 \
  --base_eval nnue --base_nnue_quant_file /path/base.nnue \
  --exp_eval nnue --exp_nnue_quant_file /path/candidate.nnue
```

## License

AGPL-3.0. See `PieBot/LICENSE`.
