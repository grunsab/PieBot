# PieBot Repository (Rust Engine + NNUE Training Pipeline)

This repository now includes the full project in one git root:
- Rust engine crate in `PieBot/`
- Python NNUE data/training pipeline in `training/nnue/`
- setup docs and scripts in `documents/` and `scripts/`

## Current State (Accurate Snapshot)

- Search/eval code is active in `PieBot/src/` with alpha-beta variants, TT, and NNUE eval paths.
- A/B workflow is supported via baseline `alphabeta.rs` and experimental `alphabeta_temp.rs`.
- Rust self-play (`selfplay` binary) produces JSONL shards with:
  - `played_move` (sampled move)
  - `target_best_move` (teacher target)
  - `best_move` (compat alias)
  - `value_cp`, `policy_top`, game outcome labels
- Stronger-teacher relabeling exists via `relabel_jsonl` binary.
- End-to-end Python pipeline (`training.nnue.run_pipeline`) supports:
  - self-play generation
  - periodic relabeling
  - training (stub or torch backend)
  - export to dense + quantized NNUE formats
  - resume mode for interrupted runs
- Unattended orchestration exists via `training.nnue.autopilot`:
  - crash-safe state file
  - single-instance lock
  - retry/backoff
  - 7-day Zen5 profile (`zen5_9755_7d`)

## Known Gaps

- Full `cargo test` currently fails due compile issues in `PieBot/src/bin/bench.rs`.
- Targeted NNUE/self-play Rust tests and Python pipeline tests pass.
- Training stack is functional and automated, but not yet validated at super-GM strength.

## Quick Start

Build key Rust binaries:
```bash
cargo build --release --manifest-path PieBot/Cargo.toml --bin selfplay --bin relabel_jsonl
```

Run a tiny end-to-end smoke cycle:
```bash
python3 -m training.nnue.autopilot \
  --out-root /tmp/piebot_smoke \
  --max-cycles 1 \
  --selfplay-games 2 \
  --selfplay-depth 1 \
  --teacher-relabel-depth 2 \
  --teacher-relabel-every 2 \
  --epochs 1 \
  --batch-size 64 \
  --trainer-backend auto \
  --trainer-device cuda
```

Run the 7-day Zen5 profile:
```bash
python3 -m training.nnue.autopilot \
  --out-root /opt/piebot_runs/zen5_7d \
  --profile zen5_9755_7d \
  --hours 168 \
  --trainer-backend torch \
  --trainer-device cuda
```

## Repo Map

- Engine crate: `PieBot/`
- Training pipeline: `training/nnue/`
- Ubuntu setup guide: `documents/ZEN5_3090_NNUE_SETUP.md`
- Team workflow requirements: `AGENTS.md`

## Validation Commands

Python:
```bash
python3 -m unittest -q training.nnue.tests.test_run_pipeline training.nnue.tests.test_autopilot
```

Targeted Rust:
```bash
cargo test -q --manifest-path PieBot/Cargo.toml --no-default-features --test nnue_eval --test nnue_quant --test selfplay_io --test selfplay
```

## License

AGPL-3.0. See `PieBot/LICENSE`.
