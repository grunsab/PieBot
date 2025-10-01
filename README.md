# PieBot (Rust) – NNUE + Alpha-Beta

This crate is being reimplemented around a CPU-only NNUE evaluation and a highly optimized alpha-beta search (with parallel search), targeting top-tier chess-engine performance.

## Current Status

- Cozy-chess integration and adapter layer.
- UCI skeleton binary (basic command handling, placeholder move selection).
- Perft implementation with tests for correctness.

## Build

```bash
cd piebot
cargo build --release
```

Release builds use `-C target-cpu=native` for NEON/AVX on the host CPU.

## Run

- UCI loop:
```bash
cargo run --bin uci
```

- Perft (depth 3):
```bash
cargo run --bin perft -- 3
```

## Roadmap (abridged)

- Minimal alpha-beta/PVS with TT and simple eval.
- NNUE v1: HalfKP-style features with incremental accumulators, scalar + NEON/AVX.
- Heuristics: move ordering, LMR, null-move, aspiration windows, quiescence.
- Parallel search: split points and work-stealing for multi-thread speedup.
- Self-play generator in Rust and NNUE training pipeline (Python).

## Notes

All old MCTS/Torch-based code and scripts have been removed to keep the codebase focused on the new NNUE + alpha-beta direction.
