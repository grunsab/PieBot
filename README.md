# PieBot (Rust Engine + NNUE Training Pipeline)

PieBot is a high-performance chess engine written in Rust (`PieBot/`) paired with an advanced neural network training and ingestion pipeline (`training/nnue/`) designed for continuous self-improvement, streaming LCZero corpus training, and empirical A/B evaluation against top-tier engines.

## Repository Layout

- **Engine Crate (`PieBot/`)**: Core chess engine featuring cozy-chess board representation, parallel alpha-beta search with advanced pruning/heuristics, hand-written AVX2 and ARM NEON SIMD inference kernels, and UCI protocol compliance.
- **Training Pipeline (`training/nnue/`)**: Streaming LCZero data ingestion, PyTorch training pipelines (dual-perspective HalfKP architecture), int8 quantization, and autonomous training orchestrators.
- **Automation & Benchmarking (`scripts/`)**: Automated CCRL ranked engine monitor (`ranked_engine_monitor.py`), cluster deployment tooling, and Elo analysis scripts.
- **Opening Books (`books/`)**: Curated opening suites including `books/openings_v1.fen` (1,279 positions).
- **Runbooks & Architectural Docs (`documents/`)**: Documentation for cluster deployment, LCZero training pipeline (`documents/LCZeroTraining.md`), and Super-GM roadmap.
- **Engineering Guidelines (`AGENTS.md`)**: Test-Driven Development (TDD) protocols, Cozy Search Change Management workflow, and cluster operations guide.

---

## Current Architecture & Milestones (September 2026)

### 1. SIMD-Accelerated NNUE Evaluation (arch-v2)
- **Network Topology (`PIENNQ02`)**: Dual-perspective HalfKP transformer at hidden dimension 1024 with Squared Clipped ReLU (SCReLU) activations, colour-anchored accumulators, and a side-to-move-first integer output head.
- **Vectorized Kernels**: Hand-written **AVX2** (x86-64) and **ARM NEON** (AArch64 / Apple Silicon) vectorized inference kernels, providing **2.02x throughput speedup** over scalar evaluation while maintaining bit-exact cross-platform parity.
- **Incremental Accumulators**: Accumulator state updates are lazily synchronized and incrementally updated during search.

### 2. High-Throughput LCZero Training Pipeline
- **Corpus Pivot**: Ingests high-quality LCZero self-play dataset streaming (July 7–September 7, 2026 corpus) using best-Q valuations and actual game outcomes, providing broad tactical and positional signal.
- **Parallel Pipeline**: Multi-worker asynchronous streaming ingestion with chunking, automated validation tracking, and disk budget eviction.
- **Cluster Deployment**: Actively training on dedicated multi-core hardware and RTX 4080 Super GPUs, reaching validation loss below 0.620 across 22,700+ chunks.

### 3. Modern Search Architecture (+268.9 Elo Banked)
Search selectivity and move ordering have been re-engineered into an interconnected, co-tuned system:
- **Move Ordering**:
  - **Capture History**: Piece-to-victim indexed capture history table for prioritizing high-potential captures.
  - **Multi-Ply Continuation History**: 1-ply and 2-ply continuation history tables capturing quiet move context.
  - **Counter-Move Heuristic**: Dynamic counter-move response table.
  - **Winning Capture Priority & MVV-LVA**: High-priority ordering for tactically sound captures.
  - **Lazy Move Ordering**: Defers move scoring, SEE calculations, and allocations until after the Transposition Table (TT) move is searched, completely bypassing ordering on >55% of interior nodes that fail high immediately (+185k NPS in game conditions).
- **Pruning & Reductions**:
  - **Principal Variation Search (PVS)** with Transposition Table cutoffs.
  - **Dynamic Late Move Reductions (LMR)**: Depth-dependent logarithmic curve modulated by check status, capture flag, and history score.
  - **Reverse Futility Pruning (RFP)** and **Futility Pruning (FP)** with improving-heuristic awareness.
  - **Null-Move Pruning (NMP)** with verification search adjustments.
  - **Static Exchange Evaluation (SEE)** pruning in both main search and quiescence search.
  - **Quiescence Search**: Delta pruning, capture-history ordering, and tactical check extensions.
- **Deterministic Node Signature**:
  - Across the 91-puzzle `matein3.txt` test suite at depth 7, search visits **8,777,932** nodes (a 25.2% tree reduction from the historical 11,742,536 baseline while solving 91/91 mate problems in ~3.2s).

### 4. External Anchoring & CCRL 3500+ Benchmarks
- Automated rating harness (`scripts/ranked_engine_monitor.py`) benchmarks live builds in long-clock tournament play (`120+1`) against verified CCRL 3500+ engines:
  - **Schoenemann 0.5.0** (CCRL 3504)
  - **Carp 3.0.1** (CCRL 3524)
  - **Lambergar 1.5** (CCRL 3508)
- Current engine performance reaches ~3,166 Elo against CCRL 3500+ competition.

---

## Quick Start

### Building the Engine
Build release binaries:
```bash
cargo build --locked --release --manifest-path PieBot/Cargo.toml \
  --bin uci --bin selfplay --bin relabel_jsonl --bin compare_play \
  --bin accept --bin accept_temp
```

The default `uci` binary is the production engine using Cozy Chess board representation with NNUE evaluation.

### Running Acceptance & Test Batteries
Run Python pipeline and script tests:
```bash
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
```

Run complete Rust test suite:
```bash
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

Run deterministic mate suite acceptance:
```bash
PIEBOT_SUITE_FILE=PieBot/src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 \
  cargo run --locked --release --bin accept --manifest-path PieBot/Cargo.toml
```

---

## Search A/B Evaluation Workflow

Per the change management policy in `AGENTS.md`, any search modifications must follow an isolated, empirical validation workflow:

1. **Fork to Temporary File**:
   - Copy `PieBot/src/search/alphabeta.rs` to `PieBot/src/search/alphabeta_temp.rs`.
   - Implement your changes in `alphabeta_temp.rs` only.
2. **Deterministic Acceptance**:
   - Ensure both `accept` and `accept_temp` solve all 91 mate-in-3 positions at depth 7.
3. **Paired A/B Screen (100–400 Games)**:
   ```bash
   cargo run --release --bin compare_play --manifest-path PieBot/Cargo.toml -- \
     --games 400 --movetime 150 --paired-openings --openings-file books/openings_v1.fen \
     --parallel-games 8 --threads 1 --json-out /tmp/ab_screen.json
   ```
4. **Promotion Criteria**:
   - Changes are promoted only if the paired-bootstrap 95% confidence lower bound (LCB) is positive (> 0 Elo).
   - Once approved, copy `alphabeta_temp.rs` to `alphabeta.rs` and reset `alphabeta_temp.rs` back to the re-export stub.

---

## Ranked Engine Monitoring

To continuously evaluate engine strength against external engines at tournament time controls:
```bash
python3 scripts/ranked_engine_monitor.py \
  --piebot-bin PieBot/target/release/uci \
  --engine-root /path/to/ccrl_engines \
  --campaign-root /path/to/campaign \
  --out-root /path/to/ranked_output \
  --baseline models/v8_cycle_000013_quant.nnue \
  --book books/openings_v1.fen \
  --time-control 120+1 \
  --games 400
```

---

## License

AGPL-3.0. See `PieBot/LICENSE`.
