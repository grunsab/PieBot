# PieBot (Rust Chess Engine + Hand-Optimized NNUE)

PieBot is a high-performance chess engine written in Rust (`PieBot/`) featuring hand-written AVX2 and ARM NEON SIMD inference kernels, parallel alpha-beta search with modern selectivity heuristics, and an advanced neural network training pipeline (`training/nnue/`) designed for continuous self-improvement and empirical validation against top-tier engines.

PieBot is evaluated through internal `120+1` matches against CCRL-listed engines. Those performance estimates are not an official CCRL ranking or a Lichess rating. See the [Lichess audit](evidence/lichess_audit_20261001/REPORT.md) and [verified Lazy SMP deployment](evidence/lichess_smp_deploy_20261001/REPORT.md) for the measured results and their limits.

---

## Quick Start: How to Run the Engine

### 1. Build the Engine

Prerequisites: Rust toolchain (1.78+ recommended, install via [rustup.rs](https://rustup.rs/)).

```bash
# Clone the repository
git clone https://github.com/grunsab/PieBot.git
cd PieBot

# Build the release UCI binary
cargo build --release --bin uci --manifest-path PieBot/Cargo.toml
```

The compiled binary will be located at:
`PieBot/target/release/uci`

### 2. Auto-Loaded NNUE (Zero Configuration Required)

When launched, PieBot **automatically detects and loads** the highest-efficiency hand-optimized NNUE model present in the repository (`models/lc0_chunk_00034965.nnue`), sets `UseNNUE = true`, and configures the optimal evaluation blend (`EvalBlend = 75`):

```text
info string Loaded default NNUE model: models/lc0_chunk_00034965.nnue
```

No manual file path configuration is required—simply pull the repo, build, and run!

---

## Running PieBot

### Option A: Interactive Command Line (CLI)

You can interact with PieBot directly using the standard Universal Chess Interface (UCI) protocol:

```bash
./PieBot/target/release/uci
```

Example UCI session:

```text
uci
info string Loaded default NNUE model: models/lc0_chunk_00034965.nnue
id name PieBot NNUE
id author PieBot Team
option name Threads type spin default 1 min 1 max 512
option name Hash type spin default 64 min 1 max 16384
option name UseNNUE type check default true
option name NNUEFile type string default 
option name NNUEQuantFile type string default models/lc0_chunk_00034965.nnue
option name EvalBlend type spin default 75 min 0 max 100
uciok

isready
readyok

# Set position and search with fixed time (1000 ms)
position startpos moves e2e4 e7e5 g1f3 b8c6
go movetime 1000
info depth 10 seldepth 12 score cp 32 nodes 45210 nps 1125000 pv f1c4 g8f6
bestmove f1c4

# Search to fixed depth
position fen r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 0 5
go depth 12

# Play with clock time control (White has 60s + 0.5s increment)
go wtime 60000 btime 60000 winc 500 binc 500

# Exit the engine
quit
```

---

### Option B: Running in a Chess GUI

PieBot works out-of-the-box with any standard UCI-compliant chess graphical interface:

#### 1. CuteChess
1. Open **CuteChess** $\to$ **Tools** $\to$ **Settings** $\to$ **Engines**.
2. Click **Add...** (+).
3. Set the binary path to `PieBot/target/release/uci`.
4. Protocol will auto-detect as **UCI**.
5. Set desired **Hash** (e.g., 256 MB or 1024 MB) and **Threads** (e.g., 4 or 8).
6. Click **OK**. PieBot is ready for matches and tournament play.

#### 2. Nibbler
1. Open **Nibbler** $\to$ **Engine** $\to$ **Choose engine...**.
2. Select `PieBot/target/release/uci`.
3. Nibbler will initialize the engine and immediately display search evaluations and PV lines.

#### 3. Banksia GUI / Arena / En Croissant
1. Navigate to Engine Management / Install New Engine.
2. Select `PieBot/target/release/uci`.
3. The engine initializes automatically with NNUE enabled.

---

## UCI Configuration Options Reference

| Option | Type | Default | Valid Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| `Threads` | `spin` | `1` | `1 .. 512` | Total Lazy SMP search threads, including the main search. Workers and their private search state are reused across moves. |
| `Hash` | `spin` | `64` | `1 .. 16384` | Transposition Table (TT) capacity in megabytes (MiB). |
| `UseNNUE` | `check` | `true` | `true / false` | Enables neural network evaluation. Defaults to `true` when a valid model is found. |
| `NNUEQuantFile` | `string` | *(auto)* | Valid file path | Path to quantized int8 NNUE model (`PIENNQ02`). Auto-detects `models/lc0_chunk_00034965.nnue`. |
| `NNUEFile` | `string` | `""` | Valid file path | Path to dense f32 NNUE model (`PIENNUE1`, for training/research development). |
| `EvalBlend` | `spin` | `75` | `0 .. 100` | Percentage weight of NNUE evaluation vs classical evaluation ($0 = 100\%$ classical, $100 = 100\%$ NNUE, $75$ is tournament optimal). |

To set options in UCI:
```text
setoption name Threads value 8
setoption name Hash value 256
setoption name EvalBlend value 75
```

The NNUE engine sizes its own search pool from `Threads`; no `RAYON_NUM_THREADS`
setting is needed. Helpers share the transposition table and immutable NNUE
weights, while the main search selects the move. Deterministic searches, explicit
node budgets, and the development dense-f32 NNUE backend use one search thread.

---

## Bundled High-Efficiency Hand-Optimized NNUE Models

All neural network weights reside in the [`models/`](models/) directory and are indexed in [`models/MANIFEST.json`](models/MANIFEST.json):

| Model File | Architecture | Elo Rating (Classical `120+1`) | Validation Loss | Description |
| :--- | :---: | :---: | :---: | :--- |
| [`lc0_chunk_00034965.nnue`](models/lc0_chunk_00034965.nnue) | `PIENNQ02` (arch-v2) | **3203.8 Elo** | **0.6201** | **CURRENT BEST.** Trained on 40,000+ chunks of LCZero games with best-Q and game outcomes. Scores 14.5% against CCRL 3500+ engines with a 28.9% draw rate and checkmate win against Carp 3.0.1. **Auto-loaded by default.** |
| [`v8_cycle_000013_quant.nnue`](models/v8_cycle_000013_quant.nnue) | `PIENNQ02` (arch-v2) | **3009.4 Elo** | 0.6550 | Campaign v8 cycle 13 self-play network. Objective-corrected normalized-cp Huber loss. |
| [`cycle_000098_quant.nnue`](models/cycle_000098_quant.nnue) | `PIENNQ01` (arch-v1) | 2422.0 Elo | — | Historical baseline retained for CPU benchmarking qualifications. |

---

## Hand-Optimized SIMD Vectorized NNUE Architecture

PieBot uses an **arch-v2 (`PIENNQ02`)** neural architecture with hardware-vectorized inference kernels written from scratch:

```
[HalfKP Dual-Perspective Features] (2 x 41,024 inputs)
                     │
         [Accumulator Layer (i16)]
                     │
        [SCReLU Activation: (x / 256)² clamped to 0..255]
                     │
    ┌────────────────┴────────────────┐
 [White Perspective]              [Black Perspective]
  (1024 i8 weights)                (1024 i8 weights)
    └────────────────┬────────────────┘
                     │  (Hand-written AVX2 / ARM NEON SIMD)
                     ▼
             [Output Value (cp)]
```

### Performance Features
1. **Hand-Written SIMD Kernels**:
   - **x86-64**: Hand-written **AVX2** 256-bit SIMD implementation (`simd_screlu_head_h1024_avx2`).
   - **AArch64 / Apple Silicon**: Hand-written **ARM NEON** 128-bit SIMD implementation (`simd_screlu_head_h1024_neon`).
   - Yields a **2.02x evaluation speedup** over scalar execution with bit-exact cross-platform parity.
2. **Incremental Accumulator Updates**:
   - Accumulator states are updated lazily and incrementally upon move application, only updating changed piece-square indices and the active king's perspective.
3. **Squared Clipped ReLU (SCReLU)**:
   - Provides smooth, non-linear activation gradient without expensive floating-point operations.

---

## Modern Search Selectivity Architecture

PieBot implements an interconnected, co-tuned search system with modern selectivity:

- **Move Ordering**:
  - **Capture History**: Piece-to-victim indexed capture history prioritizing tactically viable captures.
  - **1-Ply & 2-Ply Continuation History**: Dual-depth quiet move context tables.
  - **Counter-Move Heuristic**: Response-move prioritization based on previous ply.
  - **Lazy Move Ordering**: Defers move scoring, SEE calculations, and allocations until after the Transposition Table (TT) move is searched, bypassing ordering on >55% of interior nodes that fail high immediately (+185k NPS).
- **Pruning & Reductions**:
  - **Principal Variation Search (PVS)** with Transposition Table cutoffs.
  - **Dynamic Late Move Reductions (LMR)**: Depth-dependent logarithmic curve modulated by check status, capture flag, and history score.
  - **Reverse Futility Pruning (RFP)** and **Futility Pruning (FP)** with improving-heuristic awareness.
  - **Null-Move Pruning (NMP)** with adaptive reduction depths.
  - **Static Exchange Evaluation (SEE)** pruning in both main search and quiescence search.
  - **Quiescence Search**: Delta pruning, capture-history ordering, and tactical check extensions.
- **Deterministic Node Signature**:
  - Solves **91/91** mate problems in the `matein3.txt` test suite at depth 7 in **~1.9 seconds** (5,030,782 nodes).

---

## Testing & Quality Assurance

### Rust Unit & Integration Tests
```bash
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

### Python Training & Pipeline Tests (554 Tests)
```bash
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
```

### Deterministic Mate Acceptance Suite
```bash
PIEBOT_SUITE_FILE=PieBot/src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 \
  cargo run --locked --release --bin accept --manifest-path PieBot/Cargo.toml
```

---

## Head-to-Head A/B Evaluation Workflow

Per `AGENTS.md`, all search changes undergo empirical A/B evaluation against the baseline:

```bash
# Paired opening A/B match runner
cargo run --locked --release --bin compare_play --manifest-path PieBot/Cargo.toml -- \
  --games 400 --movetime 1000 --paired-openings --openings-file books/openings_v1.fen \
  --base-eval nnue --base-blend 75 --base-nnue-quant-file models/lc0_chunk_00034965.nnue \
  --exp-eval nnue --exp-blend 75 --exp-nnue-quant-file models/lc0_chunk_00034965.nnue \
  --parallel-games 8 --threads 1 --json-out /tmp/ab_screen.json
```

---

## License

AGPL-3.0. See [`PieBot/LICENSE`](PieBot/LICENSE).
