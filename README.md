# PieBot (Rust Chess Engine + Hand-Optimized NNUE)

PieBot is a chess engine written in Rust (`PieBot/`) with a large HalfKP NNUE, a Stockfish-style alpha-beta search (rewritten in October 2026, see [Search](#search)), Lazy SMP, optional Syzygy tablebases, and a neural network training pipeline (`training/nnue/`).

**Current measured strength (2026-10-16, one thread each, this repo's opening book):** the October search rewrite plus NNUE speedups is **+402 Elo** over the previous search (400 games, 10+0.1), and SPSA tuning added a further **+96 Elo** (1000 games). Against **full-strength Stockfish 19** at 60+0.5 the tuned build with Syzygy and `Contempt 40` scored 0 wins / 54 draws / 226 losses (19.3% win-or-draw, 280 games); the pre-tuning rewrite scored 14.3% and the previous search 1.4%. Details: [evidence/search_v2_20261016/README.md](evidence/search_v2_20261016/README.md).

PieBot is evaluated through internal `120+1` matches against CCRL-listed engines. Those performance estimates are not an official CCRL ranking or a Lichess rating. See the [Lichess audit](evidence/lichess_audit_20261001/REPORT.md) and [verified Lazy SMP deployment](evidence/lichess_smp_deploy_20261001/REPORT.md) for the measured results and their limits.

---

## Quick Start: How to Run the Engine

### 1. Build the Engine

Prerequisites: Rust toolchain (1.85+; shakmaty-syzygy needs it. Install via [rustup.rs](https://rustup.rs/)).

```bash
# Clone the repository
git clone https://github.com/grunsab/PieBot.git
cd PieBot/PieBot

# Build the release UCI binary *from inside PieBot/*
cargo build --release --bin uci
```

The compiled binary will be located at:
`PieBot/target/release/uci`

Build from inside `PieBot/`: `PieBot/.cargo/config.toml` sets `-C target-cpu=native`,
and cargo only reads it relative to the working directory. A build started from the
repo root with `--manifest-path` produces a generic x86-64 binary (about 5% slower
on a Zen 4 laptop, more on older code paths).

### 2. Auto-Loaded NNUE (Zero Configuration Required)

When launched, PieBot looks for `models/lc0_chunk_00034965.nnue` **relative to its working directory**, and if found loads it with `UseNNUE = true` and `EvalBlend = 100`:

```text
info string Loaded default NNUE model: models/lc0_chunk_00034965.nnue
```

If the engine is started from another directory the net is not found and it
silently plays on the piece-square eval. Scripts and GUIs started elsewhere should
set `NNUEQuantFile` to an absolute path and `UseNNUE` to `true`.

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
option name EvalBlend type spin default 100 min 0 max 100
option name Contempt type spin default 0 min -200 max 200
option name SyzygyPath type string default <empty>
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
| `EvalBlend` | `spin` | `100` | `0 .. 100` | Percentage weight of NNUE evaluation vs classical evaluation ($0 = 100\%$ classical, $100 = 100\%$ NNUE). The engine plays on the network alone by default; lower values mix in the piece-square score. |
| `Contempt` | `spin` | `0` | `-200 .. 200` | Draw bias in centipawns for the engine: a drawn line scores `+Contempt` for the side that is searching. Positive values seek draws, which helps against a much stronger opponent (vs Stockfish 19 at 20+0.2: 40 gave 15.0% win-or-draw against 10.4% at 0); leave it at 0 for normal play. |
| `SyzygyPath` | `string` | `<empty>` | Directories separated by `;` or `:` | Syzygy WDL/DTZ tables (3-4-5 men tested). The search probes WDL right after captures and pawn moves; at the root it plays the DTZ-best move. Prints `info string syzygy: N tables, up to K pieces` when loaded. |

Setting `PIEBOT_TUNE=1` in the engine's environment also advertises every search
constant as a spin option (used by `scripts/experiments/spsa.py`); without it those
options are hidden but still accepted.

To set options in UCI:
```text
setoption name Threads value 8
setoption name Hash value 256
setoption name EvalBlend value 100
```

The NNUE engine sizes its own search pool from `Threads`; no `RAYON_NUM_THREADS`
setting is needed. Helpers share the transposition table and immutable NNUE
weights, while the main search selects the move. Deterministic searches, explicit
node budgets, and the development dense-f32 NNUE backend use one search thread.
All published strength numbers for the October search are single-threaded; Lazy
SMP is carried over and tested for correctness, not yet for Elo.

`go movetime N` uses the whole budget. With a clock (`wtime`/`btime`) the front
end's allocation is a hard limit and the search aims to finish well inside it
(`TmSoft`, 47% of the allocation), stretching or shrinking that target by
best-move stability, the share of nodes spent on the best move, and a falling score.

Large hash sizes are cheap to set: the table is allocated from lazily zeroed
pages, so `Hash 8096` plus `ucinewgame` takes about 0.1 s.

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
The first-layer table is about 84 MB (41,024 features x 1024 x i16), so
accumulator updates are bound by memory latency, not arithmetic. What matters:

1. **Fused updates.** A quiet move is one pass over `src - out + in`, a capture
   one pass over `src - mover - victim + mover'`, so the row misses overlap.
   These are plain loops that the compiler vectorises to AVX-512 under
   `target-cpu=native`; they measured faster than the earlier hand-written AVX2
   intrinsics. The SCReLU head has an AVX-512 kernel (AVX2 and NEON remain).
2. **King-move refresh cache ("Finny table").** A king move rebuilds only its own
   perspective, by patching the cached accumulator for that king square with the
   few pieces that differ; the other perspective is patched incrementally.
3. **Lazy accumulators** (main search). Making a move records the feature change
   and prefetches its weight rows; the slot is built only when an evaluation needs
   it. Leaves answered from the TT (score or cached eval) never pay for an update.
4. All of the above are exact: parity tests compare against full recomputation,
   including sparse lazy evaluation along random games.

Measured on a Ryzen 9 7940HX, one thread: the October search went from 567k to
about 2.2-2.6M nodes/s through this work (the previous search: about 1.6M).

---

## Search

`PieBot/src/search/alphabeta.rs` was rewritten in October 2026 along modern
Stockfish lines (promoted from the experimental `alphabeta_temp.rs` on 2026-10-16):

- **Transposition table**: lockless, 4-way 64-byte buckets of packed 16-byte
  entries (move, score, static eval, depth, bound, PV flag, age), XOR-verified
  keys, age-aware replacement, prefetched as soon as a move is made. It caches the
  static eval, so loading or clearing a network clears it.
- **Move ordering**: staged picker (TT move before generation; winning captures
  by MVV, capture history and SEE; two killers and the counter move; quiets scored
  only when reached, from butterfly, pawn and continuation histories at 1, 2, 4 and
  6 plies, plus a bonus for direct checks; losing captures last).
- **Evaluation adjustments**: pawn, non-pawn (per side), minor-piece and
  continuation correction histories; a slight pull toward zero as the fifty-move
  counter runs.
- **Pruning and reductions**: PVS with aspiration windows, mate-distance pruning,
  razoring, reverse futility, null move (with verification at high depth), ProbCut,
  internal iterative reduction, late-move pruning, futility and history pruning,
  SEE pruning of quiets and captures, logarithmic LMR adjusted by PV/cut node,
  history, improving, checks and cutoff counts, with deeper/shallower re-searches.
- **Extensions**: singular extensions with double extension, multi-cut and
  negative extensions. Checks are not extended (`CheckExt=0`: +9.3 Elo
  [+0.3, +18.4] over 1600 games against extending every check).
- **Quiescence**: TT probes and stores, stand-pat from the corrected eval,
  futility and SEE filters, evasions in check.
- **Draws**: a position that repeats inside the search tree is a draw at once; one
  from the game before the root needs a real threefold. Upcoming repetitions are
  seen one move early through Stockfish-style cuckoo tables (`search/cuckoo.rs`,
  rebuilt from cozy-chess's Zobrist constants; +10.8 Elo [-1.5, +23.1]).
  `Contempt` sets the draw score.
- **Endgames**: optional Syzygy probing (`SyzygyPath`).
- **Tuning**: 55 constants are runtime tunables (`T::*` in `alphabeta.rs`); the
  defaults are SPSA-tuned (6000 game pairs, +95.8 Elo [+82, +110] validated).

Experiments tried and left off (defaults keep them inert): threat-aware quiet
ordering (`ThreatEscape`/`ThreatEnter`; -21 Elo at 50, neutral at 10), quiet
checks in the first quiescence ply (`QsChecks`, neutral), continuation history
split by check/capture (`ContSplit`, neutral), SEE-guarded check extension
(`CheckExt=2`, -14 Elo). Results: [evidence/search_v2_20261016](evidence/search_v2_20261016/README.md).

The mate suite is now read with depth escalation: the search is selective enough
that some mates at fixed depth 7 come a few plies later (**91/91 by depth 14 with
188,740 nodes**, against 91/91 at depth 7 with about 7.1M nodes for the previous
search).

## Testing & Quality Assurance

### Rust Unit & Integration Tests
```bash
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

### Python Training & Pipeline Tests
```bash
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
```

The tablebase integration test needs real tables and is skipped otherwise:
```bash
cd PieBot && PIEBOT_SYZYGY_PATH=/path/to/syzygy cargo test --release --test syzygy_temp
```

### Deterministic Mate Acceptance Suite
```bash
cd PieBot
PIEBOT_SUITE_FILE=src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=14 \
  cargo run --locked --release --bin accept
```
Expected: `cases=91 ... nodes=188740` (91/91 by depth 14); that node count is
the deterministic signature for proving a rebuild matches.

---

## Head-to-Head A/B Evaluation Workflow

Per `AGENTS.md`, search changes are A/B tested against the baseline before they
are promoted. Put the change in `PieBot/src/search/alphabeta_temp.rs` (a stub that
re-exports `alphabeta.rs` between experiments) and build the UCI binary on it with
the `exp-search` feature:

```bash
cd PieBot
cargo build --release --features exp-search --target-dir ../out/target_exp
```

Engine-vs-engine matches under real clocks (any two UCI engines, paired openings,
draws only on a threefold that actually occurred):

```bash
python3 scripts/experiments/uci_match.py \
  --a out/exp/uci --b out/base/uci \
  --a-opt NNUEQuantFile=$PWD/models/lc0_chunk_00034965.nnue --a-opt UseNNUE=true \
  --b-opt NNUEQuantFile=$PWD/models/lc0_chunk_00034965.nnue --b-opt UseNNUE=true \
  --games 400 --tc 10+0.1 --workers 14 --book books/openings_v1.fen --out out/ab.json
```

The same runner plays PieBot against Stockfish (`--b path/to/stockfish
--b-opt Threads=1`). Related tools in `scripts/experiments/`:

| Script | Purpose |
| :--- | :--- |
| `uci_match.py` | Parallel paired match; pentanomial Elo with 95% CI, per-move times |
| `ab_queue.py` | Runs a JSON queue of matches back to back; resumable |
| `spsa.py` | Asynchronous SPSA over the tunables (engine started with `PIEBOT_TUNE=1`) |
| `uci_nps.py` | Nodes per second of several engines on fixed positions |
| `loss_points.py` | Where lost games turned, judged by Stockfish |
| `gdb_profile.py` | Sampling profiler via gdb, for machines where `perf` is blocked |

Because tunables and experiment toggles are UCI options, many experiments need
no rebuild: play one binary against itself with different `--a-opt`/`--b-opt`
(start the runner with `PIEBOT_TUNE=1` so the options are advertised).

`compare_play` (in-process, fixed movetime) still works for `alphabeta` vs
`alphabeta_temp`.

---

## License

AGPL-3.0. See [`PieBot/LICENSE`](PieBot/LICENSE).
