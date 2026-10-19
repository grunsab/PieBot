# Contributor and agent guide

How to work on PieBot without breaking it or fooling yourself. The README
covers what the engine is; this file covers how changes get made and measured.
Older campaign handoffs are archived in `documents/history/`.

## Layout

| Path | What it holds |
|------|---------------|
| `PieBot/` | The Rust crate: engine, UCI front end (`src/uci.rs`, binary `src/bin/uci.rs`), tools |
| `PieBot/src/search/alphabeta.rs` | The search. `alphabeta_temp.rs` is the experiment fork (a re-export stub when idle) |
| `PieBot/src/eval/nnue/` | NNUE loader and inference (`network.rs` arch-v2, `network_v3.rs` arch-v3) |
| `PieBot/tests/`, `PieBot/src/suites/` | Integration tests, fixtures, mate suites |
| `training/nnue/` | Python data conversion and training (`lc0_to_v3bin.py`, `train_v3.py`) |
| `scripts/experiments/` | Match, SPSA, NPS and profiling tools |
| `models/` | Shipped nets (`MANIFEST.json` lists them) |
| `evidence/` | Measured results behind each decision. Treat as read-only records |
| `documents/` | Current guides; `documents/history/` for archived plans and handoffs |

## Building

Build from inside `PieBot/`:

```bash
cd PieBot && cargo build --release
```

`.cargo/config.toml` (target-cpu=native) is read from the working directory,
so `--manifest-path` from the repo root silently produces a slower generic
binary. Note that `cargo build/test --all-features` enables `board-pleco` and
overwrites `target/release/uci` with that build; rebuild normally afterwards.

## The NNUE path pitfall

The default net path is relative to the working directory, and a miss
silently falls back to the PST evaluation. Scripts must always pass
`UseNNUE=true` and an absolute `NNUEQuantFile`.

## Tests

Run the full battery before merging:

```bash
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
```

Search changes also need the mate suite. The search is selective, so read it
with depth escalation; it must solve 91/91:

```bash
cd PieBot && PIEBOT_SUITE_FILE=src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=14 cargo run --release --bin accept
```

The node count it prints is deterministic and works as a rebuild signature.
Syzygy tests run only when `PIEBOT_SYZYGY_PATH` points at a tablebase folder.

Write the failing test first (red, green, refactor). Keep tests deterministic:
single thread unless testing SMP, fixed seeds.

## Changing the search

1. Copy `alphabeta.rs` over the `alphabeta_temp.rs` stub and edit only the copy.
2. Build the experiment binary with `--features exp-search --target-dir ../out/target_exp`;
   its UCI front end runs on `alphabeta_temp.rs`.
3. Check the mate suite on both `accept` and `accept_temp`. A change that moves
   the node count by under ~5% or loses a mate is rarely worth playing games for.
4. Play it against the baseline with `scripts/experiments/uci_match.py`
   (paired openings, real clocks, pentanomial confidence interval). Use at
   least 10+0.1; very short time controls understate depth-dependent changes.
5. Promote only after a 1000-game confirmation whose 95% lower bound is above
   zero. Screens tend to overstate. Then copy the fork into `alphabeta.rs` and
   restore the stub.

Search constants are runtime tunables (`tunables!` / `tp(T::...)`). With
`PIEBOT_TUNE=1` in the engine's environment they appear as UCI spin options,
so one binary can be tested against itself. Tune with
`scripts/experiments/spsa.py`, then validate the tuned values in a separate
match before adopting them.

## Training the NNUE

- Data: LCZero training archives (ODbL/DbCL) converted with
  `training/nnue/lc0_to_v3bin.py` into 32-byte records.
- Training: `training/nnue/train_v3.py` on a rented GPU. It exports a
  quantized checkpoint at each evaluation interval and at the end.
- Gating: copy checkpoints back and play them against the current default net
  on a local machine, so the GPU rental is not spent on matches.
- A new file layout needs a new file version in the loader, plus a fixture
  exported by the Python reference evaluator and a Rust test that matches it
  exactly (see `PieBot/tests/nnue_arch_v3.rs`). Incremental updates must equal
  full recomputation along random games.
- Changing the evaluation must reset the TT, which caches static evals (the
  UCI setters already do).

## Measuring

- Draws count only on a threefold that occurred, never on a claimable one.
- Play each opening with both colours.
- Stockfish 19 reference protocol: full strength, 60+0.5, 280 games, seed
  1901, Syzygy 3-5 men, PieBot `Contempt 40`.
- Stockfish and other engines are evaluation anchors only. Never use their
  scores or moves as training labels. LCZero training data is allowed.
- Report results faithfully, with confidence intervals and game counts. Do not
  present a small sample as an Elo result.
- On a shared machine, keep the number of busy cores within the agreed budget
  (one game uses one core when both engines run single-threaded).
- `perf` may be blocked; `scripts/experiments/gdb_profile.py` samples instead.

## Operational pitfalls

- `pkill -f PATTERN` also matches the shell whose command line contains
  PATTERN, and kills it. Kill exact PIDs instead.
- Processes left running on a remote box should be detached (`nohup`/`setsid`)
  and must set their own `PATH`; they do not inherit an interactive shell.
- Training on many memory-mapped shards needs `ulimit -n 65536` first.
- Long builds and matches outlast tool timeouts. Run them in the background
  and poll their output files.
- Check that the machine was awake (elapsed vs CPU time) before reading meaning
  into a slow run.

## Git

- Commit only when asked, on a branch, with attribution lines. The user pushes.
- Never rewrite published history. Large binaries belong in a release asset or
  LFS, because deleting them later does not shrink the repository.

## Further reading

- `README.md` for the engine, results and credits.
- `documents/` for the current search, NNUE and development guides.
- `documents/history/agent_handoffs_2026.md` for the full campaign logs these
  rules came from, including the measurement corrections behind them.
