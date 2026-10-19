# Development

## Building

```bash
cd PieBot
cargo build --release --bin uci
```

Always build from inside `PieBot/`: `.cargo/config.toml` (which sets
`-C target-cpu=native`) is read from the working directory, so a build from
the repository root with `--manifest-path` produces a generic binary.

Scripts should always pass `UseNNUE=true` and an absolute `NNUEQuantFile`. The
default network path is relative to the working directory, and if it is
missing the engine silently plays on the piece-square evaluation.

## Tests

```bash
# Rust
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml

# Python (needs numpy, python-chess; the training tests also need PyTorch)
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests

# Syzygy integration test (skipped unless tables are given)
cd PieBot && PIEBOT_SYZYGY_PATH=/path/to/syzygy cargo test --release --test syzygy_temp
```

Search changes must also pass the mate-suite gate described in
[SEARCH.md](SEARCH.md#mate-suite-gate).

## Testing a search change

1. `PieBot/src/search/alphabeta_temp.rs` is normally a stub that re-exports
   `alphabeta.rs`. Copy `alphabeta.rs` over it and make the change there.
2. Build the experimental UCI binary with the `exp-search` feature:

   ```bash
   cd PieBot
   cargo build --release --features exp-search --target-dir ../out/target_exp
   ```

3. Play it against the baseline under real clocks:

   ```bash
   NET=$PWD/models/lc0_chunk_00034965.nnue
   python3 scripts/experiments/uci_match.py \
     --a out/target_exp/release/uci --b PieBot/target/release/uci \
     --a-opt NNUEQuantFile=$NET --a-opt UseNNUE=true \
     --b-opt NNUEQuantFile=$NET --b-opt UseNNUE=true \
     --games 1000 --tc 10+0.1 --workers 12 \
     --book books/openings_v1.fen --out out/ab.json
   ```

4. Adopt the change only if the result is clearly positive, then copy
   `alphabeta_temp.rs` over `alphabeta.rs` and restore the stub.

Many experiments need no rebuild: tunables and experiment toggles are UCI
options, so one binary can play itself with different `--a-opt`/`--b-opt`
values (start the runner with `PIEBOT_TUNE=1` in the environment so the
options are advertised).

`uci_match.py` plays each opening twice with colours reversed, ends games on a
threefold repetition only when it actually occurs, reports Elo with a
pentanomial 95% confidence interval, and resumes an interrupted match with
`--resume`.

## Tools

| Script | Purpose |
| :--- | :--- |
| `scripts/experiments/uci_match.py` | Parallel paired match between any two UCI engines; pentanomial Elo, per-move times |
| `scripts/experiments/ab_queue.py` | Runs a JSON queue of matches back to back; resumable |
| `scripts/experiments/spsa.py` | Asynchronous SPSA over the search tunables |
| `scripts/experiments/uci_nps.py` | Nodes per second of several engines on fixed positions |
| `scripts/experiments/loss_points.py` | Finds where lost games turned, judged by Stockfish |
| `scripts/experiments/gdb_profile.py` | Sampling profiler through gdb, for machines where `perf` is unavailable |
| `training/nnue/lc0_to_v3bin.py` | Converts LCZero V6 archives into 32-byte training records |
| `training/nnue/train_v3.py` | Trains and exports arch-v3 (`PIENNQ03`) networks on a GPU |

## Tuning with SPSA

Every new network changes the evaluation scale the pruning margins were tuned
against, so the search should be retuned after a network change.

1. Build the UCI binary and run `scripts/experiments/spsa.py` with
   `PIEBOT_TUNE=1` in the engine's environment. Previous rounds used 6000 game
   pairs at 6+0.06, then 8000 at 8+0.08.
2. Validate tuned against untuned values with `uci_match.py` over 1000 games
   at 10+0.1 before changing the defaults in `alphabeta.rs`.

## Measuring against Stockfish 19

To keep results comparable across builds, use the same protocol each time:

- `uci_match.py`, full-strength Stockfish 19, one thread each;
- 60+0.5, 280 games, `books/openings_v1.fen`, `--seed 1901`;
- PieBot with `SyzygyPath` set (3-4-5 men) and `Contempt=40`.

```bash
python3 scripts/experiments/uci_match.py \
  --a PieBot/target/release/uci --b /path/to/stockfish \
  --a-opt NNUEQuantFile=$NET --a-opt UseNNUE=true \
  --a-opt SyzygyPath=$HOME/syzygy345 --a-opt Contempt=40 \
  --b-opt Threads=1 \
  --games 280 --tc 60+0.5 --seed 1901 --workers 12 \
  --book books/openings_v1.fen --out out/sf19.json
```

Stockfish is used only as an opponent and for analysis, never as a source of
training labels.

## Training on a rented GPU

Networks were trained on a rented GPU machine while all games ran on a local
machine, which keeps rental time to the GPU work itself:

- On the GPU machine: download and convert the LCZero archives
  (`lc0_to_v3bin.py`), then train with `train_v3.py`. Raise the open-file
  limit first (`ulimit -n 65536`); the trainer memory-maps one file per
  archive.
- Locally: copy each exported checkpoint back and gate it against the current
  network with `uci_match.py` (same binary, 200 games at 10+0.1), confirming
  the best one over 1000 games before any Stockfish measurement.
