# PieBot (Rust Engine + NNUE Training Pipeline)

PieBot combines a Rust chess engine (`PieBot/`) with a Python NNUE data/training stack (`training/nnue/`) for continuous self-play, relabeling, training, and model-gated promotion.

## Repository Layout

- Engine crate: `PieBot/`
- Training pipeline: `training/nnue/`
- Setup and host docs: `documents/`
- Automation and helper scripts: `scripts/`
- Team workflow and gates: `AGENTS.md`

## Latest Changes (2026-08-16)

**Evaluation — arch-v2 NNUE.** The network is a dual-perspective SCReLU HalfKP
transformer at hidden 1024 (file magic `PIENNQ02`), with two colour-anchored
accumulators and a side-to-move-first integer head. The engine dispatches on the
file magic, so v1 (`PIENNQ01`) nets still load. Cross-language parity is enforced
by committed fixtures (`PieBot/tests/nnue_arch_v2.rs`).

**Search.** Promoted arms, each confirmed over 1000 paired games with a
paired-bootstrap 95% lower bound above zero: interior PVS + TT-move-first,
reverse futility, futility, S6 (drop null-move verification at depth <= 12, plus
quiescence SEE and delta pruning, +56 Elo), and two 2026-08-16 move-ordering
repairs — a bounded/aged history table with malus (+18.1 Elo) and winning-capture
priority above the quiet ordering band (+20.2 Elo). Together the latter two
measure **+88.7 Elo, 95% CI [+65.0, +113.3] at 1000 ms/move**.

**Promotion gate.** Two staged repairs. The SPRT indifference point was
recalibrated to the effect size the trainer actually produces, and confirmation
batches were resized from 24 to 180 opening pairs after `compare_play` was found
to clamp match workers to the pairs available — the gate had been running 24 of
184 cores. Confirmations went from ~37 min to ~9 min.

**Training objective.** campaign_v8 adds a normalised-centipawn Huber term to the
WDL/BCE loss and restores a game-outcome component (`TEACHER_MIX = 0.9`). The
previous lineage stalled for 28 cycles at +4.91 Elo (12,800 gate games) because
BCE through `sigmoid(cp/400)` keeps only 28% of its gradient at 1000 cp while 28%
of labels sit above |600| — the signal was present, the objective could not see
it. See `evidence/objective_saturation_20260816.json`.

**Pipeline.** Closed-loop model handoff, replay-window training, float-checkpoint
warm start, lagged-teacher support, game-level self-play fan-out, resume markers
binding shard/model/config hashes, and locked Cargo dependencies. The dataloader
reads `*.jsonl.gz` as well as `*.jsonl`.

## Current Status

- Baseline search: `PieBot/src/search/alphabeta.rs`
- Experimental search: `PieBot/src/search/alphabeta_temp.rs` (a re-export stub
  between experiments, so each A/B starts from an exact buildable baseline)
- Acceptance binaries: `accept`, `accept_temp` — A/B runner: `compare_play`
- Training orchestrator: `training.nnue.autopilot`
- Current best net: `models/v8_cycle_000013_quant.nnue` (arch-v2, blend 75).
  `models/cycle_000098_quant.nnue` is **retained as a dependency** —
  `scripts/cpu_benchmark.sh` reads it by path to qualify hosts.

Both searches pass all 91 `matein3` cases at deterministic depth 7. The
acceptance node count is a deterministic signature, useful for proving a remote
host really rebuilt your code: `accept` is currently **11742536**.

### Strength, and why the number carries a wide interval

Best estimate **~2650 CCRL 40/15, honest interval 2400-2900**. That interval is
instrument error, not sampling noise, and **both instruments are biased low**:

- The Stockfish `UCI_LimitStrength` ladder is not self-consistent — the same
  binary and net measured 2146 at rungs 1800/2100 and 2422 at 2400/2700 on the
  same idle host, with disjoint confidence intervals. Only compare within a
  fixed rung set; 3000/3190 is the canonical one.
- The 150 ms A/B harness **understates** changes whose value scales with depth,
  by roughly 2.3x against a 1000 ms measurement. Use >= 1000 ms for anything
  depth-dependent.
- No PieBot game has ever been played against a CCRL-listed engine, so no number
  here is externally anchored.

The default UCI path supports fixed-depth, movetime, and clock-managed searches.
Search is still synchronous, so asynchronous `stop`, pondering, and `go infinite`
are not supported; the training pipeline does not depend on them.

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

Game-level A/B sanity (smoke only — 40 games resolves nothing below ~60 Elo):
```bash
cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --games 40 --movetime 200 --noise-plies 12 --noise-topk 5 --threads 1
```

Promoting a search change requires the full protocol, not the smoke test: a
400-game screen, then a 1000-game confirmation, promoted only if the
paired-bootstrap 95% lower bound is above zero. Screens overstate — the history
rewrite screened +27.0 and confirmed +18.1.

```bash
cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --games 1000 --movetime 1000 --noise-plies 12 --noise-topk 5 \
  --threads 1 --paired-openings --parallel-games 8 --seed 1 --json-out /tmp/ab.json
```

Use `--movetime 1000` or longer for any change whose value plausibly scales with
depth. Measured 2026-08-16: the same two search arms are worth +38.3 Elo at
150 ms and +88.7 Elo at 1000 ms, because move-ordering quality compounds with
search length. Several arms rejected on the 150 ms harness may have been rejected
by an instrument that could not see them.

Strength against the Stockfish anchor (relative instrument — fix the rungs and
the host, and only compare like with like):
```bash
python3 scripts/uci_elo_ladder.py \
  --piebot-command PieBot/target/release/uci \
  --piebot-nnue models/v8_cycle_000013_quant.nnue --piebot-blend 75 \
  --stockfish-command /path/to/stockfish --rungs 3000,3190 \
  --games 100 --time-control 60+0.5 --out-dir /tmp/ladder
```

`scripts/uci_elo_arena.py` also accepts `--stockfish-full-strength`, which
disables `UCI_LimitStrength` entirely. That is the only configuration anchored to
a published CCRL rating, and it becomes usable once the strength gap is small
enough to score meaningfully.

Model-only gate-style A/B (same search, different models):
```bash
cargo run --locked --release --quiet --manifest-path PieBot/Cargo.toml --bin compare_play -- \
  --same-search --games 40 --movetime 200 --threads 1 \
  --base-eval nnue --base-nnue-quant-file /path/base.nnue \
  --exp-eval nnue --exp-nnue-quant-file /path/candidate.nnue
```

## License

AGPL-3.0. See `PieBot/LICENSE`.
