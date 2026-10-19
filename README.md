# PieBot

PieBot is a UCI chess engine written in Rust. It combines a Stockfish-style
alpha-beta search with an NNUE evaluation trained on Leela Chess Zero game
data, and includes the tooling used to train its networks and measure every
change in games.

## Strength

PieBot has no entry on an official rating list (CCRL, CEGT or similar). The
figures below come from this repository's own matches, played with
[`scripts/experiments/uci_match.py`](scripts/experiments/uci_match.py): one
search thread per engine, openings from `books/openings_v1.fen` played twice
with colours reversed, and draws only on a threefold repetition that actually
occurred, the fifty-move rule, stalemate or insufficient material.

**Against full-strength Stockfish 19** (60+0.5, 280 games, Syzygy 3-4-5,
`Contempt 40`, Ryzen 9 7940HX):

| PieBot build | W / D / L | Win or draw |
| :--- | :---: | ---: |
| Previous search (August 2026) | 0 / 2 / 138 (140 games) | 1.4% |
| October 2026 search, SPSA-tuned, arch-v2 net | 0 / 54 / 226 | 19.3% |
| October 2026 search, arch-v3 net (`v3_lc0t91_b16`) | 0 / 92 / 188 | 32.9% |

**Self-play progression** (Elo with 95% confidence intervals):

| Change | Games | Time control | Elo |
| :--- | ---: | :--- | :--- |
| Search rewrite and NNUE speed-ups, against the previous search | 400 | 10+0.1 | +401.9 [+363.8, +448.5] |
| SPSA tuning, round 1 | 1000 | 10+0.1 | +95.8 [+82.4, +109.6] |
| SPSA tuning, round 2 | 1000 | 10+0.1 | +16.7 [+5.0, +28.4] |
| Arch-v3 net against the arch-v2 net, same binary | 1000 | 10+0.1 | +95.1 [+80.1, +110.3] |

Self-play gains are measured engine against engine and do not translate
one-for-one into gains against other opponents. Full tables:
[evidence/search_v2_20261016](evidence/search_v2_20261016/README.md) and
[documents/NNUE.md](documents/NNUE.md).

## Features

- **Search:** iterative deepening with aspiration windows and PVS; a lockless
  transposition table that also caches static evaluations; a staged move
  picker with butterfly, capture, pawn and continuation histories; correction
  histories; null move, ProbCut, razoring, futility, late-move and SEE pruning;
  logarithmic late-move reductions; singular extensions; quiescence search;
  cuckoo-table detection of upcoming repetitions. See
  [documents/SEARCH.md](documents/SEARCH.md).
- **Evaluation (arch-v3):** king-bucketed inputs (16 or 32 buckets, mirrored
  horizontally, kings included as pieces), 8 output buckets chosen by piece
  count, and an optional mixture of material experts on the input layer.
  Accumulators are updated incrementally and lazily, with a refresh cache for
  king-bucket changes. The arch-v2 HalfKP net is still supported. See
  [documents/NNUE.md](documents/NNUE.md).
- **Endgames:** Syzygy WDL/DTZ probing (3-4-5 men tested).
- **Parallel search:** Lazy SMP over a shared transposition table.
- **Contempt:** a configurable draw score, useful against stronger opponents.
- **Tuning:** 55 search constants exposed as runtime parameters and tuned by
  SPSA.

## Getting started

Requirements: a Rust toolchain, 1.88 or newer ([rustup.rs](https://rustup.rs/)).

```bash
git clone https://github.com/grunsab/PieBot.git
cd PieBot/PieBot
cargo build --release --bin uci
```

Build from inside `PieBot/`. `PieBot/.cargo/config.toml` sets
`-C target-cpu=native`, and cargo reads it only from the working directory: a
build started from the repository root with `--manifest-path` produces a
slower, generic binary. The engine is `PieBot/target/release/uci`.

At start-up the engine looks for `models/lc0_chunk_00034965.nnue` **relative to
its working directory**. If the file is not found, it plays on a simple
piece-square evaluation without warning. When starting the engine from
anywhere else (a GUI, a script), set `UseNNUE` to `true` and `NNUEQuantFile` to
an absolute path.

```text
uci
id name PieBot NNUE
...
uciok
setoption name NNUEQuantFile value /path/to/PieBot/models/lc0_chunk_00034965.nnue
setoption name UseNNUE value true
setoption name Hash value 256
isready
readyok
position startpos moves e2e4 e7e5
go wtime 60000 btime 60000 winc 500 binc 500
info depth 18 ... pv g1f3 b8c6 ...
bestmove g1f3
quit
```

To use PieBot in a GUI (Cute Chess, Arena, Banksia, Nibbler, En Croissant),
add `PieBot/target/release/uci` as a UCI engine and set `NNUEQuantFile` to the
absolute path of a network. No other configuration is needed.

## UCI options

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `Threads` | spin, 1-512 | 1 | Lazy SMP search threads, including the main thread. |
| `Hash` | spin, 1-16384 | 64 | Transposition table size in MiB. Large sizes allocate quickly (lazily zeroed pages). |
| `UseNNUE` | check | true | Use the network evaluation. |
| `NNUEQuantFile` | string | `models/lc0_chunk_00034965.nnue` | Quantised network: arch-v2 (`PIENNQ02`) or arch-v3 (`PIENNQ03`). Loading a net clears the hash. |
| `NNUEFile` | string | empty | Dense f32 network (`PIENNUE1`), for development only. |
| `EvalBlend` | spin, 0-100 | 100 | Percentage weight of the network against the piece-square evaluation. |
| `Contempt` | spin, -200-200 | 0 | Draw score in centipawns for the searching side. Positive values avoid losing lines by preferring draws; 40 helped against Stockfish 19. Leave at 0 for normal play. |
| `SyzygyPath` | string | empty | Syzygy table directories, separated by `;` or `:`. |

With `PIEBOT_TUNE=1` in the engine's environment, the 55 search constants are
also advertised as spin options (they are accepted, but hidden, without it).

With a clock, the engine treats its time allocation as a hard limit and aims to
finish well inside it, adjusting by best-move stability and score trend.
`go movetime N` uses the whole budget.

## Networks

Networks live in [`models/`](models/), with hashes in
[`models/MANIFEST.json`](models/MANIFEST.json).

| File | Format | Notes |
| :--- | :--- | :--- |
| `lc0_chunk_00034965.nnue` | arch-v2 (`PIENNQ02`), HalfKP, 1024 wide, 84 MB | Current default. Trained on LCZero data. |
| `v3_lc0t91_b16.nnue` | arch-v3 (`PIENNQ03`), 16 king buckets, 8 output buckets, 1024 wide, 25 MB | Trained on 725M filtered LCZero test91 positions. |
| `v8_cycle_000013_quant.nnue` | arch-v2 (`PIENNQ02`) | Earlier self-play network (August 2026). |
| `cycle_000098_quant.nnue` | arch-v1 (`PIENNQ01`) | Historical; kept because benchmarking scripts read it by path. |

To compare two networks quickly with the same search, the in-process
`compare_play` runner takes an evaluation and a network per side (game-level
measurements use `uci_match.py`; see [DEVELOPMENT.md](documents/DEVELOPMENT.md)):

```bash
cd PieBot
cargo run --locked --release --bin compare_play -- --games 40 --movetime 200 --same-search \
  --base-eval nnue --base-nnue-quant-file ../models/lc0_chunk_00034965.nnue \
  --exp-eval nnue --exp-nnue-quant-file ../models/v8_cycle_000013_quant.nnue
```

Earlier versions of this README gave rating-list-style Elo figures (for example
"3203.8") for these networks. They came from an older match harness whose draw
adjudication was later found to be wrong, and they are not comparable with the
figures above.

## Documentation

- [documents/SEARCH.md](documents/SEARCH.md): search design and measured experiments.
- [documents/NNUE.md](documents/NNUE.md): network architectures, file formats, training pipeline and results.
- [documents/DEVELOPMENT.md](documents/DEVELOPMENT.md): building, testing, A/B testing, tuning and measurement protocols.
- [documents/history/](documents/history/): earlier plans and campaign notes, kept for reference.
- [evidence/](evidence/): match results and measurement records behind the numbers above.

## Credits

**This project is AI-generated.** Almost all of its code, tests, training
pipeline and documentation were written by AI coding agents (most recently
Anthropic's Claude, through Claude Code), with almost no code added by human
authors. The human authors set the goals, made the decisions and supplied the
hardware.

PieBot builds on the work of many people. Ideas we studied and reimplemented,
and the libraries, data and tools we use:

- **[Triumviratus](https://github.com/Tors3/Triumviratus)** by Francesco
  Torsello, in collaboration with Maurizio Platino: the idea of a mixture of
  experts on the NNUE input layer, chosen by material (the "Consilium"
  network). Our implementation was written independently from the
  description in its README.
- **[Stockfish](https://github.com/official-stockfish/Stockfish)**: the search
  follows its structure and many of its techniques (staged move picking,
  butterfly, continuation, capture and pawn histories, correction history,
  singular extensions, ProbCut, the LMR adjustments, upcoming-repetition
  detection); NNUE design ideas such as king buckets, output buckets and
  refresh caches; and the fishtest formulation of SPSA. Stockfish 19 is also
  the strength anchor PieBot is measured against.
- **NNUE** was introduced by Yu Nasu for computer shogi (2018).
- **Upcoming-repetition detection** with cuckoo tables is due to Marcel van
  Kervinck.
- **[Leela Chess Zero](https://lczero.org/)**: the training data (test91
  archives, available under the Open Database License and the Database
  Contents License) and the networks that produced it.
- The wider open-source engine community, including **Ethereal**,
  **Berserk**, **Koivisto**, **Reckless** and **Stormphrax**, for widely shared
  techniques such as correction-history weighting and accumulator refresh
  caches, and the **[Chess Programming Wiki](https://www.chessprogramming.org/)**.
- **[cozy-chess](https://github.com/analog-hors/cozy-chess)** (move generation
  and board representation); **[shakmaty](https://github.com/niklasf/shakmaty)**,
  **[shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy)** (tablebase
  probing) and **[python-chess](https://github.com/niklasf/python-chess)**
  (match tooling and data conversion), all three by Niklas Fiekas.
- **Syzygy tablebases** by Ronald de Man, downloaded from the
  [lichess](https://tablebase.lichess.ovh/) mirror.
- **[PyTorch](https://pytorch.org/)** and **NumPy** for training,
  **[rayon](https://github.com/rayon-rs/rayon)** and the Rust ecosystem for the
  engine, and **[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)**
  for playing on Lichess.
- GPU time was rented on **[Vast.ai](https://vast.ai/)**.

## License

AGPL-3.0. See [`PieBot/LICENSE`](PieBot/LICENSE).
