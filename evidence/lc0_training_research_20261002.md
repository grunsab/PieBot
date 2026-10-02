# How strong engines train on LCZero data, and what PieBot does about it

Research run 2026-10-02 (100 agents, 18 sources fetched, 89 claims extracted,
25 survived three-vote adversarial verification, none refuted).

## Read this first: what the evidence does and does not cover

- **Only Stockfish and the bullet trainer's tutorial examples are covered.**
  No claim about Obsidian, PlentyChess, Berserk, Viridithas, Stormphrax or any
  other top-20 engine survived verification. This is not a survey of top-20
  practice.
- The Stockfish evidence is from 2021-2024 (nnue-pytorch wiki, linrock's
  network PRs #4635, #4782, #5254, robotmoon.com). Current Stockfish nets are
  documented in `threats.yaml` in vondele/nettest, which was not read.
- **No source isolates the Elo of any single ingredient.** Every Stockfish
  recipe was validated as a bundle, by games. Several rationales are folklore
  in the sources themselves (the capture filter is "generally known ... for
  some reason").
- Everything was measured on Stockfish's HalfKAv2_hm multi-layer net or
  bullet's 768-style inputs. Transfer to PieBot's HalfKP -> 1024 -> 1 net is
  inference; each change needs its own game test.

## Findings against PieBot

| Stockfish / bullet practice | PieBot before | Status |
| --- | --- | --- |
| Lc0 retrain at lr 4.375e-4, x0.995 per epoch, starting from an existing good net | constant 1e-3 | In lineage v2 (commit `06eb368`) |
| Train first on own-engine data, then retrain on Lc0 data; Lc0-only from scratch "doesn't produce as good results" | self-play v8 net, then Lc0 | Already the case |
| Nets accepted by games only: 25k-node pre-screen, then SPRT over 54k-320k games | 400-game screen at 150 ms | Gate moved to 1000 ms in v2; sample size unchanged |
| Skip positions where the side to move is in check | none | Option added (`--skip-in-check`), off by default |
| Skip positions whose best move is a capture | none | Approximated (`--skip-before-capture`), off by default |
| Skip early plies (12 rising to 28 across retrains) | none | Option added (`--skip-early-plies`), off by default |
| Result weight ends at ~30% (lambda 1.0 -> 0.7; 0.8 -> 0.7 in 2024) | fixed 20% | Option added (`--teacher-mix`), off by default; no schedule |
| Lc0 pool spans several runs (T60, T77-T80), grown monthly | test91 only, 2 months | v2 widens to 6 months; still one run |
| Output buckets (8, by material) before wider nets or more layers | none | Not implemented, see below |
| Stochastic "WLD skipping" by win-rate model | none | Not implemented: needs PieBot's own score-to-WDL calibration |
| Tablebase rescoring of Lc0 data | none | Not implemented: no Syzygy support yet |
| Deduplication of positions | games only | Not implemented |
| First layer 1536 -> 3072 wide (Stockfish 2023-2024) | 1024 | Not implemented |

Shipped net is a selected intermediate epoch in Stockfish, not the last one.
PieBot already tests its validation-best checkpoint rather than the latest.

## What was implemented (commit `12b7a79`)

`training/nnue/lc0_filter.py` plus options on `lc0_autopilot` and `lc0_deploy`.
Filtering happens when a chunk is expanded for training, so the frozen corpus
and the fixed validation file are untouched.

Measured on 200,000 real corpus rows (box, 2026-10-02):

| Filter | Rows dropped | Cost per 700k-row chunk |
| --- | --- | --- |
| early plies < 16 | 13.0% | 4.2 s |
| in check | 6.2% | 11.1 s |
| before a capture | 16.6% | 5.6 s |
| all three | 33.4% | 11.8 s |

Two limits to keep in mind:

- **The capture filter is a proxy.** Corpus rows carry no best move, so the
  filter drops a position when the move actually played was a capture (the
  next row of the same game has fewer pieces). Lc0 training games use
  temperature, so the played move is not always the best move. A true
  best-move filter needs the corpus rebuilt from raw archives, which were
  evicted.
- **Changing `--teacher-mix` changes the loss scale.** Validation loss under
  0.7 is not comparable with the 0.6201 history under 0.8. The filters do not
  have this problem because validation stays unfiltered.

**None of this has been tested for playing strength.** The options exist; no
lineage uses them. Lineage v2 was launched before this research finished and
runs without them.

## Not implemented, and why

- **Output buckets.** The most promising architectural step: bullet's
  progression adds 8 material-count buckets before input buckets or extra
  layers. It changes the quantised net format and the Rust inference path
  (including the hand-written AVX2/NEON kernels), so it is a new lineage and
  an engine change that must pass the mate suites and a paired game test. A
  bucketed net can start from the current weights with the output layer
  copied into every bucket, which reproduces the current evaluation exactly,
  so the engine side can be verified bit-for-bit before any training.
- **A gate that resolves small gains.** Stockfish's retrains gain 2-5 Elo
  each and need tens of thousands of games. PieBot's 400 + 1000 games cannot
  see that. It has not mattered so far because the differences at stake were
  tens to hundreds of Elo, but it will once the easy gains are taken.

## Suggested order of test

One change per arm, judged by games at 1000 ms or longer, not by loss:

1. Filters on (early plies, in-check, before-capture) against v2 unfiltered.
2. Result weight 0.7 against 0.8.
3. Output buckets.

## Sources

- https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets
- https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py)
- https://github.com/official-stockfish/Stockfish/pull/4635
- https://github.com/official-stockfish/Stockfish/pull/4782
- https://github.com/official-stockfish/Stockfish/pull/5254
- https://robotmoon.com/nnue-training-data/
- https://github.com/jw1912/bullet/tree/main/examples/progression
