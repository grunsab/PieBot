# Search

The search lives in `PieBot/src/search/alphabeta.rs`. It was rewritten in
October 2026 along the lines of Stockfish and replaced the previous search
after a +401.9 Elo [+363.8, +448.5] self-play result (400 games, 10+0.1,
together with the NNUE speed-ups described in [NNUE.md](NNUE.md)).

## Design

**Transposition table.** Lockless, 4-way 64-byte buckets of packed 16-byte
entries (move, score, static evaluation, depth, bound, PV flag, age), with
XOR-verified keys and age-aware replacement. The entry for a child position is
prefetched as soon as a move is made. Because the table caches static
evaluations, anything that changes the evaluation (loading a network, changing
`EvalBlend`) clears it.

**Move ordering.** A staged picker: the TT move is tried before any
generation; then winning captures ordered by MVV, capture history and SEE; two
killers and the counter move; quiet moves, scored only when reached, from
butterfly, pawn and continuation histories (1, 2, 4 and 6 plies back) plus a
bonus for direct checks; losing captures last.

**Evaluation corrections.** Pawn-structure, non-pawn (per side), minor-piece
and continuation correction histories adjust the static evaluation by the
error the search has observed for similar positions. The evaluation is pulled
slightly toward zero as the fifty-move counter rises.

**Pruning and reductions.** PVS with aspiration windows, mate-distance
pruning, razoring, reverse futility pruning, null-move pruning (verified at
high depth), ProbCut, internal iterative reduction, late-move pruning,
futility and history pruning, and SEE pruning of quiets and captures.
Late-move reductions are logarithmic in depth and move number, adjusted for
PV and cut nodes, history, improving positions, checks and cutoff counts, with
deeper or shallower re-searches.

**Extensions.** Singular extensions, including double extensions, multi-cut
and negative extensions. Checks are not extended: `CheckExt=0` measured +9.3
Elo [+0.3, +18.4] over 1600 games against extending every check.

**Quiescence.** TT probes and stores, stand-pat from the corrected evaluation,
futility and SEE filters, and full evasions when in check.

**Draws.** A position that repeats inside the search tree scores as a draw at
once; a repetition of a position from before the root needs a real threefold.
Upcoming repetitions are detected one move early with cuckoo tables
(`search/cuckoo.rs`, built from cozy-chess's Zobrist constants). The draw
score is set by `Contempt`.

**Endgames.** With `SyzygyPath` set, WDL tables are probed after captures and
pawn moves inside the search; at the root the engine plays the DTZ-best move.

**Time management.** With a clock, the allocation from the UCI front end is a
hard limit. The search aims for a soft target (`TmSoft`, 47% of the
allocation by default) and stretches or shrinks it by best-move stability,
the share of nodes spent on the best move, and a falling score.

**Parallel search.** Lazy SMP: helper threads run independent iterative
deepening over the shared transposition table, and only the main thread's
result is used. Published strength figures are single-threaded; the
multi-threaded Elo of this search has not been measured.

**Tunables.** 55 constants are runtime parameters (`tunables!` and
`tp(T::...)` in `alphabeta.rs`). With `PIEBOT_TUNE=1` they are advertised as
UCI spin options, which lets one binary be tested against itself with
different settings and lets `scripts/experiments/spsa.py` tune them. The
defaults come from three SPSA rounds: 6000 game pairs at 6+0.06, then 8000 at
8+0.08, then 6000 at 8+0.08 with the `v3pw_bt4_1500` net. Experiment toggles are tunables too, and their defaults are the
measured best.

## Measured experiments

Self-play at 10+0.1, one thread, `books/openings_v1.fen`, paired openings.
Toggle experiments ran on the tuned build against itself with default options.
Source files are in
[evidence/search_v2_20261016](../evidence/search_v2_20261016/README.md).

| Change | Games | Elo [95% CI] | Outcome |
| :--- | ---: | :--- | :--- |
| Search rewrite (v1) vs previous search | 400 | +121.1 [+95.3, +148.2] | adopted |
| v1 + NNUE speed-ups (v2) vs previous search | 400 | +401.9 [+363.8, +448.5] | adopted |
| SPSA round 1 | 1000 | +95.8 [+82.4, +109.6] | adopted |
| SPSA round 2 | 1000 | +16.7 [+5.0, +28.4] | adopted |
| SPSA round 3, retuned for `v3pw_bt4_1500` | 1000 | +10.8 [+2.4, +19.1] | adopted |
| SPSA round 3 at 60+0.6 | 400 | +11.3 [+1.3, +21.3] | confirms round 3 |
| No check extension (`CheckExt=0`), two runs pooled | 1600 | +9.3 [+0.3, +18.4] | adopted |
| Cuckoo upcoming-repetition detection (`UpcomingRep=1`) | 1000 | +10.8 [-1.5, +23.1] | adopted |
| SEE-guarded check extension (`CheckExt=2`) | 600 | -13.9 [-29.1, +1.3] | rejected |
| Threat-aware quiet ordering (`ThreatEscape/Enter=50`) | 600 | -20.9 [-36.1, -5.7] | rejected |
| Threat-aware quiet ordering (`ThreatEscape/Enter=10`) | 600 | +4.1 [-11.2, +19.3] | not adopted |
| Quiet checks in the first quiescence ply (`QsChecks=1`) | 600 | +0.6 [-15.5, +16.6] | not adopted |
| Continuation history split by check/capture (`ContSplit=1`) | 600 | +4.6 [-10.1, +19.4] | not adopted |
| `Hindsight=1` (stopped after 420 of 1000 games) | 420 | -2.5 [-20.4, +15.4] | inconclusive, off |

Speed on a Ryzen 9 7940HX, one thread, arch-v2 net: the previous search ran at
about 1.6M nodes/s; the rewrite started at 567k and reached about 2.6M after
the staged picker, refresh cache, fused updates, TT prefetch and lazy
accumulators.

Against Stockfish 19, losses are mostly gradual: in the 238 losses of one
20+0.2 run, the first move that took PieBot below -150 cp (judged by
Stockfish at depth 12) came at median ply 32 with a median swing of 74 cp, and
only 9% of those moves swung 200 cp or more. This points at evaluation
quality more than search, and is why later work moved to the network.

## Mate-suite gate

The search is selective, so some mates that a fixed depth-7 search used to
find appear a few plies later. The suite is read with depth escalation and
must solve 91/91 by depth 14:

```bash
cd PieBot
PIEBOT_SUITE_FILE=src/suites/matein3.txt PIEBOT_TEST_THREADS=1 \
  PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=14 \
  cargo run --locked --release --bin accept
```

The run is deterministic, so its total node count is a signature of the
build: it changes whenever the search or its tuned defaults change. Read the
expected value from the current build rather than from documentation, and use
it to confirm that a rebuild on another machine is identical.
