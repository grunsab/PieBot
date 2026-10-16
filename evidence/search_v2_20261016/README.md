# Search v2: rewrite, NNUE speedups, tuning (2026-10-15/16)

Machine: Ryzen 9 7940HX laptop (16 cores), 14 cores used for testing. Net
`lc0_chunk_00034965.nnue` at EvalBlend 100 on every PieBot side. One search
thread and 64 MB hash per engine unless stated. Openings from
`books/openings_v1.fen`, each played twice with colours reversed. Draws only on
a threefold that actually occurred, the fifty-move rule, stalemate or
insufficient material. Matches run with `scripts/experiments/uci_match.py`; the
JSON files here keep each match's config and summary (games are in `out/`,
not committed).

## Self-play A/B (10+0.1)

| Change | Games | W / D / L | Elo [95% CI] | File |
| :--- | ---: | :---: | :--- | :--- |
| v1: search rewrite vs main `5860faf` | 400 | 191 / 152 / 57 | +121.1 [+95.3, +148.2] | `ab_v1_match_10s.json` |
| v2: v1 + NNUE speedups vs main | 400 | 335 / 58 / 7 | +401.9 [+363.8, +448.5] | `ab_v2_match_10s.json` |
| SPSA round 1 values vs previous defaults | 1000 | 350 / 569 / 81 | +95.8 [+82.4, +109.6] | `abq_spsa1.json` |
| ThreatEscape/Enter = 50 | 600 | 76 / 412 / 112 | -20.9 [-36.1, -5.7] | `abq_threat50.json` |
| ThreatEscape/Enter = 10 | 600 | 103 / 401 / 96 | +4.1 [-11.2, +19.3] | `abq_threat10.json` |
| QsChecks = 1 | 600 | 114 / 373 / 113 | +0.6 [-15.5, +16.6] | `abq_qschecks.json` |
| ContSplit = 1 | 600 | 104 / 400 / 96 | +4.6 [-10.1, +19.4] | `abq_contsplit.json` |
| CheckExt = 2 (SEE-guarded) | 600 | 84 / 408 / 108 | -13.9 [-29.1, +1.3] | `abq_checkext2.json` |
| CheckExt = 0 (no check extension) | 600 | 101 / 426 / 73 | +16.2 [+2.1, +30.4] | `abq_checkext0.json` |
| CheckExt = 0, confirmation | 1000 | 180 / 655 / 165 | +5.2 [-6.4, +16.9] | `abq_checkext0_confirm.json` |
| CheckExt = 0, both runs pooled | 1600 | | +9.3 [+0.3, +18.4] | adopted as default |
| UpcomingRep = 1 (cuckoo repetition detection) | 1000 | 188 / 655 / 157 | +10.8 [-1.5, +23.1] | `abq_upcomingrep.json`; adopted |

The toggle experiments ran on the tuned build (`999cc11`) against itself with
default options. SPSA: 6000 game pairs at 6+0.06 over 55 constants;
`spsa1_tuned.json` holds the adopted values.

## Speed (nodes/s, one thread, five positions, idle machine)

| Build | Mean | Note |
| :--- | ---: | :--- |
| main `5860faf` | ~1.6M | previous search |
| first rewrite build | 567k | scored every move; full refresh on king moves |
| + staged picker, refresh cache, fused updates, plain-loop kernels | ~2.0M | AVX2 intrinsics replaced by loops vectorised under target-cpu=native |
| + TT prefetch | 2.19M | |
| + lazy accumulators | 2.60M | middlegame -1..-5%, K+P endgame +58% |

## Against Stockfish 19 (full strength, one thread each)

| PieBot | TC | Games | W / D / L | Win or draw | File |
| :--- | :--- | ---: | :---: | ---: | :--- |
| main `5860faf` | 60+0.5 | 140 | 0 / 2 / 138 | 1.4% | `sf19_baseline_main_60s.json` |
| v2 | 60+0.5 | 140 | 0 / 20 / 120 | 14.3% | `sf19_v2_match.json` |
| v5 (SPSA-tuned, check extension on), Syzygy, Contempt 40 | 60+0.5 | 280 | 0 / 54 / 226 | 19.3% | `sf19_v5_match_60s.json` |
| v3 (v2 + lazy NNUE), no TB, Contempt 0 | 20+0.2 | 280 | 0 / 26 / 254 | 9.3% | `sf19_v3_notb.json` |
| v3, Syzygy 3-4-5 | 20+0.2 | 280 | 0 / 29 / 251 | 10.4% | `sf19_v3_tb.json` |
| v3, Syzygy, Contempt 40 | 20+0.2 | 280 | 0 / 42 / 238 | 15.0% | `sf19_v3_tb_c40.json` |
| v3, Syzygy, Contempt 80 | 20+0.2 | 280 | 0 / 40 / 240 | 14.3% | `sf19_v3_tb_c80.json` |
| v3, Syzygy, Contempt 150 | 20+0.2 | 280 | 0 / 39 / 241 | 13.9% | `sf19_v3_tb_c150.json` |

The three 20+0.2 configurations at Contempt 0/40 used the same openings and
colours; paired by opening, Contempt 40 added +0.093 win-or-draw points per
pair (se 0.052) and tablebases +0.021 (se 0.041). The baseline row was played
with `scripts/uci_elo_arena.py` (`--draw-rule occurred`), the rest with
`uci_match.py`.

## Where the losses happen

`loss_points_c40.json`: for each of the 238 losses in the Contempt 40 run,
Stockfish 19 at depth 12 after every PieBot move. The first move that took
PieBot below -150 cp came at median ply 32, with a median of 62 points of
material still on the board and a median swing of 74 cp; only 9% of those
moves swung 200 cp or more. PieBot loses by gradual middlegame drift rather
than tactical blunders, which points at evaluation quality more than search.

## Not measured

- Elo with more than one thread for the new search (Lazy SMP is carried over
  and its tests pass).
- Anything at the CCRL 40/15 time control.
