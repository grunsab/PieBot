# S110: a repetition inside the search is a draw at once

2026-10-14. The main search now scores a position as drawn when it already
occurred on the current search path (the root included, never across a null
move). Before, only a true threefold counted, so the second occurrence of a
position was answered from the transposition table with that position's own
earlier score.

## Why

Found while analysing the Lichess loss https://lichess.org/FduYSIGx (PieBot
lost as Black to a 3025 bot). On 36...d3 the search chose a move whose line
cycled (Re7 Qg5 Re8 Qd5 ...): the score stayed at exactly -522 for four
iterations that cost almost nothing, then depth 21 took 144M nodes and changed
its mind. Stockfish: 36...d3 loses by mate, 36...Qf7 is -5.7.

Mechanism, reproducible with the real net: in
`6k1/6p1/5p2/7Q/8/1r6/r1q3PP/7K w - - 0 1` White's only resource is the
perpetual Qe8+ Kh7 Qh5+ Kg8. The old search scores it -710 at every depth from
2 to 22 (and the Lichess build `7bd8cb7` -710 at depth 16); the fixed search
scores 0 from depth 2. Tests: `PieBot/tests/repetition_in_tree.rs` and the two
`draw_policy_regressions` unit tests in `alphabeta.rs`.

On Lichess, 223 of PieBot's 244 draws against bots between 1 and 14 Oct were
threefolds, and in 70 of them Stockfish had PieBot at +2 or better within the
last 40 plies. Replaying PieBot's side of those games with search state carried
between moves and the live server's node budget (225k nodes/s in the two-thread
era, 330k in the three-thread era, times the clock time really used), moves
made at +2 or better that let the opponent complete a threefold:

| Engine | Count (69 games) |
| --- | --- |
| Live games (`7bd8cb7`) | 36 |
| Main before this change | 23 |
| This change | 5 |
| This change plus S114 (in testing) | 0 |

## Results

Head-to-head, Mac, 1000 ms per move, 6 games at a time, one thread, net
`lc0_chunk_00034965` at blend 100 on both sides, paired openings:

| Run | Games | W / D / L for the fix | Elo | Pair-delta 95% CI |
| --- | --- | --- | --- | --- |
| Screen, seed 20261013 | 400 | 168 / 97 / 135 | +28.7 | [+0.075, +0.255] |
| Confirmation, seed 20261015 | 1000 | 415 / 254 / 331 | +29.3 | [+0.112, +0.224] |

In the confirmation's 254 draws the fixed engine was objectively lost in 67
final positions and winning in 15 (Stockfish 300k nodes, 2-pawn threshold), so
the gain comes mostly from saving lost games by repetition. Depth 15.26
against 15.34; nodes per second 1.5% lower (the extra repetition test).

Stockfish 18 at UCI_Elo 3000, 60+0.5, one thread each, same three opening
seeds and colours for both builds, 102 games each:

| Draw rule | Main | Fix |
| --- | --- | --- |
| Old arena rule (draw when a threefold is claimable) | 43.0 (14/58/30) | 43.0 (23/40/39) |
| Real threefold only (`--draw-rule occurred`) | 36.0 (16/40/46) | 39.5 (22/35/45) |

Paired difference under the real rule: +0.034 points per game, 95% CI
[-0.054, +0.123]. The old rule ended all 94 of its "threefold" games before
any position had occurred three times, which rewarded the old search's habit
of walking into repeatable positions; see the arena note below.

matein3 at depth 7, one thread: 91/91 for both; `accept` 7040853 -> 7087288.

## Shelved alongside

- Drawn scores stored for move ordering only: +17.4 Elo on the same 400
  openings, but it let 12 won games go to repetition against 6. Not used.
- Root search hygiene (previous best first, keep a proven better move from an
  interrupted iteration): -10.4 Elo [-27, +6], no depth gained in real games.

## Arena draw rule

`scripts/uci_elo_arena.py` (and therefore `uci_elo_ladder.py`) has ended games
with `board.outcome(claim_draw=True)` since it was written on 2026-08-03: a
game is drawn as soon as the side to move could claim, whether or not it would
repeat. All 358 threefold draws in the older local results checked
(`out/thread_anchor_20260929/`, `evidence/stockfish_30_arena*.json`) were
claim-only, 195 with Stockfish to move. `--draw-rule occurred` now ends games
only on a real threefold or the fifty-move count; the default is unchanged so
existing result files still resume. Earlier ladder draw rates, and the reading
that limited Stockfish "repeats instead of mating", should be revisited with it.

## Not shown

- Nothing here is a Lichess rating change. The bot was moved to `67081c6`
  (this change) on 2026-10-14 at 22:19:49 UTC; see
  `evidence/lichess_deploy_67081c6_20261014/`.
- The Stockfish comparison is 102 games per build and does not resolve a
  difference of this size.
