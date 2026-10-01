# PieBot Lichess audit — October 1, 2026

**The strongest identified explanation is a correctness defect in the old two-thread root-splitting search still deployed on the Lichess server.** It can select a move whose search returned only an upper bound, while reporting the winning score established by another move. The game records contain the corresponding symptom: confident positive evaluations followed by a hanging queen, an undefended piece, or mate in one.

The trained NNUE is enabled and its file is correct. The previously completed Lazy SMP replacement removes the defective root-result collection path, but that replacement has not been deployed to this server. Slower VPS hardware is a secondary handicap; rating scales also differ. Neither makes the recorded one-move blunders normal.

## Download and results

Exported all 186 completed games returned by the authenticated Lichess games API for **September 29, 2026 22:13:40 UTC through October 1, 2026 22:13:40 UTC**: September 29 at 3:13:40 p.m. through October 1 at 3:13:40 p.m. Pacific. The filter uses game creation time. All games are rated standard chess against bots. The export includes moves, clocks, available existing analysis, openings and PGN. The API token is not saved in the evidence.

| Pool | Games | Wins | Draws | Losses | Score | Rating at first game → after last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bullet | 81 | 33 | 21 | 27 | 53.7% | 2334 → 2346 |
| Blitz | 105 | 40 | 24 | 41 | 49.5% | 2408 → 2291 |
| Total | 186 | 73 | 45 | 68 | 51.3% | Separate rating pools |

Most games were 120+1 or 120+2. All 68 losses ended in checkmate. None was a timeout loss. Median PieBot clock remaining at the last recorded move in its losses was 29.11 seconds; the minimum was 11.54 seconds.

Files: [complete PGN](games.pgn), [original API export](games.ndjson), [download window and hashes](window.json), [per-game summary](game_summary.json).

## Concrete failures in the games

These moves are confirmed by both the public game and the server's actual UCI `bestmove` log. Live scores below are from PieBot's perspective. Stockfish values are independent full-strength analyses, also from PieBot's perspective, using two million nodes for the best-move search and two million for the played move restricted at the same root. Scores from different engines need not have identical calibration; immediate lost material and checkmate do not depend on that calibration.

| Game | PieBot move | Live depth / evaluation | Immediate problem | Clock before move |
| --- | --- | --- | --- | ---: |
| [MrTonnerre](https://lichess.org/zQoaxfK6#3) | 2.Bh6 | 15 / +1.26 | ...Nxh6 takes the bishop; SF changes from +0.66 to −4.38 | 120.03 s |
| [Zagreus_Engine](https://lichess.org/sEi3c6E4#43) | 22.h6 | 15 / +3.50 | ...Bb4#; Be2 instead keeps an advantage | 66.19 s |
| [entebot](https://lichess.org/aGvveGBK#66) | 33...Qf4 | 16 / +6.74 | Qxf4 takes the queen; SF changes from +4.39 to −8.35 | 48.99 s |
| [Blobfish-Bot](https://lichess.org/U0E0YGi4#61) | 31.Qh3 | 15 / +3.71 | ...Bxh3 takes the queen; SF changes from +2.05 to −6.84 | 59.39 s |
| [uSunfish](https://lichess.org/C3d1f8t5#37) | 19.Qc4 | 12 / +3.53 | ...dxc4 takes the queen; SF changes from +3.58 to −5.69 | 76.43 s |

Screened all 20,847 positions across the 186 games with Stockfish 18 at full strength, 50,000 nodes per position. Then refined one selected turning point from each of the 68 losses with the larger budgets above. **45 of those 68 positions lose at least 200 centipawns compared with the best move; 29 of these were reported as favorable by the live engine.** In 41 cases, the refined best move was at least −1.50 while the played move was at most −2.00, with a loss of at least 1.50. These are sampled turning points, not a claim that every loss has a single cause. Screening evaluations are approximate and are not used as training labels.

## Confirmed parallel-search defect

The deployed search source SHA-256 is `57036a6221b476da8f3f24074b7794b7be30423c78a07004a0f72220bb0674ae`, matching the saved pre-Lazy-SMP source. The relevant function is `search_depth_parallel` in the archived `remote_PieBot_src_search_alphabeta.rs`:

1. Lines 1136–1163 search each tail move against its own snapshot of a shared alpha threshold. If the move fails to exceed that threshold, its result is only an upper bound on the root move's true value.
2. Line 1184 returns the numerical score but discards that bound information and the alpha snapshot.
3. Lines 1190–1199 collect results in move order and choose the largest number. A fail-low bound can tie a good move's exact score and occur earlier in this collection. The strict `>` comparison keeps the earlier, unproven move.
4. Lines 1209–1216 can store the combined score and wrong move as an exact root entry because they compare against the original window, not each worker's search window.

For example: a sound move establishes +3.89. Another move proves only “at most +3.89,” possibly through a cutoff without needing to find its full refutation. If the latter occurs first in result order, the collector can choose it and reject the genuinely established +3.89 move as a tie. This is a bound-handling error, not merely redundant parallel work or insufficient depth.

A local diagnostic copy of the archived source recorded the alpha snapshot and whether the elected move was an upper bound; it did not change search or selection logic. Across the 68 selected game positions, it observed **27 completed root iterations electing an upper-bound move in 21 positions**, including two final-depth selections. A recorded depth-12 example from the uSunfish position chose `f1b1` with score 389 and alpha 389; `f3e1` established that score with alpha 350 and was later discarded as a tie. Full traces and source are in `bound_probe/` in the reproduction archive.

This reproduces the faulty mechanism. It does not prove that all 45 large losses, or a specified Elo deficit, are attributable to that mechanism. Fresh searches and abbreviated game-history replays generally did not reproduce the five historical blunders exactly. Scheduling, time limits, retained search state and transposition entries affect which move the old collector elects. The live logs nevertheless prove those blunders were returned by the deployed engine, and the diagnosed code can attach a good score to an unproven move.

A relevant existing regression test, `PieBot/tests/root_parallel_quant.rs`, uses depth 3 and a zero-weight model, and checks only equality of scores. The old root-splitting branch starts at depth 4. That test therefore did not exercise this defect or assert the safety of the chosen move.

## Deployment, NNUE and hardware checks

- Server: `vps-5d9b1da8.vps.ovh.ca`; runner PID 3933929 from the September 29 restart, concurrency one game.
- Actual configuration: Threads=2, RAYON_NUM_THREADS=2, Hash=512 MiB, UseNNUE=true, EvalBlend=75, explicit chunk-34965 path, correct repository working directory.
- All 191 engine starts in the log window advertise NNUE enabled; none advertises it disabled. Every startup explicitly receives the model path. The additional starts include games absent from the completed-game export.
- Model SHA-256: `56434c1bacf5165baaebc6a5a06d5542d69828dc7b1ff09a934fb8263ea917d0`, identical to the local and previously rated model.
- Deployed binary SHA-256: `e8c22ae36f0084c26ef1f7f19f2cbedfd0ff21370d05cd5f7aacc85f191be79e`; checkout `a7d470596a0dd18a77a15d70f1ecd070edb55a14`. It still has the old search.
- Matched 10,331 logged move searches to the exact exported game histories and played moves. Aggregate throughput was 225,581 nodes/s; median depth 15; median search time 3.09 s. These node/depth numbers do not certify that the selected root move has the reported score.
- While the bot was idle, five one-thread depth-12 probes on the actual Linux binary matched the Mac's old binary **exactly in best move, score and node count**. Two-thread fresh probes also agreed in move and score, with expected node-count variation. This gives no evidence of a model mismatch or x86-versus-ARM evaluation defect on the tested positions.
- The VPS exposes four Haswell virtual CPUs. The five one-thread probes took roughly 5.3–7.4 times as long as the corresponding Mac probes, including search initialization. This is a small position sample, not an Elo conversion. The production logs also show much lower throughput than the Mac.
- Matchmaking rate-limit errors occur between games; all exported games completed. There were no timeout losses. Pondering is configured but no ponder searches appear in the logs; book and tablebases are disabled. These are secondary operational details, not explanations for throwing away a queen at depth 15.

## Relationship to the 3100–3200 estimate

The earlier remote cohort measurement was a **one-thread** local performance estimate against pinned CCRL-listed opponents at 120+1 on different hardware. It is not an official CCRL placement or a promise of a Lichess rating. The later Mac comparison used the new Lazy SMP build in both arms against a nominal Stockfish UCI_Elo setting, and showed +72.2 Elo for eight threads over one. It did not test the currently deployed old two-thread implementation.

Lichess uses Glicko-2 and explicitly cautions against directly comparing ratings from different player pools: [Lichess rating systems](https://lichess.org/page/rating-systems). The rating-scale distinction matters, but the demonstrated search defect and concrete blunders deserve correction before judging the trained engine's online strength.

## Recommended next action

1. **Immediate mitigation:** run the currently deployed binary with Threads=1 and RAYON_NUM_THREADS=1. That bypasses the faulty parallel-root path and matches the threading mode used for the remote cohort measurement. Its online rating gain has not been measured.
2. **Preferred deployment:** build and deploy the already validated Lazy SMP implementation for Linux, retaining the verified model, blend and working directory. Its main search selects the move; helper scores are not merged by the defective collector. Keep two threads initially on this VPS. Recheck the recorded tactical cases before restarting the runner.
3. Add game-derived regression cases that exercise parallel search at depth at least 4, assert the returned move avoids immediate material loss/mate, and test bound provenance during move selection. Score-only tests are inadequate for this bug.
4. After the corrected deployment, measure a fresh online cohort and a direct old-versus-new comparison before assigning an Elo recovery number. Consider faster hardware after search correctness is established.

The production runner, binary and configuration were not changed during this audit. No additional rated games were initiated. The idle-only server diagnostic completed successfully. Source instrumentation, scripts, complete analysis output and sanitized server logs are archived for review; `findings.json` contains the machine-readable conclusions.

## Follow-up deployment — October 1, 23:20 UTC

The authorized follow-up deployed Lazy SMP with two threads after Linux correctness checks, 42 repeated/history-based trials of six prominent blunders, and replay of the wider 45-position sample. The original six blunders were avoided in all trials, but some replacement moves remain substantial errors. See the [deployment report](../lichess_smp_deploy_20261001/REPORT.md) for results, limits and remaining weaknesses. The earlier audit above describes the original deployment as observed at the time.
