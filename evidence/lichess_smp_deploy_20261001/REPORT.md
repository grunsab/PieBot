# Lazy SMP deployed: historical-blunder verification

**The Linux Lazy SMP candidate was deployed to the Lichess VPS on October 1, 2026 at 23:20:20 UTC (4:20 p.m. Pacific).** One background runner was restarted with `nohup`, Threads=2 and RAYON_NUM_THREADS=2. Its startup check loaded the verified NNUE, accepted the engine configuration, and connected as PieBot. The original executable and a one-thread rollback configuration are preserved.

**The six headline mistakes were avoided in all 42 two-thread trials.** Across the broader 45-position sample, 41 replacement moves fell below the audit's 200-centipawn error threshold. This is strong evidence that the deployed fix addresses the recorded failure pattern. It does not establish that all historical losses would become wins or quantify an online Elo recovery.

## What was tested

Tests used the exact Linux executable deployed on the same OVH VPS, the original chunk-34965 NNUE, blend 75, Hash 512 MiB, and two threads. The bot was stopped between games during the build and CPU-intensive verification. Only `PieBot/src/search/alphabeta.rs` differs from the old production source; the other added files are regression tests and their data.

The old parallel root collector could elect a move with only a fail-low upper bound, tying another move's established score. Lazy SMP removes that collector: helpers share the transposition table, but only the main search supplies the returned move and score. The source-level diagnosis and instrumented confirmation are in the [prior audit](../lichess_audit_20261001/REPORT.md).

Each of six prominent cases received:

- Three fresh searches with the exact recorded UCI clock command and full position history.
- Three fresh searches at its recorded depth, from 12 to 16.
- One search after replaying every preceding PieBot turn with its recorded clock command, retaining search state between turns. There were 117 preceding searches across these six replays.

The retained-state replays feed the actual historical moves even when the new engine would choose differently. They are counterfactual diagnostics, not new games or bit-for-bit reconstructions of the old transposition table. Every returned move was legal, and none allowed immediate opponent checkmate.

## Six prominent cases

| Game / recorded mistake | Original immediate consequence | Retained-state Lazy SMP choice | Avoided original move |
| --- | --- | --- | ---: |
| [sEi3c6E4: 22.h6](https://lichess.org/sEi3c6E4#43) | ...Bb4# | Be2 | 7/7 |
| [zQoaxfK6: 2.Bh6](https://lichess.org/zQoaxfK6#3) | ...Nxh6 wins the bishop | Nf3 | 7/7 |
| [aGvveGBK: 33...Qf4](https://lichess.org/aGvveGBK#66) | Qxf4 wins the queen | ...Qg7 | 7/7 |
| [U0E0YGi4: 31.Qh3](https://lichess.org/U0E0YGi4#61) | ...Bxh3 wins the queen | Bc2 | 7/7 |
| [C3d1f8t5: 19.Qc4](https://lichess.org/C3d1f8t5#37) | ...dxc4 wins the queen | Ne1 | 7/7 |
| [RbqMJLNn: 16...Bg5](https://lichess.org/RbqMJLNn#32) | Bxg5 wins the bishop | ...Re8 | 7/7 |

Full-strength Stockfish 18 independently assessed the root, original mistake, and every distinct replacement at two million nodes apiece, Threads=1, Hash=128 MiB, no tablebases and no strength limit. Searches reset game state. These evaluations are diagnostic evidence only, never training labels.

**Avoiding the original mistake did not always mean selecting a good replacement.** Four of the seven RbqMJLNn trials chose ...Rhd8, losing about 259 cp relative to Stockfish's best move; a ten-million-node check still found a 233 cp loss. The original ...Bg5 lost 546 cp. The other 38 of 42 trial choices were below 200 cp loss. All six retained-state final choices were below that threshold. The C3d1f8t5 alternatives also varied in quality, with losses up to 165 cp.

An additional one-thread Linux control ran all six positions at recorded clocks and recorded depths: 12/12 avoided the original blunders and all were below 200 cp loss. It chose ...Re8 in both RbqMJLNn searches, so this control did not reproduce the weaker ...Rhd8 replacement. The residual error warrants separate search-quality investigation; these probes do not establish a second SMP correctness bug or prove that two threads are always stronger than one.

## Wider sample

All 45 sampled turning points with at least 200 cp loss in the earlier audit were replayed once using their exact clock commands and complete move histories. The candidate changed the move in 42/45 positions. Independent analysis put 41/45 replacement moves below 200 cp loss; one changed move remained a substantial error. None allowed mate in one.

| Remaining case | Candidate choice | Loss versus Stockfish's best |
| --- | --- | ---: |
| [M7txMmIo, ply 30](https://lichess.org/M7txMmIo#30) | ...Na4 | 237 cp |
| [vLiwuMUL, ply 39](https://lichess.org/vLiwuMUL#39) | h4 | 212 cp |
| [vRdCb6Is, ply 81](https://lichess.org/vRdCb6Is#81) | Qe1 | 410 cp |
| [xZ2n1t26, ply 71](https://lichess.org/xZ2n1t26#71) | Nxb6 | 221 cp |

These are selected failure positions, not a representative game sample. Finite-node Stockfish scores are estimates, especially in the queen ending. No claim is made that the fix would win 41 games, eliminate 91% of future blunders, or recover a specified Elo amount.

## Correctness and deployment

- New game-derived regression tests passed on the Mac under default and all-feature builds. They use the real NNUE and two-thread depth-eight search, check legal immediate refutations of the recorded blunders, reject those moves, and reject any choice allowing mate in one.
- Linux verification passed 43 library tests and 24 selected integration tests: Lazy SMP, game-derived regressions, search-core regressions, UCI moves and UCI stop behavior.
- Linux mate-in-three acceptance passed 91/91 at depth seven with one thread and 91/91 with two threads, with no fallbacks.
- Earlier full default/all-feature Rust batteries and both Python suites already passed for this exact engine-source change; see the [Lazy SMP validation report](../lazy_smp_20260929.md). No additional production code was changed during this deployment.
- The initial offline Linux test command could not fetch a missing cached development dependency. Retrying with the pinned lockfile and downloads enabled resolved setup; no test assertion failed. Both attempts are archived.
- The user had explicitly waived the long 400/1,000-game promotion gate. This deployment does not claim to replace that game-level measurement.

Deployment identity:

- Host: `vps-5d9b1da8.vps.ovh.ca`; initial new runner PID `2318383`.
- Binary: `/home/ubuntu/lichess-bot/engines/PieBot-lazy-smp-20261001/PieBot/target/release/uci`.
- Binary SHA-256: `9c3327de9d27074659ece9104142ee236a78c8cd07b5866546d4919535387331`.
- Search-source SHA-256: `d6686e65337edc4bb23b754f1eb40e6a3ceb5f5b12e9c0600ce2b236d8130f97`.
- NNUE SHA-256: `56434c1bacf5165baaebc6a5a06d5542d69828dc7b1ff09a934fb8263ea917d0`.
- Runner log: `/home/ubuntu/lichess-bot/nohup.lazy-smp.2t.20261001T232021Z.log`.
- Rollback config: `/home/ubuntu/lichess-bot/deployment_lazy_smp_20261001/config.rollback_1t.20261001T232020Z.yml`. It selects the preserved old binary with one thread. Stop the runner between games before restoring it.

Post-startup logs confirm the actual UCI process received the explicit model path and Threads=2. At the final check the runner was connected and awaiting games; Lichess was rate-limiting outgoing challenges, so no post-deployment game outcome is claimed. `live_verification.json` records this check.

The fresh online cohort begins with this deployment. Online rating recovery and a direct old-versus-new two-thread Elo difference remain unmeasured. The previous +72.2 Elo result was new-build eight versus one thread on the Mac, not this deployment comparison.

## Evidence

`summary.json` links each trial to its independent assessment. `linux_six_cases.ndjson`, `linux_all_45.ndjson` and `linux_one_thread_control.ndjson` preserve all results and replay prefixes. `sf_judgements/` and `sf_rbq_10m/` contain the independent analyses. `remote_validation.tar.gz` preserves Linux build/test/replay logs. `candidate.tar.gz`, `source_manifest.json`, `source_delta.json`, `lazy_smp.patch`, scripts and the regression fixture make the candidate and procedure reviewable. `post_deploy.json` records sanitized startup/process evidence. Credentials and the live configuration contents are excluded.
