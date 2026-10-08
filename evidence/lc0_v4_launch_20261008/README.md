# Full NNUE: engine default, gate, monitors, and LCZero lineage v4

Written 2026-10-08 ~23:55 UTC. Facts below were observed on the box at that
time; re-verify before acting.

## Decision

The owner decided the engine plays on the network alone (EvalBlend 100) and
is optimised from there. No strength comparison between 100 and 75 was made;
the owner said not to. Until now the UCI default, the promotion gate and both
measurement monitors all used 75% network plus 25% piece-square score.

## What changed

| Where | Before | After | Commit |
| --- | --- | --- | --- |
| UCI default and advertised `EvalBlend` | 75 | 100 | `b6aa0be` |
| Static evaluation at 100 | piece-square score computed, weight 0 | not computed | `1047d9d` |
| Promotion gate, both nets | 75 | 100 | `a3d114f` |
| Stockfish ladder monitor | 75 | 100 | `a3d114f` |
| Ranked cohort monitor | 75 | 100 | `2b64594` on `deploy/ranked-full-nnue` |

`EvalBlend` still accepts 0..100. The piece-square skip does not change the
search tree: matein3 depth 7 with v2's best net (`fce3f862`) gives 7086746
nodes at blend 100 and 7100730 at 75 before and after, and 7040853 without a
net for `accept` and `accept_temp`. Speed at blend 100 on the Mac went from
about 1.20M to 1.28M nodes per second on that suite (two short runs each).

## Lineage v4

The gate blend is part of the lineage identity and the engine sources are
digested into it, so the trainer needed a new lineage again. v4 continues v3
the way v3 continued v2 (`evidence/lc0_v3_launch_20261008/README.md`).
Source commit `a3d114f8da649749342adbfa4d786b449d4b3ac4`.

| Founding parameter | v3 | v4 |
| --- | --- | --- |
| Engine source | `4bef963` | `a3d114f` (default 100, piece-square skip) |
| Gate blend, incumbent and candidate | 75 | 100 |
| Corpora, seed, validation set | v2's five corpora, 20260907, `f2c3fdde...` | the same |
| Starting weights | v2 chunk 3869 | v3's last training checkpoint, chunk 389 |
| Starting cursor in pass 0 | 3870 | 4260 |
| Optimizer | fresh Adam | fresh Adam |
| Learning rate | 4.0026471065457984e-4 | 3.967080433924763e-4, same decay |
| Deadline | 2026-12-07T18:57:28Z | the same |

The incumbent is still v8 cycle 168. It was accepted at blend 75 by the
self-play campaign; at 100 it is expected to be weaker (v8 nets lost about 46
Elo at 100, `evidence/objective_saturation_20260816.json`), so this gate is
probably easier for an LCZero net to pass than the old one. That is a change
in what the gate asks, not evidence that the nets improved.

## Cutover, done 2026-10-08 (owner-directed)

- 23:40Z: monitors stopped; `/workspace/piebot_engine_repo` moved to
  `a3d114f` and rebuilt, `/workspace/piebot_ranked_repo` moved to `2b64594`.
- 23:40:43Z: `supervisorctl stop piebot_lc0v3`. It saved at the next chunk
  boundary: `status: paused`, 390 chunks, pass 0, cursor 4260, 178,063,388
  positions, best validation loss 0.617715, last 0.617765. It never reached a
  gate (the first was due about 19:00Z on 2026-10-09).
- Starting checkpoint
  `/workspace/piebot_campaign_v8/lc0v4_bootstrap/lc0v3_chunk_00000389_checkpoint.json`,
  sha256 `cce07354b44723b31d586454cec28b3de392cacca7814dc7d75739267559b43f`.
- Checkout `/workspace/piebot_lc0v4_repo` at `a3d114f`; matein3 depth 7:
  `accept` 7040853, `accept_temp` 7040853; `uci` sha `6bd5dfe8...`, the same
  as the monitors' build, and it advertises `EvalBlend default 100`.
- 23:42Z: v3's confs removed (backups `/workspace/piebot_lc0v3*.conf.bak.prev4.*`
  and `/workspace/piebot_memguard.conf.bak.prev4.*`); `piebot_lc0v4`,
  `piebot_lc0v4_anchor`, `piebot_lc0v4_ranked` installed; `piebot_memguard`
  guards `piebot_lc0v4`.
- 23:49Z, after checksum verification: first `lc0-chunk-start` shows
  `number 0, cursor 4260, chunk_index 3214, learning_rate
  0.0003967080433924763`. Index 3214 is what the seed's pass-0 order gives at
  position 4260. The starting weights' validation loss is 0.6177645847702027,
  exactly v3's last value. First three chunks: 0.617755, 0.617805, 0.617764.
  State identity: `blend_percent 100`; `active_model_blend_percent 100`.

Output root `/workspace/piebot_lc0v4_20261008`.

## Measurement

Numbers at blend 75 and at blend 100 are different instruments. New roots:

- Stockfish ladder: `/workspace/piebot_anchor_fullnnue_20261008`
- Ranked cohort: `/workspace/piebot_ranked_v4_20261008`

Both begin by measuring cycle 168 at blend 100. Last results at blend 75 on
the engine without the S76 guard, for the record:

| Net | Ladder | Cohort |
| --- | --- | --- |
| cycle 168 | 2922 [2861, 2975] (29.5% vs 3000, 27.0% vs 3190) | abandoned at about 45 of 400 games per opponent |

The cohort baseline in `/workspace/piebot_ranked_v3_20261008` and the ladder
measurement of v3's best net that had just started were abandoned.

Off-box backups: Mac watcher restarted against the v4 root, state in
`out/lc0_backup_monitor_v4/`. v3's only off-box snapshot is its chunk 11.

## What must not be deleted

- `/workspace/piebot_lc0v2_20261004/data` (290 GB): v4 trains on it in place.
- `/workspace/piebot_campaign_v8/lc0v4_bootstrap/` and the pinned checkout
  `/workspace/piebot_lc0v4_repo`.
- The v2 and v3 roots outside `data/` are records only (about 4 GB each).

## Open

- Each further engine change the trainer's gate should use needs another
  lineage under the present identity rule (fresh Adam, re-measured
  baselines). Batch engine changes, or decide to take the engine out of the
  lineage identity.
- The ranked tooling is still on a deployment branch, not on main.
- `PieBot/tests/lichess_smp_blunders.rs` still sets blend 75 explicitly; it is
  a regression test of positions found at that setting.

To return to v3: stop and remove the three v4 programs, restore the
`*.bak.prev4.*` confs, move the two monitor checkouts back to `cd927b7` and
`f7578a5`, `supervisorctl reread`, `update`.
