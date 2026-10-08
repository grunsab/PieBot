# LCZero lineage v2: launch record and cutover runbook

Written 2026-10-02 ~22:30 UTC. Facts below were observed on the box at that
time; re-verify before acting.

## Why

The first LCZero lineage (`/workspace/piebot_lc0_20260907`) stopped improving:

- Fixed-validation loss 0.6413 -> ~0.6207 by pass 10, flat through pass 21.
  Best 0.62010 at chunk 34965 (pass 14).
- Pre-step loss on revisited chunks kept falling (0.6218 -> ~0.611), so the
  train/validation gap grew 0.002 -> 0.010: memorisation of a 1.06B-position
  corpus seen ~21 times at a constant Adam learning rate of 1e-3.
- The 150 ms promotion gate rejected 19/19 candidates (-25 to -67 Elo vs v8
  cycle 168), while at 120+1 on the same search binary the LCZero net scored
  174/1200 against the ranked cohort and cycle 168 scored 63/1200.

## What changed (source commit `06eb368718b0b2b4ae06d9c9ae0ac72052e1e686`)

Branch `campaign/lc0-v2-lr-decay-gate-window`, one commit on top of main
`7bd8cb7`. Not pushed; shipped to the box as a git bundle.

| Founding parameter | First lineage | v2 |
| --- | --- | --- |
| Learning rate | 1e-3 constant | 4.375e-4, x0.995 per 100M positions trained |
| Gate movetime | 150 ms | 1000 ms |
| Corpus | 2026-07-07..09-07 | 2026-03-07..09-07 (old corpus + four new monthly windows) |
| Starting weights | v8 cycle 206 | first lineage's validation-best, chunk 34965, fresh Adam |
| Validation set | fixed 100k | the same file (old corpus stays primary) |

The learning-rate numbers are the nnue-pytorch retraining recipe, not a
measurement on PieBot. All three are bundled as the founding parameters of one
new lineage, so their effects are not separable from this run alone.

Box layout:

- Checkout `/workspace/piebot_lc0v2_repo` (the live run's
  `/workspace/piebot_lc0_repo` was not touched).
- Output root `/workspace/piebot_lc0v2_20261002`.
- Starting checkpoint copy
  `/workspace/piebot_lc0v2_bootstrap/lc0_chunk_00034965_checkpoint.json`,
  sha256 `dd349dfc4417ca6c7ec8bf8ad5bdcdb7d91247785a9e4344baf7dcb93be91d62`.
- Supervisor program `piebot_lc0v2`, conf
  `/etc/supervisor/conf.d/piebot_lc0v2.conf`, currently `--prepare-only` with
  `autorestart=false`.
- matein3 depth-7 signatures of the new build: `accept` 5030782,
  `accept_temp` 5030776.

Disk plan (decimal GB, 478 ceiling, 50 GiB reserve): raw sizes per window are
88.8 / 73.7 / 141.0 / 92.4 GB. A window is downloaded whole before conversion
evicts it, which is why the span is split; the worst peak leaves ~109 GB free.
Expected end state ~315 GB used, ~163 GB free. The derived/raw ratio (0.408)
is taken from the first corpus and is an estimate for these months.

## Success criteria, fixed before launch

- Loss: best fixed-validation loss beats 0.62010 by more than 0.001 within the
  first pass (~2.7B positions). If it does not, neither the lower learning
  rate nor the fresh data moved generalisation; do not rationalise it.
- Memorisation: pre-step chunk loss minus fixed-validation loss stays above
  -0.005 through the first pass (it was -0.010 at the end of the old run).
- Gate: the first 1000 ms screen should not reject the starting-quality net
  the way the 150 ms gate did. A local 200-game preview at 1000 ms is recorded
  in `h2h_1000ms.json` next to this file when it finishes.

## Cutover, done 2026-10-02 (owner-authorised)

- 22:33 UTC: `piebot_lc0v2` switched from `--prepare-only` to full
  acquire-then-train mode, `autorestart=unexpected`, `HOURS="1566"`. Prepare-phase
  conf backup: `/workspace/piebot_lc0v2.conf.bak.prepareonly.20261002T2232*Z`.
  `HOURS` is the time to the original 2026-12-07T18:57:28Z deadline minus a
  14 h allowance for acquisition and checksum verification, because the
  deadline clock starts when training state is first created. The run
  therefore ends at or shortly before the original deadline. `HOURS` is part
  of the lineage identity; do not change it.
- 22:33:38 UTC: old `piebot_lc0` stopped gracefully. State `paused`,
  49,828 chunks, pass 21 cursor 478, `last_error` null, best validation loss
  0.62010 unchanged. Its root, corpus and checkpoints are untouched; the old
  corpus is now the primary corpus of v2 and must not be deleted.
- The old run cannot simply be restarted: its checkout is at `fcf7d65` while
  its source pin is `8b71ff3`, so the launcher would refuse. Do not start it
  alongside v2 in any case (two trainers on one GPU).

What to expect: training begins by itself once all four windows are
downloaded and converted (download measured ~17-20 MB/s, 396 GB raw) and
every chunk checksum (~268 GB) has been verified. The first
`lc0-chunk-start` line in `/workspace/piebot_lc0v2_supervisor.log` should
show `learning_rate` 0.0004375. State file:
`/workspace/piebot_lc0v2_20261002/training/lc0_state.json`.

Not done: `piebot_lc0_anchor` and `piebot_ranked` still watch
`/workspace/piebot_lc0_20260907`; they need to be pointed at the new root to
measure v2 candidates.

To abandon v2: `supervisorctl stop piebot_lc0v2`, remove its conf,
`supervisorctl reread && supervisorctl update piebot_lc0v2`, and delete
`/workspace/piebot_lc0v2_20261002` to return the disk.

## Result: local 1000 ms head-to-head (2026-10-02, M4 Pro, 6 parallel games)

LCZero chunk-34965 net vs v8 cycle 168, same search, blend 75, paired noisy
openings, 200 games: **LCZero 70 W / 50 D / 80 L, score 0.475, -17 Elo,
paired 95% CI [-42, +7]**. LCZero searched deeper (15.2 vs 14.2 plies) at
slightly higher NPS. The first ~10 minutes overlapped a local Rust test run,
which slowed both sides equally.

Reading: the head-to-head deficit shrinks from about -40 Elo at 150 ms to
-17 at 1000 ms, but does not turn positive. So the 1000 ms gate will most
likely still reject candidates near this strength, even though the same net
scores 174/1200 against the external cohort at 120+1 where cycle 168 scores
63/1200. Head-to-head between two PieBot nets and play against other engines
disagree; the gate measures the former. Raw data: `h2h_1000ms.json`.

## 2026-10-04: container wiped, lineage v2 rebuilt from scratch

The host's Docker daemon failed on 2026-10-03. After a restart the instance
came back as a new container (created 2026-10-04T01:26:24Z) with an empty disk
and a new SSH port (`ssh -p 40436 root@104.8.120.185`). Everything described
above that lived on the box is gone: both corpora, the old lineage's state and
float checkpoints, the v2 roots, the self-play campaign directory, the ranked
engines and all measurement outputs. Lineage v2 had not started training.

Rebuilt the same day at source commit
`5ec8912095815b33444885facca73e558cd00ca8`:

- Checkout `/workspace/piebot_repo`, output root
  `/workspace/piebot_lc0v2_20261004`, supervisor programs `piebot_lc0v2` and
  `piebot_memguard` (confs checked in under `deploy/vast/`).
- Binaries rebuilt; matein3 depth-7 signatures match the earlier build
  (`accept` 5030782, `accept_temp` 5030776).
- **Starting weights** are recovered from the quantised best net
  `models/lc0_chunk_00034965.nnue` with `training/nnue/quant_to_checkpoint.py`.
  Re-exporting the recovered checkpoint reproduces that net byte for byte
  (sha `56434c1b...`). Checkpoint sha `f38a6ec2...`, identical on the Mac and
  the box. Its weights sit on the quantisation grid (w1/b1 steps of 1/255,
  w2 steps of 1/64) and there is no optimizer state, which v2 never planned to
  carry over anyway.
- **Corpus** is re-acquired in five windows. The first is the original
  2026-07-07..2026-09-07T19:39:28.530592Z window so that it stays primary;
  its inventory again lists 1,433 archives. Check when it completes: corpus id
  should be `a1920eac...` and the validation file sha `f2c3fdde...`. If they
  match, losses are comparable with the old lineage's 0.6201; if not, they are
  not, and that must be said.
- **Deadline** is now absolute: `--deadline-utc 2026-12-07T18:57:28+00:00`.
- **Memory**: conversion runs 8 worker processes (was 16).
  `scripts/vast_memory_guard.sh` logs non-reclaimable memory every minute to
  `/workspace/memory_guard.log` and stops `piebot_lc0v2` if it stays at 80% of
  the container limit (about 42.5 GiB) for three samples. Whether memory had
  anything to do with the host failure is not known.

Founding parameters are otherwise unchanged from the table above. Still not
done: outside-engine and Stockfish-ladder monitors (their binaries were lost),
and off-box backups of training checkpoints.

## 2026-10-05/06: 18-hour crash loop on one corrupt upstream game, fixed

Conversion of the May window stopped at archive 254 of 744
(`training-run2-test91-20260517-1317.tar`): one of its 6,830 game files,
`training.207753962.gz`, is cut off mid-stream on storage.lczero.org. The
converter raised `EOFError`; `autorestart=unexpected` restarted it 128 times
between 2026-10-05T12:46Z and 2026-10-06T06:51Z with no progress.

Fix: commit `4e4a79c` skips, logs and counts (`corrupt_games`) a game whose
compressed stream cannot be read. Deployed 2026-10-06 ~07:05Z with the owner's
authorisation, using `scripts/migrate_vast_source_commit.py --mode
lc0-pretraining` to move the source pin `5ec8912` -> `4e4a79c` (audit record
under the campaign root's `source_commit_migrations/`). Engine sources are
identical between the two commits; no training state existed yet.

Verified at that point: the primary corpus rebuilt identically (corpus id
`a1920eac...`, validation sha `f2c3fdde...`), so validation losses are
comparable with the first lineage. Memory guard peak: 2 GB held (5%).

Cycle-168 baselines on the current engine build, measured while conversion was
running: Stockfish 16 ladder 2886 [2827, 2942] (27.5% vs 3000, 21.5% vs 3190);
cohort 42/1200 (Carp 0W 14D, Lambergar 0W 17D, Schoenemann 0W 53D).

## 2026-10-08: measurement engine changed; trainer unchanged

The engine built from main had lost more than half of its draws against the
ranked cohort because of search arm S76 (see
`evidence/search_arms/s76_check_extension_unguarded_20261008/`). The fix is
commit `9d316d5`.

The trainer could not adopt it: `migrate_vast_source_commit.py --mode
lc0-pretraining` refuses once training state exists, and the lineage identity
digests `PieBot/src/**/*.rs`. So the trainer stays pinned at `4e4a79c`, and its
promotion gate keeps using the S76 engine on both sides.

The two monitors were switched at 17:27Z to a separate checkout,
`/workspace/piebot_engine_repo` at `cd927b7` (`uci` sha `eed3db2f...`,
matein3 `accept` 7040853), with fresh output roots:

- Stockfish ladder: `/workspace/piebot_anchor_unguard_20261008`
- Ranked cohort: `/workspace/piebot_ranked_unguard_20261008`

Both start by re-measuring cycle 168. Numbers from the old roots are not
comparable with the new ones. Last results on the S76 engine:

| Net | Ladder | Cohort |
| --- | --- | --- |
| cycle 168 | 2886 [2827, 2942] | 42 / 1200 |
| chunk 0 (`85ed2799`) | 2926 [2866, 2980] | |
| chunk 1 (`c27f1052`) | | 72 / 1200 |
| chunk 2012 (`febdb22e`) | 2928 [2866, 2984] | |
| chunk 2471 (`2c5c2c36`) | | 46.5 / 618, abandoned part-way |

Conf backups: `/workspace/piebot_lc0v2_{anchor,ranked}.conf.bak.preunguard.*`.
