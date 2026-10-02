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
