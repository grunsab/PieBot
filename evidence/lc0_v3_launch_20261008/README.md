# LCZero lineage v3: founded so the trainer's gate uses the fixed engine

Written 2026-10-08 ~19:05 UTC. Facts below were observed on the box at that
time; re-verify before acting.

## Why

Search arm S76 cost more than half of PieBot's draws against stronger outside
engines (`evidence/search_arms/s76_check_extension_unguarded_20261008/`). The
fix is commit `9d316d5`. Lineage v2 could not adopt it: the lineage identity
digests `PieBot/src/**/*.rs`, and `migrate_vast_source_commit.py --mode
lc0-pretraining` refuses once training state exists. So v2's promotion gate
kept playing both sides on the S76 engine. The owner asked for a new lineage.

## What v3 is

A continuation of v2 with the engine changed and nothing else, as far as the
tooling allows. Source commit `4bef963ec2a54d8fbf2c266f3c118af59d0f717a`.

| Founding parameter | v2 | v3 |
| --- | --- | --- |
| Engine source | `4e4a79c` (with S76 guard) | `4bef963` (guard removed) |
| Corpora | five windows, 2026-03-07..09-07 | the same five, reused in place |
| Seed and chunk order | 20260907 | the same; starts at cursor 3870 |
| Starting weights | first lineage's chunk 34965 | v2's last training checkpoint, chunk 3869 |
| Optimizer | fresh Adam | fresh Adam (project rule for a new lineage) |
| Learning rate | 4.375e-4, x0.995 per 100M positions | 4.0026471065457984e-4, same decay |
| Gate | 400 games, 1000 ms, vs cycle 168 at blend 75 | the same |
| Deadline | 2026-12-07T18:57:28Z | the same |
| Validation set | sha `f2c3fdde...` | the same file |

The learning rate is v2's schedule value after its 1,774,560,773 trained
positions (0.0004375 x 0.995^17.7456), so the decay continues without a step.
The Adam moments are the one thing that does not carry over.

Two trainer options were added for this (commit `4bef963`):
`--reuse-corpus-manifest` trains on existing frozen corpora instead of
acquiring them, and `--initial-cursor` starts the first pass part-way through
the chunk order. The cursor enters the lineage identity only when non-zero.

## Cutover, done 2026-10-08 (owner-authorised)

- 18:52:44Z: `supervisorctl stop piebot_lc0v2`. It was 23 minutes into its
  once-a-day gate (`gates/chunk_00003870_fce3f8624e8d`, S76 engine on both
  sides), which was killed with it. v2's state therefore reads
  `status: error`, `evaluation_pending: true`, with the killed `compare_play`
  command as `last_error`. Training itself stopped at a clean boundary:
  3,870 chunks complete, pass 0, cursor 3870, `in_progress` null, best
  validation loss 0.61768, last 0.61788. The state file was not edited.
- v2's last checkpoint was copied to
  `/workspace/piebot_campaign_v8/lc0v3_bootstrap/lc0v2_chunk_00003869_checkpoint.json`,
  sha256 `d014f0a880c21069417166218431056a48a4f9bfda89edea5e309bbb178b2b88`
  (identical to the source file).
- Checkout `/workspace/piebot_lc0v3_repo` at `4bef963`, built at low priority.
  matein3 depth-7 signatures: `accept` 7040853, `accept_temp` 7040853. Its
  `uci` binary has the same sha (`eed3db2f...`) as the monitors' build in
  `/workspace/piebot_engine_repo`.
- 18:55Z: v2's three confs removed from `/etc/supervisor/conf.d` (backups
  `/workspace/piebot_lc0v2*.conf.bak.prev3.*` and
  `/workspace/piebot_memguard.conf.bak.prev3.*`); `piebot_lc0v3`,
  `piebot_lc0v3_anchor`, `piebot_lc0v3_ranked` installed from `deploy/vast/`;
  `piebot_memguard` now guards `piebot_lc0v3`.
- 19:01Z, after verifying every chunk checksum: first `lc0-chunk-start` shows
  `number 0, cursor 3870, chunk_index 213, learning_rate
  0.00040026471065457984`. Chunk index 213 is what the seed's pass-0 order
  gives at position 3870 (v2's last chunk, position 3869, was index 3979 in
  both the order and v2's log), so the traversal continues where v2 stopped.

Output root `/workspace/piebot_lc0v3_20261008`; state file
`training/lc0_state.json` under it.

## What must not be deleted

- `/workspace/piebot_lc0v2_20261004/data` (290 GB): v3 trains on these
  corpora in place. The rest of the v2 root (state, 4 GB of checkpoints,
  gates) is v2's record.
- `/workspace/piebot_campaign_v8/lc0v3_bootstrap/`: v3's starting weights,
  pinned by sha in the lineage identity.
- `/workspace/piebot_lc0v3_repo`: pinned checkout; do not pull, build or edit
  it while `piebot_lc0v3` exists.
- `/workspace/piebot_repo` is no longer used by any program. v2 could only be
  resumed from it, and never alongside v3.

## Measurement

- Stockfish ladder: `/workspace/piebot_anchor_unguard_20261008` (kept; the
  cycle-168 baseline that began at 17:27Z resumed after the restart).
- Ranked cohort: `/workspace/piebot_ranked_v3_20261008`, new, because the
  campaign root is part of that monitor's configuration. Its cycle-168
  baseline restarted from zero; `/workspace/piebot_ranked_unguard_20261008`
  holds the 49 games played before the switch.
- Off-box backups: the Mac watcher was restarted against the v3 root, state
  in `out/lc0_backup_monitor_v3/`. v2's last off-box snapshot is chunk 3848.

## What to expect

- v3's best validation loss starts over at its starting weights' own loss,
  0.6178761525726318. That is exactly v2's last value, which confirms the
  weights loaded are v2's. The first four chunks on fresh Adam read 0.61796,
  0.61802, 0.61802, 0.61796: no jump. Comparisons with v2's best 0.61768 are
  valid (same validation file).
- The gate runs once a day and at the end of a pass, so the first gate on the
  fixed engine is due about 2026-10-09 19:00Z. Head-to-head between two
  PieBot nets was level-to-negative for LCZero nets on the S76 engine; whether
  the fixed engine changes that verdict is not known yet.
- Pass 0 has 2,491 chunks left (6,361 total); pass 1 then walks all 6,361.

To return to v2 instead: stop and remove `piebot_lc0v3` and its monitors,
restore the `*.bak.prev3.*` confs, `supervisorctl reread`, `update`. v2 would
first rerun the gate it was interrupted in.
