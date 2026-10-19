# LCZero training campaign — September 7, 2026

The user authorized a separate **14-day** supervised-training lineage using the
existing dual-perspective HalfKP/SCReLU v2 NNUE (1024 hidden units, PIENNQ02).
The starting weights are the last backed-up self-play learner, cycle 206. Adam
starts fresh because the target changes. The old campaign remains restartable.

## Preserved self-play campaign

The original local archive is:

```
~/piebot_backups/v8_cycle 206_20260825T195236Z/piebot_v8_cycle 206_coherent_20260825T195236Z.tar.zst
SHA256 513c6d169872a605e69c80020e69f1efd7f96704b3a272e1c1f8c37940124b9b
```

It is a coherent stopped snapshot: 206 completed cycles, next 207. Restore its
paths under `/workspace`, because the state uses absolute paths. It includes
the cycle 206 float checkpoint and matching optimizer, cycles 203–206 replay,
accepted quants, bootstrap assets, state, source pin, restore notes and Git bundle.
Keep the archived supervisor configuration outside `/etc/supervisor/conf.d`
until the user requests a self-play restart.

| Artifact | SHA256 |
| --- | --- |
| cycle 206 float checkpoint | `144699077a19f50f7097de8426ed73f0f9a9fa30971d9ba0219250044311697d` |
| cycle 168 accepted quant, blend 75 | `271a5a108ee03e20a0036b683e108f76dd44fc2e6d289cc3d3f8c839082c519c` |
| original source commit | `22f3e0c3d736b2151b418558c9b5f4eec8dd1905` |

Cycle168's float checkpoint was pruned before this work; its accepted quant and
promotion evidence survive. Cycle206 is the restartable learner. Keep both roles.
To restart self-play later, first stop both LCZero supervisor programs and verify
their process groups exited. Restore the original repository from the supplied
Git bundle at the source pin, verify all backup checksums, build that pinned
source, then install the archived self-play supervisor configuration. Do not
change the self-play state, source pin or optimizer by hand. Its empty partial
cycle 207 was omitted intentionally and will rerun from the committed boundary.

## Fixed corpus and targets

Source: `https://storage.lczero.org/files/training_data/test91/`. Initial discovery
found 1,432 archives totaling 259,016,028,160 bytes. The source inventory is frozen on
first acquisition; subsequent restarts verify the same inventory and checksums.
No daily refresh or older-data substitution occurs.

Select games collected from July 7 inclusive through the September 7 acquisition
cutoff. The UTC cutoff is written into the manifest. Filter **tar-member mtimes**,
not just archive names: the July 9 backfill contains June games. These are
collection timestamps, not verified play times. Record missing date coverage.

Official V6 games are gzip members named `training.<id>.gz`, not `.bin` files.
The importer checks versions, supported formats, board validity and finite Q
values; discards deletion-marked/invalid records; preserves provenance; and
partitions whole games permanently. One percent of games is held out, with a
fixed deterministic 100,000-position sample spanning the selected corpus.

LCZero Q is W−L. For white-relative records:

```
p = 0.8 * (1 + best_q)/2 + 0.2 * (1 + result_q)/2
loss = BCEWithLogits(pred_cp / 400, p)
```

For max-length outcomes use search Q alone. The existing v2 trainer flips labels
once into side-to-move perspective. No LCZero score is treated as centipawns;
the auxiliary centipawn loss is disabled. Existing self-play targets are unchanged.

Archives remain compressed. Preparation uses 16 workers and writes gzip JSONL
chunks of at most 700,000 positions; only one chunk is expanded for training.
The corpus is traversed completely before repeating in a deterministic shuffled
order. Defaults: Adam, LR 0.001, batch 16,384, one epoch/chunk. Latest weights and Adam
advance after every completed chunk; globally best validation and accepted models
are separate. The immutable initial validation measurement remains the baseline.
Resume replays only unfinished work. A 50 GiB reserve protects checkpoints/backups.

## Deployment and monitoring

Verified box: `ssh -p 40728 root@104.8.120.185`. Use its `/etc/vast-agents-guide.md`.
The new source checkout is `/workspace/piebot_lc0_repo`, campaign root
`/workspace/piebot_lc0_20260907`. Supervisor launches acquisition, preparation,
then CUDA training through `scripts/run_vast_lc0.sh`. The 336-hour deadline begins
after corpus preparation and is preserved on resume. Deadline exit is clean and
is not automatically restarted. Launch preflights validate the pinned bootstrap,
source, compiled engine binaries, CUDA and disk before creating a source pin.

```
supervisorctl status piebot_lc0 piebot_lc0_anchor
tail -n 30 /workspace/piebot_lc0_supervisor.log
jq '{status,completed_chunks,pass_number,cursor,started_at,deadline_at,
     training_checkpoint_path,best_validation_loss,active_model_path,
     last_gate}' /workspace/piebot_lc0_20260907/training/lc0_state.json
df -h /workspace
```

Daily/pass-end changed candidates face the same-search 75%-blend incumbent:
400-game screen then 1,000-game confirmation, 150 ms, 1 thread/game, noise 12/top-5,
paired openings, 95% paired-bootstrap lower bound strictly positive, plus the
existing PST gate. Rejected candidates do not roll back ongoing learning.

The separate `piebot_lc0_anchor` supervisor measures the frozen cycle 168 baseline
and changed validation-best candidates daily without blocking GPU training.
It snapshots each model and uses official SF16 AVX2, SHA
`8f60a016dc767e0d648a8665b8ede3e6e4d28c086ad90517ad26f55b9960bd84`,
at fixed rungs 3000/3190, 100 games/rung, 60+0.5, blend 75. These rungs follow the
later August 15 rung-dependence study; do not compare their ratings directly with
the earlier 1500/1800 ladder or the separate Blunder scale. Measurements record
concurrent training load. Validation loss alone does not demonstrate Elo gains.

## Validation

Run the complete Python NNUE and scripts suites, Rust all-targets tests with and
without all features, and both depth 7 single-thread matein3 acceptance binaries.
Additional LCZero tests cover date/backfill filtering, download durability,
actual member formats, Q perspective, whole-game isolation, full-pass traversal,
checkpoint/optimizer recovery, source identity, and protected self-play paths.
The end-to-end fixture trains a v2 model from binary games, exports PIENNQ02,
and the release Rust UCI searches with that quant. Before the full run, repeat
ingestion and a resumed CUDA training chunk on real test91 data using the restored
h1024 checkpoint in a separate promotion-ineligible smoke root.
