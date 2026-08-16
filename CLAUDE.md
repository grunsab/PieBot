Cozy Search Change Management and A/B Comparison
================================================

This document defines the workflow for making search changes and validating them with real‑world, game‑level comparisons before merging.

Goals
- Avoid regressions by testing with practical game play, not only suites.
- Encourage iterative, isolated changes to the search without destabilizing the baseline.
- Keep iterations fast (short per‑move time budget) and diverse (opening noise).

Test‑Driven Development (TDD) Policy
------------------------------------

- Always write or update tests before changing code (Red → Green → Refactor).
- Add a minimal failing test that captures the intended behavior or bug.
- Choose the smallest effective test:
  - Unit tests for modules/utilities (e.g., PGN/SAN formatting, move logic).
  - Integration/acceptance for end‑to‑end behavior (e.g., mate suites, search).
- Keep tests deterministic: prefer single‑thread for acceptance unless testing SMP; control seeds when relevant.
- Search changes: ensure acceptance suites pass (e.g., `matein3` at depth 7) for both baseline and experimental engines.
- Engine I/O changes (e.g., UCI/PGN): add explicit unit tests that cover corner cases (disambiguation, castling, en passant, promotions, checks/mates).
- CI is a gate: do not merge if any tests fail. Acceptance runs for both engines are required.
- Refactor only when tests are green; maintain coverage for new code paths.

Workflow
1) Fork the search implementation in a temporary file
   - Copy the baseline file `PieBot/src/search/alphabeta.rs` to:
     - `PieBot/src/search/alphabeta_temp.rs`
   - Implement and iterate on your changes in `alphabeta_temp.rs` only.
   - The project builds by default because a stub `alphabeta_temp.rs` re‑exports the baseline; replace it with your modified copy when testing.

   TDD checklist before editing code
   - Add/update tests that will fail without your change.
   - For search tweaks, add or reference an acceptance case (or suite subset) that demonstrates the intended improvement.
   - For formatting/PGN logic, add focused unit tests that assert exact SAN/PGN output.

2) Run acceptance tests first (sanity)
   - Examples:
     - `PIEBOT_SUITE_FILE=src/suites/matein3.txt PIEBOT_TEST_THREADS=1 PIEBOT_TEST_START_DEPTH=7 PIEBOT_TEST_MAX_DEPTH=7 cargo run --release --bin accept`
   - Ensure no new failures before doing game‑level comparison.

3) Compare via head‑to‑head games (movetime or fixed depth)
   - Use the provided compare runner to pit baseline (alphabeta) vs experimental (alphabeta_temp).
   - Add opening noise for the first M plies to diversify games.
   - Keep per‑move time small for iteration speed (e.g., 200 ms).
   - Examples:
     - Movetime: `cargo run --release --bin compare_play -- --games 40 --movetime 200 --noise-plies 12 --noise-topk 5 --threads 1`
     - Fixed depth: `cargo run --release --bin compare_play -- --games 20 --depth 7 --noise-plies 0 --threads 1`
   - The runner alternates colors each game and reports wins/draws.

4) Decision criteria
   - If the experimental search clearly performs better or roughly equal, proceed.
   - If it regresses, iterate further in `alphabeta_temp.rs` until acceptable.

5) Promote or discard
   - If accepted:
     - Replace `alphabeta.rs` with the contents of your `alphabeta_temp.rs`.
     - Reset `alphabeta_temp.rs` back to re‑export (or remove changes) to keep the repo building for the next iteration.
   - If not accepted:
     - Keep `alphabeta.rs` unchanged and discard/rollback the temp file changes.

Notes
- Movetime vs fixed‑depth: acceptance runs (fixed depth) and compare runs (movetime) capture different aspects; use both. You can force fixed‑depth mode in compare_play with `--depth N` (plies); when set, movetime is ignored.
- Threads: for reproducibility start with `--threads 1`. You may also probe SMP scaling with higher threads after passing single‑thread comparisons.
- Noise: The compare runner samples among the top‑K ordered moves (uniform over K) for the first N plies to avoid repeated openings.

Super-GM Campaign Handoff (2026-08-16) — CURRENT
================================================

Written for any agent/LLM taking over. Re-verify every live fact before acting
on it; the previous handoff was five days stale and several of its numbers were
wrong in ways that cost real time (see "Corrections" below).

### Mission and standing
- Goal (user-set 2026-08-16): **~3650 CCRL 40/15**, i.e. a top-ten engine.
  This supersedes the earlier "~2700 on the pinned Stockfish anchor scale".
- Best estimate 2026-08-16 (end of day): **~2430-2580 CCRL 40/15. Gap
  ~1070-1220 Elo.** This is a DOWNWARD revision from the long-standing
  ~2650/2705 and it is evidence-driven, not pessimism: all 60 ladder draws
  were replayed and evaluated at depth 18 by full-strength Stockfish, and
  **76.7% were positions PieBot had already lost -- 19 of them with a forced
  mate available to the opponent, which repeated instead.** The draws are
  unconverted wins, not holds. Reassigning them at 60-100% conversion gives
  2584 / 2429. See `evidence/ladder_draws_are_unconverted_wins_20260816.json`.
  Search-arm A/B numbers are UNAFFECTED -- those are engine-vs-engine and
  never went through the ladder.
- **CORRECTION to a standing claim: the ladder is biased HIGH, not low.**
  CLAUDE.md previously said both instruments are biased low and used that to
  argue the truth sits at the upper end of the interval. Drop that argument.
  PieBot's opponent-insensitive draw floor (point 1 below) inflates every
  ladder rating, and inflates it more at higher rungs. The 150 ms A/B harness
  is separately and genuinely biased low (point 2); that finding is unrelated
  and still stands. The two biases are in opposite directions and must not be
  netted against each other -- they apply to different measurements.
- **Audit verdict (55 agents, 45 of 48 claimed Elo sources refuted): 3650 needs
  12-24 months plus a datagen and architecture rewrite.** The whole verified
  search queue was +39 to +100 Elo, i.e. 4-10% of the gap. A realistic target
  is **2900-3100 over 6-12 months**. Do not pad estimates to reach 3650.
- The remaining ~1000 Elo is two multi-month programs, not a list of patches:
  search selectivity as a co-tuned system (~400-500) and eval/datagen
  (~400-500, the harder half).

### MEASUREMENT: the two instruments are biased in OPPOSITE directions — read this first
(The ladder reads HIGH; the 150 ms A/B harness reads LOW. They measure
different things and must never be netted against each other.)
1. **The rung ladder disagrees with itself because PieBot has a draw floor.**
   A full 100-game-per-rung ladder at 3000/3190 returned 2705 [2636, 2760]
   and 2882 [2809, 2938] -- disjoint, from one binary.
   **The cause is PieBot, not the anchor.** A 200-game control played the two
   rungs DIRECTLY against each other (`scripts/experiments/anchor_rung_saturation_probe.py`):
   the high rung scored 76.0%, a **measured gap of 200.2 Elo, CI [166, 238]**,
   so the nominal 190 is correct and the limiter is properly calibrated here.
   What is broken is PieBot's score: at 2705 it should score 5.8% against
   rung 3190 and it scored 14.5%, overperforming by ~177 Elo. Its draw rate
   is **31% vs rung 3000 and 29% vs rung 3190 -- flat across a 190 Elo
   increase in opponent strength**, almost all threefold repetitions, on top
   of 0 wins in 200 games. That fixed block of draws is a score FLOOR, and
   because it does not fall as the opponent strengthens, the derived rating
   RISES with the rung.
   **Therefore every ladder rating is inflated, and more so the higher the
   rung. 2705 is an UPPER bound, not a central estimate.** Do not re-derive
   from a higher rung to get a nicer number. See
   `evidence/ladder_draw_floor_20260816.json`.
   (An earlier claim today that the `UCI_Elo` limiter was SATURATED above
   3000 was committed and is now RETRACTED -- it was inferred through PieBot
   alone, which cannot separate "equally strong opponents" from "a score rate
   insensitive to opponent strength". Run the direct control first.)
2. **The 150 ms A/B harness UNDERSTATES search changes.** Measured 2026-08-16:
   H1+H4 is +38.3 Elo at 150 ms but **+88.7 Elo, CI [+65.0, +113.3] at 1000 ms**
   (200 games), depth edge 0.67 -> 1.29 ply. Ordering quality compounds with
   search length. **Use >= 1000 ms for anything depth-dependent.** Every arm
   rejected at 150 ms may have been rejected by an instrument that could not
   see it.
3. **The ladder runs `60+0.5` = a whole-game clock, ~1.1-1.5 s/move.**
   CCRL 40/15 is 22.5 s/move — roughly 18x longer. Nothing has ever been
   validated at the target time control.
4. **No PieBot game has ever been played against a CCRL-listed engine.**
   +-250 Elo of instrument error exceeds every Elo banked to date. This is the
   single highest-value unaddressed item. `scripts/uci_elo_arena.py` now takes
   `--stockfish-full-strength` (added 2026-08-16) for when the gap narrows
   enough to make it measurable; today PieBot would score ~0-4% at equal time,
   and a ~1000 Elo gap cannot be bridged by time odds.

### Working branch and repo state
- **`main` is now CURRENT** — fast-forwarded 2026-08-16 from the long-stale
  `7a1e791` to `5e28a98` (74 commits). `campaign-v2` points at the same commit.
  Do not rebase published history.
- Committed assets:
  - **`models/v8_cycle_000013_quant.nnue` — CURRENT BEST.** campaign_v8 cycle
    13, gate-accepted at blend 75, arch-v2 `PIENNQ02`, 80 MB, sha `2ef89594...`.
  - **`models/cycle_000098_quant.nnue` — RETAINED DELIBERATELY, DO NOT DELETE.**
    It is a dependency, not an archive: `scripts/cpu_benchmark.sh:22` reads it
    BY PATH to qualify successor boxes, and its sha is pinned in
    `run_vast_campaign_v2.sh:67` and `test_cpu_benchmark.py:35`.
  - `books/openings_v1.fen` (1,279 openings, sha `d35b81a1...`); `evidence/`;
    `scripts/experiments/`; `documents/CampaignPlan_SuperGM_v1.md`.
- **Deleting files from git does NOT reclaim GitHub space** — blobs persist in
  history, and reclaiming needs a rewrite this repo forbids. `.git` is 1.3 GB,
  dominated by a 92.8 MB `AlphaZeroNet_20x256_rust.pt` blob. Use a Release
  asset or LFS for large artifacts (the 108 MB dense bootstrap already follows
  that precedent).

### Infrastructure (live at writing — RE-VERIFY)
- Production box: Threadripper PRO 7995WX + RTX 4090, 150 GB disk:
  `ssh -p 14790 root@81.166.173.12`. Read `/etc/vast-agents-guide.md`.
- **Rental `end_date` is 2026-08-26 00:00 UTC**, verified from
  `vastai show instance 47024265 --raw` (the `vastai` CLI on the box IS
  authenticated). The previous handoff said "~2026-08-21, migrate by ~08-19" —
  **wrong by 5 days**, and acting on it would have abandoned a productive run a
  week early. Verify with the CLI, never from a handoff.
- **192 cores is a lie**: `nproc` reports SMT threads on **96 physical cores**,
  and `/sys/fs/cgroup/cpu.max` caps the container at **184 CPU-equivalents**.
  All lane math uses 184; `GATE_PARALLEL_GAMES=192` would fail the launcher
  preflight and crash-loop the supervisor.
- Supervisor program `piebot_campaign_v2` (name is historical), conf at
  `/etc/supervisor/conf.d/piebot_campaign_v2.conf` (source:
  `deploy/vast/piebot_campaign_v2.conf`). `stopasgroup`/`killasgroup` mandatory.
- The box reaches GitHub **anonymously over HTTPS**, so no deploy key is needed
  despite `origin` still being the stale local bundle.
- Lane split: `SELFPLAY_PARALLEL_GAMES=112`, `RELABEL_THREADS=112`. **112 is
  deliberate**, reserving ~48 for the search-arm A/B farm. A 2026-08-15
  excursion to 160 was reverted: it bought only -11.6% relabel wall (the
  workload is memory-bandwidth-bound, and past 96 physical cores the marginal
  thread yields 0.16-0.32 cores) while starving the farm that produced S6.

### Search-arms track (biggest proven Elo source)
- Workflow: the A/B process at the top of this file (fork `alphabeta_temp.rs`,
  matein3 acceptance on BOTH engines, 400-game screen, 1000-game confirmation,
  promote only if paired-bootstrap 95% LCB > 0). Screens overstate: H1 screened
  +27.0 and confirmed +18.1.
- Banked: S1 PVS+TT-first, S2 reverse futility, S3 futility, **S6** (delete
  null-move verification at depth<=12, qsearch SEE + delta pruning; +56 Elo
  [+44,+68] — the largest single arm, and missing from the previous handoff),
  and 2026-08-16: **H1** history rewrite (+18.1) and **H4** winning-capture
  priority (+20.2). H1+H4 together are **+88.7 Elo at 1000 ms**.
- **S5b log-log LMR (2026-08-16): PROMOTED, +22.3 Elo, paired-bootstrap 95%
  CI [+12.2, +32.4] over 1000 games at 1000 ms**, +0.85 ply at equal NPS.
  Replaces a FLAT reduction of 1 with `clamp(ln(d)*ln(i)/2.25, 1, d-2)`. This
  is the arm shelved 2026-08-07 whose note said to revisit once ordering
  improved -- H1+H4 did exactly that, and the original judgement had also been
  made on the 150 ms harness that understates depth-dependent arms ~2.3x.
  Screen +22.6 and confirmation +22.3, so this one did NOT overstate. The
  2.25 divisor is inherited from the S5 v2 retune and is untested headroom.
  See `evidence/search_arms/s5b_loglog_lmr_promoted_20260816.json`.
- **SMP (2026-08-16): root splitting was DELETED and replaced with Lazy SMP.**
  The old `search_depth_parallel` was unsound, not merely unscalable: it
  scouted each tail root move against a racing `alpha_shared`, so a fail-low
  scout returned a fail-soft UPPER BOUND, and the aggregation then compared
  those bounds -- taken against *different* alphas, with cancelled workers'
  results silently dropped -- using `score > best_score` to pick the root
  move. It could return a move it never verified. Measured **-161 Elo at 4
  threads vs 1 thread**. Lazy SMP (independent ID loops over the shared TT,
  helper scores never read) measures **+94.9 Elo vs 1T** and **+129 Elo
  [+91, +170] head-to-head against root splitting at 4T over 200 games**;
  NPS scaling 2.47x -> 3.81x. Single-threaded play is bit-identical
  (`accept` still 11742536). Applied to BOTH iterative-deepening loops --
  `search_movetime` *and* `search_with_params`, the latter being what the
  real UCI `go` handler uses (`uci.rs:651`). See
  `evidence/smp_lazy_smp_replaces_root_split_20260816.json`.
  **Open question that decides whether to invest further: it is unconfirmed
  whether CCRL 40/15's main list gives an engine one core or four.** If one,
  this buys nothing on that list specifically. Verification failed --
  computerchess.org.uk 403s automated fetches, ccrl.chessdom.com did not
  resolve -- so a human should check.
- **The matein3 pre-screen IS trustworthy for move-ordering arms.** It is
  blind only to effects it cannot express -- thread count, NPS, time
  management -- because it is single-threaded and fixed-depth. Ordering
  changes ARE what it measures: ordering changes the tree and the tree is the
  node count. On 2026-08-16 the <5% rule was overridden for H2 continuation
  history on the argument that the suite was blind to it; the override was
  wrong, the 400-game screen confirmed the pre-screen's null, and it cost an
  hour. Reserve the override for arms whose effect the suite structurally
  cannot express (the Lazy SMP fix was one; ordering arms are not).
- **The history table is BIMODAL; never scale a constant as a fraction of
  `HIST_MAX`.** Measured 2026-08-16 on real searches: `HIST_MAX` is 16384, but
  the 90th percentile of non-zero entries is **4-9**, with a thin tail reaching
  ~11,000 (midgame d11: 764 non-zero, p50 -5, p90 4, p99 3740, max 11126;
  matein3 d7: 619 non-zero, p90 9, p99 272). Anything expressed as
  `HIST_MAX / k` is therefore a no-op for all but a handful of moves. This had
  never been measured and cost three failed attempts at one arm.
- **Move ordering is a FLAT SUM, so terms must be scaled against each other.**
  Measured ceilings before H4: capture ~10,112 vs quiet 16,474, so a saturated
  history quiet outranked the best capture on the board at 54.1% of depth-10
  nodes. See `evidence/` and the `piebot-search-ordering` memory.
- **Rejected at 150 ms — RE-TEST AT >= 1000 ms BEFORE TRUSTING**: H2
  continuation history (+2.6), log-log LMR (+6.9), history-modulated LMR
  (+3.5), LMP (~1% nodes, loses a mate if pushed). All are depth-dependent and
  all were judged on the understating harness.
- Node signatures (deterministic; use them to prove a remote rebuild):
  `accept` **9983611**; `accept_temp` **10011374** when the fork is the stub.
  (Both changed 2026-08-16 when log-log LMR was promoted; the pre-LMR values
  were 11742536 and 11763048.)
  History for `accept`: 14298048 (pre-H1) -> 13184884 (H1) -> 11742536 (H4)
  -> 9983611 (S5b log-log LMR).
  **The two binaries are NOT comparable to each other** and never were: they
  use different option sets (`opts=alphabeta` vs `opts=(default)`), so they
  legitimately differ by ~20k nodes on an identical tree. Compare each only
  against its own history. (A previous handoff claimed `accept_temp` was
  11742536 on the stub; that was wrong and cost a false alarm.)
- **matein3 CANNOT measure NPS changes** — it loads from FEN with a ~1-entry
  game history. It only proves whether the tree changed. Its node count plus
  mates-solved is however an excellent ~4 s pre-screen: if a change moves nodes
  <5% or loses a mate, do not spend games on it.
- Corrected: the "arch-v2 runs at 0.231x v1 NPS" claim is **wrong** (measured on
  a random-weight net). Real ratio ~0.489x, and live gate logs show ~1.0M NPS at
  depth 7.9 — the handicap is not visible at the gate. SIMD eval kernels are
  NOT the top lever; the accumulator is memory-bandwidth-bound on an 84 MB table.

### NNUE training: v7 stalled, v8 is the objective fix
- **v7 (retired 2026-08-16 at cycle 155, 12 accepted; state preserved at
  `/workspace/piebot_campaign_v7`, restartable).** It stalled for 28 cycles at
  **+4.91 Elo, CI [+1.66, +8.14] over 12,800 gate games, slope -0.04/cycle** —
  flat, and half the gate's +10 indifference point, so the gate was RIGHT.
- **Root cause: objective saturation, not signal exhaustion.** A depth-7 search
  still disagreed with its own net by 617 cp on average and on 57.5% of best
  moves — but BCE through `sigmoid(cp/400)` keeps only 28% of its gradient at
  1000 cp and 18% at 1200, while 28% of labels sit above |600|. Getting stronger
  made the net invisible to its own loss. See
  `evidence/objective_saturation_20260816.json`.
- How the ceiling got installed: `TEACHER_MIX` 0.8->1.0 and
  `TEACHER_SAMPLE_FRACTION` 0.15->1.0 were each justified alone, but together
  they removed the game outcome from 100% of rows — the only signal not derived
  from the net's own search. The launcher still documents the old assumption
  ("rows WITHOUT a teacher label ... target the outcome alone"); at 100%
  coverage there are no such rows.
- **REFUTED, do not re-run**: unfreezing the teacher (swapping c118 for c146 as
  teacher offers KL 0.00074 nats, 3% of the headroom the learner already cannot
  close); Adam/optimizer pathology (mean_sqrt_vhat flat 7.35e-8 -> 7.33e-8);
  the blend ramp (blend 100 is tried first every cycle and loses by ~46 Elo,
  0/51 lifetime).
- **campaign_v8 (live, `OUT_ROOT=/workspace/piebot_campaign_v8`, commit
  de34ac7)**: same architecture and weights, different TARGET. `CP_LOSS_WEIGHT=1.0`
  adds Huber on the wdl_scale-normalised cp error; `TEACHER_MIX=0.9` restores
  the outcome signal; `EPOCHS=1` (held-out loss rose after epoch 1 in 8/8
  measured cycles and epoch 3 was selected 0/8). Weights-only bootstrap from
  v7 cycle 147 with fresh Adam.
- **v8 is working**: reference cp RMSE **326 -> 218 over 11 cycles**,
  `best_epoch = 1` EVERY cycle, and **2 acceptances in 13 cycles** against v7's
  1 in 28. `cp_loss_weight=1.0` is a principled default, NOT measured — if the
  cp term dominates, the symptom is val WDL loss regressing while cp RMSE
  improves; try 0.3.
- **DO NOT implement "select checkpoints by the primary split"** (a 2026-08-15
  audit recommendation). `train_stub.is_better_checkpoint` deliberately accepts
  an epoch when EITHER split improves, and its docstring records why: requiring
  the primary split is what stalled campaign_v6. The real defect is narrower —
  the reference shard is labelled by the cycle-98 net at depth 6 while targets
  come from a depth-7 teacher — so the fix is RE-LABELLING that shard.

### Corpus scale (the eval half of the gap)
- Measured: **374.5 bytes/row raw, 10.35x gzip => ~36 bytes/row.** So 1e9 rows
  is **~36 GB compressed**, not the ~319 GB the audit assumed. Scale is not
  disk-impossible in principle.
- `training/nnue/dataloader.py` now reads `*.jsonl.gz` transparently (it only
  globbed `*.jsonl`, so compressed shards were unreadable and accumulation was
  impossible regardless of policy).
- **Still blocked on this box.** Accumulating instead of pruning costs ~155 MB
  per cycle compressed, ~7 GB/day at ~45 cycles/day, against ~74 GB free and a
  rental ending 2026-08-26. It fills the disk and kills the run. **This needs
  bigger storage or an object store, not a code change.**
- The field trains on 1e9-1e10 rows; PieBot uses 4.3M per generation. A ~2650
  teacher cannot mint a 3650 student.

### Operational pitfalls (each cost real time — do not repeat)
- Detached/nohup scripts on Vast boxes start WITHOUT cargo/python on PATH:
  export `PATH="/root/.cargo/bin:/venv/main/bin:$PATH"` first.
- `pkill -f` can kill its own wrapper shell — use bracket patterns or `pkill -x`.
- Bash `${VAR:-default}` swallows EMPTY strings — hence `FRESH_INIT` is a flag.
- **A supervisor restart re-runs the ENTIRE in-flight cycle from self-play**,
  even with `.piebot_stage_complete.json` markers present (changing a
  parallelism knob appears to invalidate the stage fingerprint). Cost ~23 min.
  Time config changes at a true cycle boundary.
- **Never enforce a throughput heuristic with `die()`** in the launcher:
  host-tunable knobs plus supervisor `autorestart` turn it into a crash-loop.
  Warn instead. The real safety check is `EFFECTIVE_CPUS >= REQUIRED_CPUS`.
- Long `cargo`/game runs exceed the 10-minute foreground tool timeout — run in
  background and poll. Long-lived SSH sessions on this box get dropped; prefer
  short reconnecting polls over one long connection.
- `du --count-links` misreads `jsonl_train` (shards are hardlinked across the
  replay window, nlink 3-4): 32 G reported vs 17 G true. Use plain `du -sh`.
- Do not delete `/workspace/campaign_v3_bootstrap` (conf-pinned bootstrap
  source) or `/workspace/piebot_campaign_v2.bundle` (it is `origin` for the box
  repo).

### Durable operational rules (carried forward)
- Relabeling is PieBot self-teacher ONLY. Never use Stockfish, another
  engine, or downloaded evaluations as training labels; Stockfish is a fixed
  external evaluation anchor only.
- TDD: failing test before code change; full battery green before any merge:

```bash
git status --short --branch
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

  Search changes additionally require the matein3 acceptance suites for both
  engines and the paired A/B game workflow at the top of this file. Do not
  use a tiny game sample as proof of Elo.
- Lineage semantics: `training_checkpoint_path` advances after every eligible
  completed cycle even when the gameplay gate rejects; `active_model_path`
  advances only after a statistically accepted gate and is what generates
  self-play and teacher labels. `best_epoch == 0` is a genuine no-op (epoch 0
  is the exact incoming weights + optimizer state). A changed checkpoint SHA
  does not prove changed weights - compare parameter tensors, quant SHA, and
  fixed-FEN probe outputs.
- Objective/feature-set/target-schema/width changes require a NEW lineage:
  fresh OUT_ROOT, weights-only or fresh start, fresh Adam. Never restore
  incompatible optimizer moments. A new lineage is the boundary that may
  bundle several founding parameters; otherwise one change per arm.
- Never hand-edit `autopilot_state.json`, `source_git_commit`, checkpoints,
  gate JSON, or accepted-model records. Let autopilot retention manage cycle
  artifacts. Sanctioned source-pin moves only via
  `scripts/migrate_vast_source_commit.py` with explicit old/new 40-char SHAs,
  after a verified supervisor stop and explicit user authorization.
- Never start the launcher/autopilot manually while the supervisor program
  exists (duplicate trainers). Do not pull/checkout/build/edit the box repo
  while a pinned run is active. Do not stop a run merely because a gate
  outlasts an average cycle - check supervisor status, child processes, and
  artifact timestamps first.
- Do not run CPU/GPU-hungry diagnostics on the production box while a
  campaign is active; a separate output root does not isolate compute.
- Inspect state with a jq projection, never by dumping the multi-MB file:

```bash
jq '{status, next_cycle, completed: ((.completed_cycles // []) | length),
     active: {path: .active_model_path, sha256: .active_model_sha256,
              blend: .active_model_blend_percent},
     accepted: ((.accepted_models // []) | map({cycle, blend_percent})),
     checkpoint: .training_checkpoint_path, last_error,
     last_gate: {reason: .last_gate.reason, accepted: .last_gate.accepted}
    }' "$OUT_ROOT/autopilot_state.json"
```

- Do not deploy depth-6 self-play (tested 2026-08-07: noise-level data-shape
  gains at 2.2x cost). Depth 5 is the deployed actor.


### Immediate queue for the next agent
1. **DONE -- the re-test queue is CLOSED.** All four arms rejected at 150 ms
   were re-examined at 1000 ms and **one of four paid**: log-log LMR PROMOTED
   (+22.3 Elo), history-modulated LMR shelved (structural no-op), H2
   continuation history shelved (screened -8.7 Elo, NPS -4.1% and depth flat),
   LMP shelved (<1% tree change at every budget; note that tightening it
   removed FEWER nodes, so the obvious retune direction is wrong). Do not
   re-open these without a new mechanism.
   **But the search track is NOT exhausted** -- that queue only covered arms
   already judged once. Phase 9 lists four heuristics that are still entirely
   ABSENT from `alphabeta.rs` (verified by grep 2026-08-16): **singular
   extensions, probcut, razoring, and IIR**. IIR was implemented and screened
   the same day at **+4.3 Elo, CI [-9.6, +18.3]** -- shelved as positive but
   unresolved, and notably it DID buy what it promises (+0.34 ply at equal
   NPS), unlike H2 which bought nothing. Singular extensions and probcut
   **Singular extensions were also implemented and screened 2026-08-16:
   +4.3 Elo, CI [-11.3, +20.0]** -- shelved, same shape as IIR. Probcut and
   razoring remain untried.
   **THE NEXT THING TO TRY IS THE IIR + SINGULAR-EXTENSION BUNDLE.** Both
   screened at exactly +4.3 Elo against the same baseline and they are
   opposite halves of one idea -- IIR reduces nodes with no trustworthy first
   guess, SE extends nodes whose answer hinges on one move -- so they do not
   compete for the same nodes. If additive, ~+9 Elo, which resolves in
   ~800-1200 games instead of the ~4,000 a 4 Elo effect needs. Run it as a
   DECLARED bundle (screen the pair, then confirm at 1000 games), not as a way
   to rescue two arms that individually failed. ~2 h of compute.
   See `evidence/search_arms/iir_shelved_20260816.json` and
   `evidence/search_arms/singular_extensions_shelved_20260816.json`. **H2 continuation history was re-tested 2026-08-16 and SCREENED
   NEGATIVE**: -8.7 Elo, CI [-25.2, +7.8] over 400 games, with NPS down 4.1%
   and depth FLAT (12.92 -> 12.89). The signal was real (6x more non-zero
   entries than plain history at depth 11) but a 5.2 MB table that is 0.3%
   dense costs more in cache pressure than its clamped contribution can
   recover. See
   `evidence/search_arms/h2_continuation_history_shelved_20260816.json`. (log-log LMR was re-tested this way on 2026-08-16 and
   PROMOTED at +22.3 Elo -- the premise is validated, not just plausible.
   **History-modulated LMR was also re-tested and is SHELVED as a structural
   no-op**: ordering already sorts by history, so high-history moves sit at
   idx < 3 and never enter the LMR region, leaving nothing to modulate. Four
   divisors spanning 512x moved matein3 nodes by at most 0.017%, so no games
   were played. Its original +3.5 Elo was almost certainly noise. See
   `evidence/search_arms/h5_history_modulated_lmr_shelved_20260816.json`. Any
   replacement must use a signal ORTHOGONAL to the one ordering consumes --
   which is exactly what makes H2 continuation history the interesting one.) All are depth-dependent and all
   were judged on a harness now proven to understate this class of change by
   ~2.3x. Cheapest real Elo available.
2. **Watch v8**: `cp_loss_weight=1.0` is unmeasured. Failure signature is val
   WDL loss regressing while cp RMSE improves; try 0.3. Track acceptances —
   2 in 13 cycles so far against v7's 1 in 28.
2b. **AN OFF-BOX BACKUP NOW EXISTS (2026-08-16), verified by sha256.**
   `~/piebot_backups/v8_20260816/` on the user's Mac, 1.3 GB, all six hashes
   matched against the box at copy time:
   `autopilot_state.json` (lineage record, 119,251,713 B, sha `4e4bf1c6...`),
   `cycle_000043_checkpoint.json` (resumable weights + optimizer, sha
   `43ce3629...`), and the accepted-model quants for cycles 3, 13, 26 and
   **41** (the active model, sha `e9c8c198...`). `SOURCE_SHA256.txt` in that
   directory is the manifest.
   **This is insurance, not a migration.** It does NOT contain the 18 GB of
   self-play shards or the replay window; reproducing those means re-running
   cycles. It is also a single copy on one laptop, which is a stopgap rather
   than a policy -- `rclone` is installed on the box but unconfigured, and a
   real destination is still an open decision.
3. **Box migration before 2026-08-26** (verified end_date, not the ~08-21 the
   old handoff claimed). Qualify successors with `scripts/cpu_benchmark.sh`
   (which is why `models/cycle_000098_quant.nnue` must not be deleted).
4. **Get a real anchor.** No PieBot game has ever faced a CCRL-listed engine;
   +-250 Elo of instrument error exceeds everything banked. Needs a decision on
   downloading a CCRL-rated opponent of comparable strength.
5. **Storage decision for the corpus program** (1e9 rows = ~36 GB compressed;
   this box cannot hold it at ~7 GB/day). Blocks the eval half of the gap.
6. Every few days: v8 external instruments, disk check, off-box backup of
   state/quants/checkpoints.

Related Documentation
- documents/CampaignPlan_SuperGM_v1.md - authoritative Super-GM campaign plan
  (engine-first strategy, loop restart, measurement protocol).
- documents/PostDeadlineBattery.md - post-deadline runbook.
- The engine roadmap is inline below. (docs/NNUE_Training_Strategy.md and
  docs/PieBotPlan.md, referenced historically, were never created in-tree.)

PieBot Engine Roadmap (inline; phases 1–8 complete)
---------------------------------------------------

Objectives

- Strong engine: aim CCRL 40/15 top‑20 with CPU‑only NNUE.
- Fast search: parallel alpha‑beta with top‑tier heuristics.
- Efficient training: bullet self‑play in Rust; Python pipeline for NNUE training.
- Modular design: start with cozy-chess movegen; keep path open for custom movegen.

Architecture & Tech Choices

- Board/Movegen: cozy-chess for now; wrap in a local adapter layer to ease future replacement.
- Protocol: UCI first; XBoard optional later.
- Eval: NNUE (HalfKP/A‑variant), int8 weights, int16 accumulators, efficient incremental updates.
- Search: iterative deepening, aspiration windows, PVS, TT, null‑move, LMR/LMR+, SEE, history/continuation history, killers, counter‑move, quiescence, late‑move pruning, probcut/singular extension (later).
- Parallelism: jamboree (split points + root parallel), lock‑free TT, work‑stealing pool.
- SIMD: AArch64 NEON on M‑series; x86 AVX2/AVX512 optional via features; scalar fallback.
- Endgame: Syzygy WDL/DTZ probing for 3–6 men (later).
- Build: -C target-cpu=native, LTO=thin, panic=abort, PGO (later), profile‑guided tuning.
- Tools: criterion for benches, cargo-asm/perf/Instruments for hotspots, cutechess-cli for Elo.

Status: phases 1–8 (bootstrap, perft, minimal search, NNUE v1
integration, heuristics v1, parallel search, self‑play generator, NNUE
training pipeline) are complete and in production. Remaining:

Phase 9: Heuristics v2 + Endgame

- Goals: SEE pruning; singular extensions; probcut/razoring; Syzygy.
- In progress via the search-arms track. Promoted: S1 PVS, S2 RFP, S3
  futility, S6 (+56 Elo, largest single arm), H1 history rewrite (+18.1)
  and H4 winning-capture priority (+20.2); H1+H4 measure +88.7 Elo at
  1000 ms. AVX2 eval kernels are NOT the lever (the accumulator is
  memory-bandwidth-bound). Continuation history measured +2.6 and was
  rejected at 150 ms — re-test at >= 1000 ms. Parallel search was rewritten
  from root splitting to Lazy SMP on 2026-08-16: +129 Elo head-to-head at 4
  threads, and it fixed a defect that made multi-threaded play *lose* 161
  Elo to single-threaded.
- Acceptance: tactical boosts; endgame correctness; fewer zugzwang/fortress traps.

Phase 10: Tuning, Tooling, Release

- Goals: SPSA/Texel; arenas; PGO; UCI options.
- Acceptance: Elo gains in 1k–5k matches; reproducible builds.

Performance Targets

- Eval: incr NNUE <200 ns scalar, <80 ns NEON (measured 2026-08: ~44 ns
  quiet apply/revert at h64 after the feature-major cache).
- Search: ≥1–3 Mnps early; ≥5–10 Mnps post‑LMR/ordering.
- Parallel: 4T ≥3.5x; 8T ≥6x. **4T met 2026-08-16 at 3.81x** once root
  splitting was replaced by Lazy SMP (it had been stuck at 2.47x *and*
  negative Elo). 8T measured 2026-08-16 at 7.27x NPS vs 1T (target met)
  but only **+34.9 Elo over 4T, 95% CI [-16.7, +88.0] over 100 games -- not
  significant**. Scaling is sharply diminishing: 4T over 1T was +0.86 ply,
  8T over 4T is +0.25. Resolving a ~35 Elo effect needs ~800-1000 games.
  NOTE the measurement box has 12 physical cores and NO SMT, so those 8
  threads were 8 real cores; an i7-4770k's 8 threads are 4 cores plus
  hyperthreading and will scale worse. Further gains past ~8 threads need
  diversification (varied aspiration windows, per-thread ordering noise).

