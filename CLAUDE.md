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
- Best estimate 2026-08-16: **~2650 CCRL 40/15, honest interval 2400-2900.
  Gap ~1000 Elo.** Both available instruments are biased LOW (see below), so
  the true figure is probably at the upper end.
- **Audit verdict (55 agents, 45 of 48 claimed Elo sources refuted): 3650 needs
  12-24 months plus a datagen and architecture rewrite.** The whole verified
  search queue was +39 to +100 Elo, i.e. 4-10% of the gap. A realistic target
  is **2900-3100 over 6-12 months**. Do not pad estimates to reach 3650.
- The remaining ~1000 Elo is two multi-month programs, not a list of patches:
  search selectivity as a co-tuned system (~400-500) and eval/datagen
  (~400-500, the harder half).

### MEASUREMENT: both instruments are biased low — read this first
1. **The rung ladder disagrees with itself by 276 Elo.** The same binary and
   net measured 2146 at rungs 1800/2100 and 2422 at 2400/2700 on the same idle
   box, CIs disjoint. Rung dependence flattens at the top (+100, +148, then
   +39), so **3000/3190 is the canonical set**. Never compare across rung sets.
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
- **Move ordering is a FLAT SUM, so terms must be scaled against each other.**
  Measured ceilings before H4: capture ~10,112 vs quiet 16,474, so a saturated
  history quiet outranked the best capture on the board at 54.1% of depth-10
  nodes. See `evidence/` and the `piebot-search-ordering` memory.
- **Rejected at 150 ms — RE-TEST AT >= 1000 ms BEFORE TRUSTING**: H2
  continuation history (+2.6), log-log LMR (+6.9), history-modulated LMR
  (+3.5), LMP (~1% nodes, loses a mate if pushed). All are depth-dependent and
  all were judged on the understating harness.
- Node signatures (deterministic; use them to prove a remote rebuild):
  `accept` **11742536**, `accept_temp` 11742536 when the fork is the stub.
  History: 14298048 (pre-H1) -> 13184884 (H1) -> 11742536 (H4).
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
1. **Re-test the 150 ms rejections at >= 1000 ms**: H2 continuation history,
   log-log LMR, history-modulated LMR, LMP. All are depth-dependent and all
   were judged on a harness now proven to understate this class of change by
   ~2.3x. Cheapest real Elo available.
2. **Watch v8**: `cp_loss_weight=1.0` is unmeasured. Failure signature is val
   WDL loss regressing while cp RMSE improves; try 0.3. Track acceptances —
   2 in 13 cycles so far against v7's 1 in 28.
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
  rejected at 150 ms — re-test at >= 1000 ms.
- Acceptance: tactical boosts; endgame correctness; fewer zugzwang/fortress traps.

Phase 10: Tuning, Tooling, Release

- Goals: SPSA/Texel; arenas; PGO; UCI options.
- Acceptance: Elo gains in 1k–5k matches; reproducible builds.

Performance Targets

- Eval: incr NNUE <200 ns scalar, <80 ns NEON (measured 2026-08: ~44 ns
  quiet apply/revert at h64 after the feature-major cache).
- Search: ≥1–3 Mnps early; ≥5–10 Mnps post‑LMR/ordering.
- Parallel: 4T ≥3.5x; 8T ≥6x.

