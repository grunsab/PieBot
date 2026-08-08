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
   - Copy the baseline file `piebot/src/search/alphabeta.rs` to:
     - `piebot/src/search/alphabeta_temp.rs`
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

Super-GM Campaign Handoff (2026-08-08) — CURRENT
================================================

Written for any agent/LLM taking over. Re-verify every live fact before
acting on it. (Older handoffs were pruned 2026-08-08; their still-binding
rules are folded into the durable-rules subsection below.)

### Mission and standing
- Goal (user-set): super-grandmaster strength, ~2700 Elo on the pinned
  Stockfish anchor scale. Budget: 1-2 months of Vast.ai time from ~2026-08-05;
  the user will add hours if asked.
- Measured strength 2026-08-08 (era-2 canonical ladder, 100 games/rung,
  60+0.5s, 1T): S1-era engine + cycle-98 net at blend 25 scored 84.5% vs
  SF16-1500 and 69% vs SF16-1800 → pooled **1892 Elo, 95% CI [1833, 1961]**
  (`evidence/ladder_era2_s1_cycle98_20260808.json`). Gap to goal: ~800 Elo.
- Two tracks bank Elo independently: search arms (promoted S1+S2 are worth
  roughly +100-200 anchor Elo over era-1) and NNUE training (plateaued in
  v1-v4; v6 arch-v2 rebuild minted, deployment pending — see below).

### Working branch and repo state
- Work lives on branch `campaign-v2`, pushed to `origin` (GitHub
  `grunsab/PieBot`); `main` is stale at `7a1e791`. At writing, tip is
  `a45c120`. Do not rebase published history.
- Committed assets: `models/cycle_000098_quant.nnue` (active/incumbent, sha
  `3fa9bae3...`) and `cycle_000094_quant.nnue` with `models/MANIFEST.json`;
  `books/openings_v1.fen` (1,279 openings, sha `d35b81a1...`);
  `evidence/` (promotions, probes, ladders, benchmarks);
  `scripts/experiments/` (h128 twin builder, NPS bench, depth-9 cost probe);
  `documents/CampaignPlan_SuperGM_v1.md` (authoritative plan) and
  `documents/PostDeadlineBattery.md`.

### Infrastructure (live at writing — RE-VERIFY)
- Production box: Threadripper PRO 7995WX + RTX 4090, 150 GB disk:
  `ssh -p 14790 root@81.166.173.12` (65.6 c/hr). Read
  `/etc/vast-agents-guide.md` after login.
- **The TR box's rental expires ~2026-08-21. Re-migrate the campaign to a new
  box by ~2026-08-19** (task #14). Qualify successors with
  `scripts/cpu_benchmark.sh`; the cutover sequence is rehearsed in this
  session's history: stop supervisor → bundle/fetch code → stage bootstrap by
  SHA → install conf → verify node signatures → start.
- The previous box (192.220.55.116, in the historical handoff) is DEAD; the
  EPYC candidate was released. Do not use those endpoints.
- Supervisor program `piebot_campaign_v2`, conf at
  `/etc/supervisor/conf.d/piebot_campaign_v2.conf` (source:
  `deploy/vast/piebot_campaign_v2.conf`), logs at
  `/workspace/piebot_campaign_v2_supervisor.{log,err}`. `stopasgroup=true`.
- Box git quirk: `/workspace/piebot_rust`'s `origin` is a stale LOCAL BUNDLE
  (`/workspace/piebot_campaign_v2.bundle`), and the box has NO GitHub
  credentials yet. A read-only keypair was generated on the box
  (`~/.ssh/id_ed25519.pub`, comment `piebot-tr-box-readonly`) awaiting
  registration as a GitHub deploy key (see deployment block below).
- Standing user directive: only ~150 GB disk; proactively delete old
  self-play game shards (never state files, checkpoints, gate JSON, or
  accepted quants) to keep the run alive. At writing: 126 GB free, v4 root
  only 7 GB — no action needed yet. Autopilot retention keeps 8 full cycles.
- Leave ~24 threads free for SSH/arena lanes (training lanes use 160).

### Strength measurement protocol (era-2)
- Anchor: official SF16 avx2 release, sha256
  `8f60a016dc767e0d648a8665b8ede3e6e4d28c086ad90517ad26f55b9960bd84`, at
  `/workspace/stockfish16` on the TR box (`evidence/anchor_repin_20260807.json`).
  Era-1's pinned binary died with the old box; era-1 numbers (~1650-1800
  pooled for the pre-S1 engine + cycle-98) are comparable within a few Elo.
- Ladders: `scripts/uci_elo_ladder.py` (parallel rungs, pooled performance
  rating + bootstrap CI). 100+ games/rung; place rungs within ±400 of
  expected strength; SF16 UCI_Elo clamps silently outside 1320-3190.
- NEVER mix scales: local-Mac SF18 numbers (~2000 for the same engine) are a
  different scale used only for fast iteration signals.
- Queued: ladder the S2-era engine (baseline now includes S1+S2) at the next
  checkpoint.

### Search-arms track (biggest proven Elo source)
- Workflow: exactly the A/B process at the top of this file (fork
  `alphabeta_temp.rs`, matein3 acceptance both engines, 400-game Mac screen
  at 150 ms noise 12/top-5 paired, 1000-game confirmation, promote only if
  paired-bootstrap 95% LCB > 0).
- Banked: S1 interior PVS + TT-move-first ordering (+0.206 mean pair delta,
  1000g) and S2 reverse futility pruning (+0.12, CI [+0.058, +0.184], 1000g).
  S5 log-log LMR shelved (flat). Evidence in `evidence/`.
- Build-verification practice: the matein3 acceptance run is deterministic;
  the post-S2 baseline `accept` signature is 20117448 total nodes. Use node
  signatures to prove a remote box actually rebuilt your code.
- Queue (in order): S3 futility pruning (child-level, the alpha-side sibling
  of S2), AVX2 eval kernels (box-only; also shrinks the h128 cost), S8
  continuation history. `PieBot/src/search/alphabeta_temp.rs` is currently
  the re-export stub — clean start.

### NNUE training: lineage history and diagnosis
- v1 (original 72h run, old box): promoted cycles 94 and 98, then 66 cycles
  of nothing. Cycle-98 at blend 25 is STILL the active/incumbent model.
- campaign_v2 (data fixes: opening book, adjudication, actor budget):
  25 cycles, 0 promotions. campaign_v3 (C8: diverged learner as teacher):
  26 cycles, 0 promotions — the fixed point re-formed one level up (epoch-0
  no-ops). campaign_v4 (250cp outcome target, depth-5 actor): 29+ cycles,
  0 promotions, still running at writing.
- Decisive 2026-08-07 evidence — pure-network blunder protocol (300 games
  each, depth 3, blend 100, book openings, seed 20260821, PST depth-5 judge):
  cycle-98 ACPL 34.0 / 1.77 blunders/game / 81 zero-blunder games vs v4
  cycle-22 learner 36.6 / 2.15 / 64. **The v4 learner is weaker than its own
  teacher's source net.** There is no gate-masked progress; the h64
  self-distillation loop cannot outrun its teacher.
- Supporting measurements (all in `evidence/`, scripts in
  `scripts/experiments/`):
  - h128 speed probe: a function-identical hidden-128 twin of cycle-98
    (duplicate hidden units, halve w2_scale — `make_h128_twin.py`) searches
    bit-identical trees at 0.809× NPS → width doubling costs 19.1%, ~15-20
    Elo at fixed time. Eval is only ~24% of node cost. The Rust loader reads
    hidden_dim from the file header — h128 needs zero engine changes.
  - h128 pre-screen: +0.51% val loss vs h64 on frozen identical data.
  - Depth-9 teacher cost (150 book positions, blend 25, cycle-98): median
    4.10M nodes, mean 4.58M, p95 8.75M (`depth9_cost_probe.py`).

### campaign_v6 (arch-v2, minted 2026-08-08 — deployment pending)
- Supersedes the v5 h128-old-arch spec BEFORE it ever ran: the user directed a
  standard-design NNUE rebuild (chessprogramming.org/NNUE) with accumulator
  >= 1024. v6 = dual-perspective SCReLU learner from fresh random weights.
- Architecture (PIENNQ02, engine + trainer + exporter all landed and tested):
  perspective-relative shared HalfKP transformer (40,960 inputs/perspective),
  two color-anchored accumulators, side-to-move-first concatenation, SCReLU
  clamp(0,QA)^2 integer head, QA=255 QB=64 SCALE=400, i16 first-layer quant.
  Engine dispatches by file magic; v1 (PIENNQ01) and v2 coexist, so the
  cycle-98 v1 incumbent remains actor/teacher/gate opponent. Cross-language
  parity is enforced by committed fixtures (PieBot/tests/nnue_arch_v2.rs:
  index fixture, incremental==full, SCReLU==reference, and a Python-exported
  gold model asserted integer-exact from Rust).
- Trainer: train_torch --arch v2 (stm-ordered dual bags, white-POV labels
  flipped to stm-relative, w2 clamped to the int8@QB envelope, checkpoint
  format piebot-halfkp-dp-screlu-v1-torch); quantization via
  run_pipeline._export_v2_checkpoint; autopilot --train-arch derives lineage
  identity (input_dim 40960, feature_set halfkp-dp-screlu-v1) and accepts
  both quant magics; launcher TRAIN_ARCH env wired and contract-tested.
- Conf env (deploy/vast/piebot_campaign_v2.conf): OUT_ROOT=
  /workspace/piebot_campaign_v6, TRAIN_ARCH=v2, HIDDEN_DIM=1024,
  FRESH_INIT=1, RELABEL_DEPTH=9, RELABEL_EVERY=6, RELABEL_MAX_NODES=2500000,
  SELFPLAY_DEPTH=5, TARGET_CP=250, bootstrap active model = cycle-98 v1.
- MEASURED COST (evidence/arch_v2_h1024_speed_probe_20260808.json): h1024 v2
  runs at 546k NPS vs 2.37M for v1 h64 (0.231x, ~1.5-2 plies at equal time)
  on auto-vectorized scalar kernels. Hand-written SIMD (AVX2 box / NEON Mac)
  is now the TOP search-side priority: until it lands, v2 candidates carry
  roughly 100-150 Elo of speed penalty into every 150ms gate game.
- v4 (old arch) was STOPPED on the box 2026-08-08 00:47Z by user order; its
  state is preserved on disk. Nothing is running on the box.

### Operational pitfalls (each cost real time — do not repeat)
- Detached/nohup scripts on Vast boxes start WITHOUT cargo/python on PATH:
  export `PATH="/root/.cargo/bin:/venv/main/bin:$PATH"` first. Verify remote
  rebuilds via the matein3 node signature, not by trusting exit codes.
- `pkill -f` can kill its own wrapper shell — use bracket patterns
  (`pgrep -f '[t]raining.nnue.autopilot'`) or `pkill -x`.
- Bash `${VAR:-default}` swallows EMPTY strings — that is why FRESH_INIT is
  a flag, not an empty INITIAL_CHECKPOINT_SOURCE.
- The box supervisor stop previously orphaned the python autopilot;
  `stopasgroup=true`/`killasgroup=true` are mandatory in the conf.
- Never write the source pin before all preflights pass (a poisoned root
  refuses relaunch); the launcher now orders this correctly — keep it so.
- supervisorctl restarts can race the state-file flock: deploy scripts retry
  (6 × 15 s) rather than failing.
- Vast key-rotation can wipe appended `authorized_keys` entries; transfers
  from the Mac use the user's Vast-managed key (`ssh-add ~/.ssh/id_ed25519`).
- Long `cargo`/game runs exceed the 10-minute foreground tool timeout — run
  in background and poll.
- This session's permission classifier blocks outbound file transfer and
  credential-granting commands (scp/ssh-cat/gh deploy-key). Do not try to
  smuggle payloads (e.g., embedding tokens in URLs); surface the exact
  command for the user to run instead.

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
1. Get the v6 deploy unblocked (one user command above), deploy, verify.
   S3 futility screening result may also be ready to act on.
2. While v6 trains: hand-written SIMD eval kernels (AVX2 box / NEON Mac)
   — now the top search arm; the 4.3x v2 slowdown is the campaign's
   biggest lever. Then S8 continuation history.
3. Ladder the S2-era engine (era-2 anchor) — baseline Elo credit for S2.
4. Prepare the ~2026-08-19 box migration (task #14): qualify a successor
   box, rehearse cutover, budget ~2h downtime at a cycle boundary.
5. Every few days: v6 external instruments (blunder protocol + ladder),
   disk check, off-box backup of state/quants/checkpoints.

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
- In progress via the search-arms track (S1 PVS and S2 RFP promoted; S3
  futility, AVX2 eval kernels, S8 continuation history queued).
- Acceptance: tactical boosts; endgame correctness; fewer zugzwang/fortress traps.

Phase 10: Tuning, Tooling, Release

- Goals: SPSA/Texel; arenas; PGO; UCI options.
- Acceptance: Elo gains in 1k–5k matches; reproducible builds.

Performance Targets

- Eval: incr NNUE <200 ns scalar, <80 ns NEON (measured 2026-08: ~44 ns
  quiet apply/revert at h64 after the feature-major cache).
- Search: ≥1–3 Mnps early; ≥5–10 Mnps post‑LMR/ordering.
- Parallel: 4T ≥3.5x; 8T ≥6x.

