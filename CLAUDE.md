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

This section supersedes the 2026-08-06 handoff below it (kept for its durable
rules; its live facts describe a dead box and a finished run). Written for any
agent/LLM taking over. Re-verify every live fact before acting on it.

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
  v1-v4; v5 pivot minted, deployment pending — see below).

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

### campaign_v5 (minted, tested, PUSHED — deployment pending)
- Design (user-directed): hidden-128 student from FRESH random weights,
  taught by cycle-98 searching depth 9 capped at 2,500,000 nodes (= measured
  p95 of depth-7 cost, mirroring how depth-7 was capped at depth-5's p95),
  relabeling every 6th ply (was every 2nd). Fewer labels, each far deeper.
  Actor (depth-5 self-play), teacher engine, and gate incumbent all stay
  cycle-98 h64. PieBot-only teacher invariant unchanged.
- Launcher changes (contract-tested in
  `scripts/tests/test_vast_campaign_v2_deployment.py`): `FRESH_INIT=1` skips
  checkpoint staging and omits `--initial-checkpoint*` so train_torch
  initializes at `--hidden-dim` (width-mismatched warm starts hard-fail by
  design); `RELABEL_DEPTH` accepts the two measured shapes {7@144k, 9@2.5M}.
- Conf env (deploy/vast/piebot_campaign_v2.conf): OUT_ROOT=
  `/workspace/piebot_campaign_v5`, HIDDEN_DIM=128, FRESH_INIT=1,
  RELABEL_DEPTH=9, RELABEL_EVERY=6, RELABEL_MAX_NODES=2500000,
  SELFPLAY_DEPTH=5, TARGET_CP=250, 160/160 lanes, bootstrap active model
  from `/workspace/campaign_v3_bootstrap/cycle_000098_nnue_quant.nnue`.
- Full battery green before push: 231 training + 87 scripts Python tests,
  `cargo test --locked --all-targets` and `--all-features`.
- **BLOCKED**: this session's permission sandbox denied every payload route
  to the box (scp, ssh-cat, `gh repo deploy-key add`). The user must run ONE
  of (from the repo root on the Mac, `!` prefix in Claude Code):
  1. `gh repo deploy-key add <path-to-box-pubkey> --repo grunsab/PieBot
     --title piebot-tr-box-readonly` — pubkey is on the box at
     `~/.ssh/id_ed25519.pub` (preferred: box can then fetch all future
     deploys from GitHub), or
  2. `scp -P 14790 <a campaign-v2 git bundle> root@81.166.173.12:/workspace/deploy_v5.bundle`.
- Then deploy: on the box, fetch/checkout the campaign-v2 tip by 40-char SHA
  and run `NEW_SHA=<sha> bash scripts/deploy_v5_tr.sh` detached (nohup, and
  note it must run with `PATH=/root/.cargo/bin:/venv/main/bin:...` exported —
  the script does this). It gracefully stops v4 (state preserved), verifies
  no orphan workers, SHA-verifies the checkout, installs the conf, starts v5.
- Post-deploy verification checklist: supervisor RUNNING; launcher log shows
  `fresh-init lineage: hidden-dim 128` and `teacher node cap: 2500000`;
  `/workspace/piebot_campaign_v5/autopilot_state.json` appears and advances;
  cycle-0 train stage logs hidden_dim=128; first gate runs h128-candidate vs
  h64-incumbent without dimension errors.
- v5 success metric is NOT the gate alone: run the blunder protocol and
  anchor ladder on v5 candidates every few days. If h128 under the deep
  teacher also flatlines vs cycle-98 on external instruments after ~40-60
  cycles, escalate teacher depth/cap (the launcher now supports measured
  re-calibration) or revisit blend-25 gate dilution before burning weeks.

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

### Standing rules that still bind (from the historical handoff below)
- PieBot-only teacher labels; Stockfish is an evaluation anchor ONLY.
- TDD: failing test before code change; full battery (both Python suites +
  both cargo suites) green before any merge; search changes additionally
  need the A/B game workflow.
- Lineage semantics, "never hand-edit state", safe-experiment rules, and the
  read-only jq inspection recipes in the 2026-08-06 section remain valid —
  substitute the current OUT_ROOT for the dead run root.
- One experimental change per arm; a new lineage (new OUT_ROOT + fresh
  optimizer) is the boundary that may bundle several founding parameters.
- Do not deploy depth-6 self-play (tested 2026-08-07: noise-level data-shape
  gains at 2.2× cost). Depth 5 is the deployed actor.

### Immediate queue for the next agent
1. Get the v5 deploy unblocked (one user command above), deploy, verify.
2. While v5 trains: run the S3 futility-pruning arm end-to-end on the Mac.
3. Ladder the S2-era engine (era-2 anchor) — baseline Elo credit for S2.
4. Prepare the ~2026-08-19 box migration (task #14): qualify a successor
   box, rehearse cutover, budget ~2h downtime at a cycle boundary.
5. Every few days: v5 external instruments (blunder protocol + ladder),
   disk check, off-box backup of state/quants/checkpoints.

Live NNUE Training and Model-Quality Handoff (2026-08-06) — HISTORICAL
----------------------------------------------------------------------
> SUPERSEDED by the 2026-08-08 handoff above. The box, SSH endpoint,
> run root, and live snapshot below are DEAD/finished. Only the durable
> rule subsections (lineage semantics, safe-experiment rules, inspection
> recipes, P0-P7 framework) remain useful, with paths updated.

### Scope and objective
- This section is the current operational handoff for improving the quality and playing strength of PieBot's NNUE output models. The older 2026-02-07 progress snapshot in `AGENTS.md` is historical.
- Optimize real chess strength, not merely training loss, checkpoint churn, or promotion count.
- Relabeling must remain PieBot self-teacher only. Never use Stockfish, another engine, or downloaded engine evaluations as training labels. Stockfish may be used only as a fixed external playing-strength evaluation anchor.
- Preserve the search/TDD workflow in this file. Write a failing test before a code fix, keep search experiments in `PieBot/src/search/alphabeta_temp.rs`, and do not merge or deploy with a red test gate.
- Treat every live fact below as a timestamped snapshot. Re-query before drawing current conclusions.

### Repository and access
- Local repository: `/Users/rishisachdev/Documents/GitHub/chess_engines/piebot/piebot_rust`
- Git remote: `git@github.com:grunsab/PieBot.git`
- This tracked handoff intentionally contains the live SSH endpoint requested by the owner, but no key or password. Treat the repository as operationally sensitive and redact access details before publishing it or copying it to a public fork.
- Local `main` and `origin/main` were clean and equal at `7a1e791e67392f110a66c695ebabffd17c006672` (`Fix NNUE training promotion plateau`) when this handoff was written.
- Vast.ai repository: `/workspace/piebot_rust`
- Vast.ai engine crate: `/workspace/piebot_rust/PieBot`
- SSH login, using the existing authorized key:

```bash
ssh -p 21990 root@192.220.55.116
```

- SSH login with the requested local port forward:

```bash
ssh -p 21990 \
  -L 8080:localhost:8080 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  root@192.220.55.116
```

- Immediately after login, read the provider guide before acting. `vast-capabilities` is a live capability report, not a replacement for the guide:

```bash
cat /etc/vast-agents-guide.md
vast-capabilities
date -u '+%Y-%m-%dT%H:%M:%SZ'
```

- SSH can be intermittent while all CPU slots are busy. A timeout after the Vast.ai banner is not proof that training stopped. Retry sparsely with bounded timeouts; do not create a connection storm.

### Authoritative production paths
- Launcher: `/workspace/piebot_rust/scripts/run_vast_5090_self_teacher_72h.sh`
- Supervisor source template: `/workspace/piebot_rust/deploy/vast/piebot_training_72h_self_teacher.conf`
- Supervisor program: `piebot_training_72h_self_teacher`
- Legacy supervisor that must remain stopped: `piebot_training_48h`
- Run root: `/workspace/piebot_runs/main_72h_self_teacher_repair_v1`
- Autopilot state: `/workspace/piebot_runs/main_72h_self_teacher_repair_v1/autopilot_state.json`
- Pinned source SHA: `/workspace/piebot_runs/main_72h_self_teacher_repair_v1/source_git_commit`
- Cycle artifacts: `/workspace/piebot_runs/main_72h_self_teacher_repair_v1/cycles/cycle_XXXXXX`
- Current accepted quantized model at the snapshot: `cycles/cycle_000098/nnue_quant.nnue`
- Supervisor autostarts on reboot and restarts unexpected failures. A clean deadline exit is expected to stay stopped because `exitcodes=0` and `autorestart=unexpected`.

### Exact live snapshot
- Observation time: `2026-08-06T00:44:01Z`.
- Run status: `running`; `last_error` was `null`.
- Run began: `2026-08-04T03:54:03Z`.
- Persisted absolute deadline: `2026-08-07T03:54:03Z`. Do not reset it by launching a different output root.
- Supervisor was `RUNNING`, PID `107224`, at `2026-08-06T00:39:49Z` with uptime `17:11:52`.
- Completed cycles: 164. Cycle 165 began at `2026-08-06T00:43:49Z`.
- Only two models had been accepted: cycle 94 and cycle 98. The active actor/teacher remained cycle 98 at 25% NNUE blend.
- Active SHA-256: `3fa9bae3127319930ec16ebb1ee3117656abe7001984f6c8655108a08d278c3a`.
- Cycle-94 SHA-256: `7a922154c40e276197caf4ffd933e5bc6172bf84c6511b4afc4db3574feb9b0f`.
- The learner checkpoint still advances after eligible non-promoted cycles; actor/teacher advancement is deliberately promotion-gated.
- Cycle 164 completed in about 24m05s and was not promoted. At 50% blend, its 24-game screen mean was `+0.333333`, but its 96-game confirmation mean was `-0.083333` with CI `[-0.250000, +0.083333]`. At the 25% fallback, its screen mean was `+0.083333`, but confirmation was `-0.020833` with CI `[-0.229167, +0.187500]`.
- Across cycles 99-164 there were 66 completed cycles and no promotion: 29 `unchanged-training-checkpoint`, 20 `screen-rejected`, and 17 `confirmation-rejected` final outcomes. These are final cycle outcomes, not the number of per-blend gate attempts.
- Earlier genuine gains did pass the gate. Cycle 94 confirmed at mean `+0.3125`, 95% CI `[+0.125, +0.5]`; cycle 98 confirmed at `+0.25`, CI `[+0.041667, +0.458333]`.
- The host reported 192 logical CPUs, 251 GiB RAM, an RTX 5090 with 32,607 MiB, and 109 GiB free workspace disk. The production launcher is configured and preflighted for 46 concurrent CPU slots; do not assume all 192 host CPUs are available to this container without checking affinity and cgroup quota.

### Read-only live inspection

```bash
PIEBOT_RUN_ROOT=/workspace/piebot_runs/main_72h_self_teacher_repair_v1
PIEBOT_STATE="$PIEBOT_RUN_ROOT/autopilot_state.json"

stat -c 'state_mtime=%y state_bytes=%s' "$PIEBOT_STATE"

supervisorctl status piebot_training_72h_self_teacher
supervisorctl status piebot_training_48h
pgrep -af 'training\.nnue\.autopilot|selfplay|relabel_jsonl|compare_play|train_torch'

sed -n '1p' "$PIEBOT_RUN_ROOT/source_git_commit"
git -C /workspace/piebot_rust rev-parse HEAD
git -C /workspace/piebot_rust status --short --branch
```

Do not print the entire multi-megabyte state file. Use a projection:

```bash
jq '{
  queried_at_utc: (now | floor | todateiso8601),
  status,
  started_at_utc:
    (if .started_at then (.started_at | floor | todateiso8601) else null end),
  deadline_utc:
    (if .deadline_ts then (.deadline_ts | floor | todateiso8601) else null end),
  next_cycle,
  completed_count: ((.completed_cycles // []) | length),
  current_cycle: (
    .current_cycle |
    if type == "object" then {cycle, status, out_dir, started_at} else . end
  ),
  active_model: {
    path: .active_model_path,
    sha256: .active_model_sha256,
    blend_percent: .active_model_blend_percent
  },
  accepted_models: (
    (.accepted_models // []) |
    map({cycle, blend_percent, quant_sha256, promotion_evidence_status})
  ),
  training_checkpoint_path,
  training_checkpoint_sha256,
  last_error,
  last_gate: {
    reason: .last_gate.reason,
    accepted: .last_gate.accepted,
    blend_percent: .last_gate.experimental_blend_percent
  }
}' "$PIEBOT_STATE"
```

Aggregate final gate outcomes after cycle 98:

```bash
jq '[
  (.completed_cycles // [])[]
  | select(.cycle >= 99)
  | (.gate.reason // "missing")
] | sort | group_by(.) | map({reason: .[0], count: length})' "$PIEBOT_STATE"
```

Inspect a cycle without dumping all game records:

```bash
PIEBOT_CYCLE=$(jq -r '
  if (.current_cycle | type) == "object"
  then .current_cycle.cycle
  else .next_cycle
  end
' "$PIEBOT_STATE")
PIEBOT_CYCLE_DIR=$(printf '%s/cycles/cycle_%06d' "$PIEBOT_RUN_ROOT" "$PIEBOT_CYCLE")
find "$PIEBOT_CYCLE_DIR" -maxdepth 3 -type f \
  \( -name '.piebot_stage_complete.json' \
     -o -name 'metrics.json' \
     -o -name 'pipeline_summary.json' \
     -o -name 'gate_*.json' \) \
  -print
```

### Current production configuration
- 8,000 games per cycle, all starting from the normal initial position.
- Self-play: depth 2, one search thread per game, 46 parallel games, maximum 160 plies.
- Exploration: temperature starts at 1.0 and falls to 0.1 over 24 plies; Dirichlet alpha 0.30, epsilon 0.25 for the first 12 plies.
- Teacher: PieBot-only depth 5, relabel every second ply, 46 worker threads, 4 GiB hash, teacher lag 0.
- A lag-0 teacher means the latest accepted model, not the latest unpromoted training checkpoint.
- Training: CUDA, 700,000 sampled positions, batch 4,096, one epoch, hidden width 64, HalfKP all-pieces v2 input dimension 81,920.
- Target schema: `soft-cp-wdl-v2`; objective schema: `nnue-objective-v1`; WDL scale 400 cp; decisive outcome target magnitude 100 cp; teacher mix 0.8 on teacher-labeled rows.
- Sampling: 50% current-cycle data and 50% across a six-cycle replay window; 50% of sampled rows are teacher-labeled. On outcome-valid rows, the nominal label mixture is roughly 40% depth-5 teacher and 60% outcome. Outcome-invalid but teacher-usable rows are 100% teacher, unusable rows are excluded, and effective gradient contribution also depends on prediction error; do not treat 40/60 as a measured gradient split.
- Warm-start learning rate: 0.001. Adam state is continued when compatible.
- Primary validation is game-hash disjoint but rolls with the current/replay distribution. A pinned PieBot depth-6 reference corpus only vetoes a greater-than-1% loss regression relative to that cycle's incoming model; it does not rank epochs or prove Elo improvement.
- Promotion: 24-game positive-mean screen, then 96-game paired confirmation, 150 ms/move, 12 parallel games, one search thread per game, opening noise for 12 plies among top five moves.
- Same-lineage candidates normally try the next blend (currently 50%) and fall back to the incumbent blend (currently 25%). Promotion requires the confirmation paired-bootstrap lower confidence bound to exceed zero versus the incumbent. Incremental PST comparison is a regression veto, not a claim of strict PST superiority.

### Lineage semantics that must not be broken
- `training_checkpoint_path` advances after every eligible completed training cycle, even when gameplay promotion fails. This is how improvements accumulate across rejected cycles.
- `active_model_path` advances only after a statistically accepted gameplay gate. The active model generates self-play and is the lag-0 self-teacher.
- Epoch 0 is the exact incoming weights and optimizer state. If `best_epoch == 0`, the selected cycle is a genuine no-op.
- A changed checkpoint file SHA does not by itself prove changed learned weights; metadata or optimizer serialization can change. Compare parameter tensors/hashes, quantized model SHA, and predictions on a fixed FEN probe.
- Objective, feature-set, target-schema, or architecture changes require an explicit compatible migration. Incompatible Adam moments must not be restored. Prefer a new experimental output root and a weights-only start.
- Never hand-edit `autopilot_state.json`, `source_git_commit`, checkpoints, gate JSON, or accepted-model records.

### Important fixes already implemented
- `6b2705d`: repaired search correctness and enforced PieBot-only self-teacher training. External Stockfish relabeling is disabled.
- `47d7d34`: made learner checkpoints continue across promotion rejection so cumulative training is not discarded.
- `34c1498`: repaired game-disjoint primary checkpoint selection, fixed-reference eligibility, optimizer/objective binding, and epoch selection.
- `333982a`: parallelized paired promotion games while preserving paired-opening evidence.
- `7a1e791`: made the 24-game screen a positive-mean resource filter, retained strict 96-game confirmation, changed successor-vs-PST handling to a regression veto, added same-blend fallback handling and auditable source-pin migration, and accelerated incremental NNUE updates with a feature-major cache.
- Incremental NNUE quiet apply/revert improved from about 229 ns to 44 ns; a 12-ply update line improved from about 4.25 microseconds to 1.96 microseconds.
- A real-search comparison still measured NNUE around 1.907 Mnps versus PST around 3.080 Mnps, with average depths about 6.31 versus 6.81. Evaluation quality must outweigh its remaining search-speed cost.
- The nanosecond, NPS, depth, dataset, and quantization figures in this handoff came from prior one-off production audits, not a durable checked-in benchmark artifact. Reproduce them with recorded commands, hardware, and retained outputs before using them for a new deployment decision.

### Current plateau diagnosis
1. The strongest explanation is a shallow self-distillation fixed point. The depth-2 actor and depth-5 teacher both use the last accepted cycle-98 model at 25% blend. The learner repeatedly approximates labels produced by that same shallow accepted policy/search, so novel improvement signal decays rapidly.
2. Candidate generation often reaches the same quantized gate identity. From cycles 99-164, 29/66 final outcomes were `unchanged-training-checkpoint`. That reason proves an unchanged quantized candidate/configuration for gate purposes, not necessarily identical float tensors or optimizer state. Use the P0 ledger to distinguish a true epoch-0 no-op from a float update erased by quantization. Either case occurs before promotion and cannot be caused by the gameplay gate.
3. Changed candidates are usually flat or weaker. Seventeen reached full 96-game confirmation after cycle 98 and none passed. The gate can miss tiny improvements, but it is not the main explanation for the observed plateau.
4. The data distribution is narrow and correlated. Every game starts from the initial position; noise creates branches, but there is no opening corpus. Half the sample is replay from recent versions of essentially the same policy.
5. A retained data audit found about 54.44% draws and about 47.95% threefold terminations. These produce a large neutral outcome component and suggest repetition-heavy self-play. Re-measure on current retained cycles before assuming those percentages are unchanged.
6. Outcome pressure is weak by construction. A decisive +/-100 cp target under a 400 cp sigmoid scale corresponds to only about 0.562/0.438 rather than hard 1/0 labels. Teacher rows are half the configured sample, subject to outcome-validity filtering and teacher-only handling described above.
7. Rolling validation measures fit to the same self-generated distribution, not independent playing strength. The fixed reference is only a per-cycle regression veto and can permit cumulative drift.
8. Continued Adam moments with a flat warm-start LR of 0.001 may be poorly adapted after many cycles or teacher/distribution changes. Restoration is implemented correctly; whether the schedule is good remains unproven.
9. The 64-unit `EmbeddingBag -> ReLU -> linear` network may be at a representation ceiling and lacks explicit side-to-move/tempo and castling-rights features. Wider models must be judged on equal-time games because slower evaluation reduces search depth.

### Leading explanations already weakened or falsified
- Lost checkpoint lineage is fixed: learner checkpoints continue across promotion rejection.
- The teacher is not accidentally Stockfish and score orientation is correct.
- The teacher is not stale because of a path-resolution bug; it is intentionally frozen at the most recent accepted model until promotion.
- Adam restore/binding is coherent. This does not prove the current LR or moment age is optimal.
- Quantization is unlikely to be primary. On 20,000 real positions, float-to-quant MAE was 1.54 cp, correlation was 0.999946, and only 3.43% of float deltas at least 0.1 cp disappeared.
- The latest promotion rules do pass real improvements, as cycles 94 and 98 demonstrate. There were no final PST-veto outcomes across cycles 99-164.
- The sampler can theoretically cycle scarce strata, but prior production inspection found pools ample; there is no evidence that quota oversampling caused this plateau.

### Ranked investigation and experiment plan

#### P0 - Build a factual cycle ledger before changing hyperparameters
- For every cycle from 99 onward, extract active/teacher SHA and blend, incoming and selected parameter identity, optimizer SHA and Adam step, quant SHA, `best_epoch`, initial/selected primary and fixed-reference losses, sample hashes/counts, and every gate attempt where the artifact is retained.
- Production retains only eight full cycles and prunes many older non-accepted artifacts. Use durable state/gate metadata for older cycles and explicitly mark unavailable tensor, optimizer, sample, and loss fields; a complete historical ledger cannot be reconstructed retroactively.
- Classify each cycle as: exact epoch-0 no-op; float update erased by quantization; changed quant model rejected by gameplay; or promoted model.
- Prove from artifacts which accepted model actually generated self-play and teacher labels.
- Plot absolute metrics on one immutable probe. Do not compare rolling primary loss values across different cycle datasets as if they were the same validation set.

#### P1 - Measure data and teacher novelty
- Audit unique `(run_id, game_id, ply)`, unique FENs, duplicate multiplicity, games per source, valid-outcome fraction, W/D/L, termination reasons, truncation rate, ply/phase/material distribution, teacher coverage/depth, CP histogram, and clamp rate.
- Compare active depth-5 labels with latest-candidate depth-5 labels on the same fixed 10,000 positions. Measure CP delta and best-move agreement. Near-total agreement confirms the self-distillation ceiling.
- Measure whether replay contributes genuinely different positions or mostly repeats the same policy distribution.

#### P2 - Run a frozen-data optimizer ablation
- Use one identical checkpoint, one frozen sampled dataset, and separate output directories.
- Compare continued Adam at 0.001 (control), fresh Adam at 0.001/0.0005/0.0002, and continued Adam with a lower or decayed LR.
- Compare one versus two epochs only if epoch 2 wins the same primary validation and stays within the fixed-reference safety envelope.
- Record Adam step and moment norms, gradient norm, update/weight ratio, parameter deltas, fixed-probe prediction deltas, quantized-byte change fraction, and held-out losses.
- Promote nothing from this ablation without paired equal-time games.

#### P3 - Increase self-teacher information at comparable compute
- Run isolated experimental roots for depth-5/every-2 control, depth-6/every-4, and depth-7/every-8 or a small high-depth stratum. Cadence alone does not make these compute-matched: enforce a fixed node budget where possible, or measure and report total teacher nodes and CPU-seconds for every arm.
- Use only PieBot as teacher. Initially keep the accepted model as teacher. Testing an unpromoted learner as teacher is riskier and requires a frozen accepted rollback target and explicit contamination guard.
- Measure teacher-label disagreement, fixed-probe gain, quantized change, NPS-normalized cost, and paired gameplay strength.

#### P4 - Improve self-play coverage and effective sample size
- Add a deterministic, diverse opening suite instead of always starting from the initial position; retain game-level validation isolation and paired evaluation openings.
- Ablate replay windows 0/2/6 and primary fractions 0.5/0.75/0.9 on fixed experimental runs.
- Deduplicate positions or cap duplicate multiplicity using a declared semantic key that preserves side to move, castling rights, en-passant state, halfmove/rule state, and any separate repetition metadata. Do not merge placement-identical positions with different legal or draw state. Stratify by game phase, decisive/valid outcome, and tactical content.
- If truncation or repetition is excessive, compare fewer depth-3 games against more depth-2 games at an equal CPU budget.

#### P5 - Test the learning objective
- Compare current WDL targets with controlled teacher-mix and outcome-target magnitudes such as 100 versus 200 cp, and optionally Huber CP loss.
- An objective or target-schema change must start a compatible new lineage with fresh optimizer state. Raw losses from different objective semantics are not comparable.
- Evaluate calibration, tactical ordering, fixed-reference behavior, quantization, and games rather than choosing the lowest numeric loss alone.

#### P6 - Test the capacity/speed frontier
- Compare hidden widths 64 and 128 first; test 256 only if 128 is promising. Use identical data and seeds.
- Measure incremental eval time, NPS, reached depth, memory, validation, quantization error, and equal-time games at blends 25/50/75/100.
- Consider SIMD/layout improvements before deploying a wider network. A more accurate but slower evaluator can lose Elo by reducing search depth.

#### P7 - Strength validation and gate power
- Use paired openings, reversed colors, deterministic seeds, equal movetime or equal nodes, and confidence intervals or SPRT.
- Compare candidate versus active at the same blend, candidate versus PST, and periodically versus fixed Stockfish 2200/2500 anchors. Stockfish games remain evaluation only.
- Three-to-five anchor games are smoke tests, not an Elo estimate. Use materially more games before claiming strength.
- Pool pair-level evidence from consecutive candidates with the same active teacher only as a diagnostic; do not silently change the promotion rule or combine non-identical models as if they were one candidate.
- Report relative-to-active evidence, strict-PST evidence, and external-anchor evidence separately.

### Required dashboard
- Training: best-epoch-0 rate, initial-to-selected loss deltas, immutable-reference trend, optimizer steps/moments, gradient/update norms.
- Data: unique games/FENs, duplicate rate, valid outcomes, W/D/L, threefold/truncation rates, teacher coverage and label distribution.
- Model: float tensor hashes/deltas, output deltas on a fixed FEN probe, best-move/order agreement, quantized-byte change fraction and quantization error.
- Chess: paired score, confidence interval/Elo, tactical acceptance, fixed-depth and equal-time matches.
- Runtime: eval/update nanoseconds, NPS, nodes, reached depth, and 1T/4T scaling.
- Lineage: active, actor, teacher, learner checkpoint, quant model, objective, and optimizer identities for every cycle.

### Safe experiment and deployment rules
- Do not mutate or benchmark against the live production resources while the 72-hour supervisor is active. Run diagnostics that materially consume CPU/GPU on a separate machine, or only under an explicitly authorized bounded allocation on the same host. A separate output root alone does not isolate CPU/GPU resources.
- Never manually start the launcher or autopilot while the supervisor program exists; that can create duplicate trainers.
- Do not pull, checkout, build, or edit the remote repository while the run is active. Its source commit is pinned.
- Do not stop or restart merely because a promotion gate takes longer than an average cycle. Confirm supervisor status, child process, current-cycle artifacts, and timestamps first.
- Let autopilot retention manage cycle artifacts. Never delete them manually.
- Make one experimental change per arm, freeze seeds/data where possible, and retain enough artifacts to replay the result.
- A high epoch-0 selection rate may reflect real convergence; do not reset Adam or weaken validation without a controlled ablation.
- Keep the legacy 48-hour supervisor stopped.

### Before merging any code change

```bash
git status --short --branch
python3 -m unittest discover -v training/nnue/tests
python3 -m unittest discover -v scripts/tests
cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml
cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml
```

For search changes, also run the baseline and experimental acceptance suites and the paired A/B workflow described earlier in this file. Do not use a tiny game sample as proof of Elo.

### Remote source-pin migration, only after explicit authorization
- First finish local tests, commit, push, and record exact old/new 40-character Git SHAs.
- Stop the supervisor gracefully and verify it is stopped. Do not begin with `kill -9`; the supervisor has a 300-second graceful-stop allowance.
- Update the clean remote checkout to the exact approved commit.
- Do not hand-edit the source pin. From the new checkout run:

```bash
/venv/main/bin/python /workspace/piebot_rust/scripts/migrate_vast_source_commit.py \
  --repo-root /workspace/piebot_rust \
  --out-root /workspace/piebot_runs/main_72h_self_teacher_repair_v1 \
  --expected-old-commit <OLD_40_CHAR_SHA> \
  --expected-new-commit <NEW_40_CHAR_SHA>
```

- Inspect the prepared migration audit and source pin, then restart only `piebot_training_72h_self_teacher` through supervisor. Verify the supervisor, state, source SHA, active model, learner checkpoint, and persisted deadline after restart.
- Prefer a new output root for incompatible model/objective experiments rather than migrating the production lineage.

### Recommended first action for a new model
- Do P0 and P1 read-only first. The strongest current hypothesis is a shallow self-distillation/data-novelty ceiling, and the 29 unchanged quantized gate identities are its clearest observable symptom.
- Then run the frozen-data optimizer ablation and a compute-matched deeper/sparser PieBot-teacher ablation in separate roots. These two tests distinguish an optimizer/update problem from a teacher-information ceiling without introducing external labels.

Related Documentation
- docs/NNUE_Training_Strategy.md — outlines the NNUE training approach we use. Read this before changing evaluation paths so search and eval improvements cohere.
- docs/PieBotPlan.md — long‑term plan for evolving the engine into a top chess engine, including milestones and priorities. Use it to guide which search changes are most impactful.

Full PieBot Plan (inline copy of docs/PieBotPlan.md)
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

Repo Layout (Proposed)

- piebot/Cargo.toml: features simd-neon, simd-avx2, simd-avx512, syzygy.
- piebot/src/
  - main.rs: CLI entry (uci/perft/bench/selfplay).
  - uci.rs: protocol engine.
  - board/:
  - mod.rs, cozy.rs: adapter types (bitboards, mailbox, zobrist).
- tt/: transposition table, zobrist, replacement policy.
- search/:
  - iter.rs, pvs.rs, qsearch.rs, ordering.rs, nullmove.rs, lmr.rs,
    extensions.rs, pruning.rs, time.rs, see.rs, split.rs, threads.rs.
- eval/nnue/:
  - features.rs, accumulator.rs, network.rs, loader.rs, quant.rs.
- io/: fen.rs, epd.rs, pgn.rs (minimal), book.rs (optional).
- selfplay/: game loop, sampling/noise, exporters.
- piebot/benches/: search_throughput.rs, eval_throughput.rs.
- piebot/tests/: perft, search smoke, EPD suites.
- piebot/scripts/: arena scripts, data tooling.
- training/nnue/: Python NNUE trainer, dataset readers, exporters.

Phase Overview

1. Bootstrap Engine
2. Deterministic Core & Perft
3. Minimal Search
4. NNUE v1 Integration
5. Search Heuristics v1
6. Parallel Search
7. Self‑Play Generator (Rust)
8. NNUE Training Pipeline (Python)
9. Heuristics v2 + Endgame
10. Tuning, Tooling, and Release

Phase 1: Bootstrap Engine

- Goals: UCI shell; cozy-chess integrated.
- Deliverables: uci.rs, board/cozy.rs, fen.rs, main.rs; perft cmd.
- Acceptance: FEN loads, legal moves, GUI round-trip.

Phase 2: Deterministic Core & Perft

- Goals: correctness guardrail.
- Deliverables: perft to depth 6–7; tricky cases.
- Acceptance: perft suites pass.

Phase 3: Minimal Search

- Goals: ID + PVS + TT + material; qsearch captures+checks; seed SEE.
- Deliverables: search core; time manager.
- Acceptance: beats random/greedy; stable NPS/PV.

Phase 4: NNUE v1 Integration (CPU)

- Goals: HalfKP features; int8/int16 incr accum; scalar+NEON.
- Deliverables: eval/nnue/*; Python exporter.
- Acceptance: incremental vs full recompute parity (±1–2 cp); ≥20x CNN.

Phase 5: Search Heuristics v1

- Goals: ordering (TT, captures, killers, history, cont history); null-move; LMR; aspiration; IID; basic extensions.
- Acceptance: ≥1.5–2.5x depth at fixed time vs Phase 3; stable tactics.

Phase 6: Parallel Search

- Goals: root/in‑tree split; work stealing; shared TT; deterministic test mode.
- Acceptance: 4T ≥3.5x, 8T ≥6x (root heavy); no deadlocks; TT contention OK.

Phase 7: Self‑Play Generator (Rust)

- Goals: bullet self‑play; exporters.
- Acceptance: O(10k–100k) pos/s; reproducible with seed.

Phase 8: NNUE Training Pipeline (Python)

- Goals: WDL/CP targets; efficient dataloader; EMA; export weights.
- Acceptance: offline validation; A/B improvements; exact weight round-trip.

Phase 9: Heuristics v2 + Endgame

- Goals: SEE pruning; singular extensions; probcut/razoring; Syzygy.
- Acceptance: tactical boosts; endgame correctness; fewer zugzwang/fortress traps.

Phase 10: Tuning, Tooling, Release

- Goals: SPSA/Texel; arenas; PGO; UCI options.
- Acceptance: Elo gains in 1k–5k matches; reproducible builds.

Self‑Play Details: opening coverage, root noise, WDL labels, sampling, storage.

NNUE Model & File: layout, quantization, loader checks.

Concurrency & Memory: TT buckets, history/cont-history, arenas, prefetch.

Performance Targets

- Eval: incr NNUE <200 ns scalar, <80 ns NEON.
- Search: ≥1–3 Mnps early; ≥5–10 Mnps post‑LMR/ordering.
- Parallel: 4T ≥3.5x; 8T ≥6x.

Testing & Benchmarks: perft, EPD, benches, stability, arena.

Risks & Mitigations: NNUE correctness, SIMD portability, search races, overfitting.

Timeline: phased schedule across ~20 weeks with ongoing tuning.
