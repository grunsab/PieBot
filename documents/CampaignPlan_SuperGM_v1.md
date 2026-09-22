> **STATUS 2026-08-16 — historical planning document, goal superseded.**
> This plan targeted ~2700 on the pinned Stockfish anchor scale. The user reset
> the goal to **~3650 CCRL 40/15** (top-ten engine) on 2026-08-16. A 55-agent
> audit priced that at **12-24 months plus a datagen and architecture rewrite**;
> the whole verified search queue was +39 to +100 Elo against a ~1000 Elo gap.
> A realistic target on this architecture is **2900-3100 over 6-12 months**.
>
> The reasoning and phase structure below remain useful. The numbers do not:
> current strength is ~2650 CCRL 40/15 (interval 2400-2900), the lineage is
> campaign_v8 (not v2/v5), and **both measuring instruments are biased low** —
> the rung ladder disagrees with itself by 276 Elo, and the 150 ms A/B harness
> understates search changes by ~2.3x. **For live status, read the Super-GM
> Campaign Handoff in CLAUDE.md, which is maintained; this file is not.**

# PieBot Campaign Plan — Final (Authoritative)

**Version:** 1.0 — synthesized 2026-08-05 from three designs, three critiques, two judge reports.
**Skeleton:** integrated-milestones (milestone gates, attribution, slot partition, frozen anchor conditions) + search-first (wind-down discipline, arm mechanics, A/B power spec, v2 root hygiene) + data-first (trap-closure checklist, diagnostics, lineage identity rules).
**Hard invariants (restated, non-negotiable):** PieBot-only self-teacher labels (Stockfish evaluation-only, never a label source); TDD + A/B game gating for every search change per CLAUDE.md; strictly one change per experimental arm; isolated out_roots for incompatible lineages; disk < verified capacity with retention enforced; no hand-editing autopilot state; all box deploys from pinned pushed commits.

---

## 1. Strategy summary

### The bet
The two deficits are multiplicative, and the engine deficit is the higher-confidence, cheaper-to-fix half — so it goes first.

1. **Engine first.** The search core is missing most of the modern pruning stack — verified against the code audit: no interior PVS scout (serial loop alphabeta.rs:1186-1272 vs the working root pattern at 816-848), no futility/LMP (public no-op stubs at alphabeta.rs:316-317), fixed r=1 LMR (1199-1240), no SEE/delta pruning in qsearch (382-454; SEE is ordering-only at 1145-1149), no RFP, continuation history removed (1271, 1726), no IIR, single full-window aspiration re-search (1510-1520), empty SIMD features (Cargo.toml:44-47). Each is a documented Elo source at weak baselines. This work runs on the free Mac under CLAUDE.md TDD/A-B while the box finishes its 72h run and then does measurement + confirmation duty. Every promoted search change also makes every later training dollar cheaper: deeper teacher at equal nodes, stronger actor at equal caps, better gate play.
2. **Then restart the loop on the stronger actor.** The current run is a converged self-distillation fixed point (zero promotions cycles 99-166, 30+ genuine epoch-0 no-ops, 45% threefold rows, near-uniform depth-2 root policy). More identical cycles are worthless. The relaunch (fresh isolated out_root `campaign_v2`) changes the data distribution: opening book, adjudication, real actor budget, node-capped deeper teacher — each individually piloted first — with a gate reworked (SPRT + gate book) **before** the root is minted so gate knobs are genuinely frozen per lineage (F8).
3. **Measure everything on one frozen anchor.** Pinned SF16 (sha `00628bd9c9855c1b7ff93d7f8d51b413586cdc6336fe637f9a14a61531a05aca`), TC 60+0.5, wall cap raised once, clamp rejection installed, rungs always within ~±400 of expected strength. A per-milestone Elo/day attribution table (search = binary deltas at fixed model; training = model deltas at fixed binary) drives a 70/30 portfolio reallocation with a 30% floor.

### Elo decomposition (anchor scale, honest ranges; self-play→anchor discount ~1.5–3× applied to devlog-derived numbers)

| Tranche | Anchor Elo | Confidence |
|---|---|---|
| Baseline recovery (fresh ladder; current binary post-search-repair + cycles 94/98 gains) | +0 to +200 | measurement, not work |
| Search wave 1 (5–6 promoted arms of S1–S10) + AVX2 (+20–45 after Amdahl) | +180 to +350 | highest |
| Loop restart (book, adjudication, actor budget, node-capped depth-7 teacher, LR decay) | +80 to +200 | medium |
| Scaled v2 loop + teacher escalation + optional hidden-128 | +200 to +450 | widest band (self-distillation ceiling risk; C8 hedge) |
| Search wave 2 (singular, probcut, staged movegen, qsearch checks, time manager, tuning) | +60 to +150 | medium-high |
| **Sum over fresh baseline** | **+520 to +1150** | |

**Honest bottom line.** From a fresh baseline of ~1100–1300, the day-45 median landing is **~2100–2400**; with the pre-planned 15-day reserve (days 46–60, inside the 2-month budget), **~2300–2500**. **2700 is explicitly an upper-tail outcome** requiring both tracks near the top of their ranges plus real compounding. No number is invented: cycle time is ~16 min (derived: run started 2026-08-04T03:54Z, cycle 166 done ~45.2h later) until re-measured; every milestone re-prices the priors above with measurements; every strength claim is CI-bounded at a real rung. The reserve is the base-case continuation of the highest-measured-slope track, not an admission of failure.

---

## 2. Immediate actions

### Before the deadline (now → 2026-08-07T03:54Z) — box untouched

1. **Commit + push the uncommitted AGENTS.md/CLAUDE.md plateau handoff (F7) before ANY tree manipulation** — it exists only in the local working tree of HEAD 7a1e791 and is the sole copy of the P0–P7 diagnosis. Then create branch `campaign-v2` and pin the base commit. *(First action; nothing else touches the tree before this.)*
2. Do **not** touch the live run, its deadline, or `autopilot_state.json`. Let it expire.
3. **Mac — start search arm S1 (interior PVS) under TDD immediately:** failing test → `alphabeta_temp.rs` → matein3 depth-7 acceptance both engines → 400-game paired `compare_play` screen (150ms, noise 12/top-5).
4. **Mac — prepare (with unit tests) the measurement + infra commits, ready to deploy the moment the box frees:**
   - C9: add `--bin uci` to the Vast build list (script:292-294); SF16 staging via `stage_verified_file` (script:130-145) with sha `00628bd9…a05aca`; UCI_Elo min/max clamp-rejection at the preflight regex (uci_elo_arena.py:633); raise the 300s wall cap once to 900s (uci_elo_arena.py:53-54, 787-791) — this is the single, permanent calibration condition; record SF sha + wall cap in every arena JSON.
   - Ladder wrapper: loops `--stockfish-elo` with distinct `--results` files, runs rungs as **parallel processes** (each rung internally sequential, concurrency=1 — calibration preserved), and computes **pooled multi-rung estimates** (this pooling does not exist yet — build it).
   - Battery scripts: cycle-log selfplay-vs-relabel wall-clock parser (F5); per-cycle `du` (D); teacher-agreement harness over a fixed 10k-FEN set (frozen now, from cycle-165 shards + curated positions; stored in git-tracked `evidence/`).
5. Curate the opening book on the Mac: ~2,500 FEN lines, 6–10 plies deep, broad ECO coverage. **Positions only — no external labels (invariant-safe).**

### First 48h after the deadline (Aug 7 ~04:00Z → Aug 9)

1. **Snapshot read-only:** final `autopilot_state.json`, gate evidence, newest accepted quant; copy off-box to the Mac. No edits.
2. **Verify disk capacity (F6):** `df` the actual volume (150GB claim vs 109GiB launcher-deploy reference). All retention math uses the verified number. **Verify CPU affinity** (192 logical vs 46 preflighted slots) before final slot budgeting.
3. **Read the incumbent's actual blend from the live `autopilot_state.json` (F4)** — never from docs.
4. **Bootstrap-source verification (mandatory, closes the fatal flaw all designs shared):** quants are NOT checkpoints — `_load_initial_checkpoint` accepts only dense JSON checkpoints in `_DIRECT_CHECKPOINT_FORMATS` (train_torch.py:63-66, 109-152). Check with `ls`:
   - (a) `cycles/000098/train/checkpoint.json` — almost certainly pruned by model-only retention (run at cycle 166, retain=8), but verify;
   - (b) the current run's **protected bootstrap artifact** (legacy cycle-83 dense checkpoint; bootstrap artifacts are never deleted, autopilot.py:1367-1392).
   **Decision rule (pre-registered):** use (a) if it exists; else (b) — the loss vs cycle-98 is only two tiny gated gains (+0.31/+0.25 pair-mean), negligible. The cycle-166 training checkpoint (68 cycles of un-gated drift, confirmation means straddling 0) is **rejected** as a source. Fresh random weights only if the baseline ladder shows PST-only ≥ the net at blend 100. In all cases the cycle-98 accepted **quant** is `INITIAL_ACTIVE_MODEL` (protected forever).
5. **Deploy the measurement commit and run the full battery:**
   - (a) Baseline ladder — {current binary + cycle-98 quant at live blend; same at blend 100; PST-only} × rungs **1320 and 1500** (never 1800 at a ~1100 baseline — the 2026-08-04 measurement at 1800 was 0W 2D 58L, the infinite-CI regime; add 1800 only if the 1500 score exceeds 40%). 100 games/rung, 60+0.5, 3 parallel arena processes (2 CPUs each) → up to ~17h wall.
   - (b) Wall-clock selfplay-vs-relabel split from cycle logs.
   - (c) Measured per-cycle bytes (`du` over a retained full cycle).
   - (d) Teacher-agreement (best-move match % + mean |Δcp|) between active model and latest checkpoint on the fixed 10k FENs.
   - (e) Depth-5 teacher per-position node-count distribution (median, p90, p95) from a sample relabel run — feeds the WP6 node cap.
6. **Record all branch decisions in a committed report (Section 7), then take one full day of slack before minting any out_root identity.** Objective identity and `deadline_ts` are minted once (autopilot.py:2426, 289-316); a wrong mint costs a root relaunch.
7. Box also runs: 1000-game confirmations for finished Mac arms (pre-launch, parallel 12, ~40min each) and AVX2 scalar-vs-SIMD parity tests (Mac is ARM — x86 validation happens only on the box).

---

## 3. Work packages

**Search A/B power spec (governs ALL search arms — the only adequately powered promotion rule in the packet):** 400-game paired `compare_play` screen on the Mac (150ms, noise 12/top-5) → 1000-game confirmation on the box (150ms; parallel 12 pre-launch ≈ 40min, parallel 8 on the A/B lane post-launch ≈ 60min) → promote to `alphabeta.rs` only on paired-bootstrap 95% **LCB > 0**. Pure-speed changes promote on **parity (CI within ±10 Elo) + measured NPS gain**. Never promote on 20–40-game "parity". Each arm gets an iteration budget of ≤2 retune passes before being discarded; a post-hoc margin retune across promoted arms is itself one promotable diff.

### WP1 — Wind-down + measurement battery + anchor infrastructure
- **Objective:** eliminate the five open measurements; freeze anchor conditions permanently.
- **Code:** C9 items above (script:292-294; script:130-145; uci_elo_arena.py:633, 53-54, 787-791); ladder wrapper with pooled estimation; battery scripts.
- **TDD:** unit test for clamp-rejection parsing; wrapper test on synthetic arena JSONs verifying pooled CI math; contract test that arena JSON records SF sha + wall cap.
- **Gate:** all five measurements recorded as numbers in a committed report; finite 95% CIs at ≥2 rungs for the current binary+model.
- **Elo:** +0 (may reveal +0–200 banked). **Duration:** days 0–3. **Runs:** Mac (prep) + box (battery).

### WP2 — Search wave 1 (strictly one heuristic per arm, unbundled)

| Arm | Change | Location |
|---|---|---|
| S1 | Interior PVS zero-window scout | serial loop alphabeta.rs:1186-1272, mirroring root pattern 816-848/900-927 |
| S2 | Reverse futility / static-null | reuse lazily computed static eval from null path, alphabeta.rs:1018-1020 |
| S3 | Futility pruning | wire the `set_use_futility` no-op stub, alphabeta.rs:316 |
| S4 | Late-move pruning | wire `set_use_lmp`, alphabeta.rs:317 — **separate arm from S3** |
| S5 | Log-log LMR table (history/PV/improving adjusted) | replaces fixed r=1 at idx≥3, alphabeta.rs:1199-1240 |
| S6 | SEE<0 pruning in qsearch | alphabeta.rs:382-454 (see.rs complete, used for ordering at 1145-1149) |
| S7 | Delta pruning in qsearch | **separate arm from S6** |
| S8 | Reinstate continuation history | removed at alphabeta.rs:1271, 1726 |
| S9 | IIR | **separate arm** |
| S10 | Gradual aspiration widening | current single full-window re-search at 1510-1520 |

- **TDD:** each arm = failing test first → `alphabeta_temp.rs` → matein3 depth-7 acceptance for BOTH engines → A/B power spec above → promote → reset temp to re-export (CLAUDE.md workflow, verbatim).
- **Gate (cumulative):** consolidated wave-1 binary vs pre-campaign binary ≥ 68% over 1000 paired games at 150ms (LCB > +100); perft + acceptance green; **engine-delta anchor: ladder BOTH binaries at 1320/1500 at identical model/blend** (4 rung-runs ≈ 15–20h on the arena lane — budgeted, not "3h"), delta ≥ +150.
- **Elo:** +160–300 (search) — priority order S5, S3, S2, S1, S4, S6, S8, S7, S9, S10. **Honest throughput: 5–6 promoted arms in 10–12 days**, not 10 in 9.
- **Duration:** days 1–12 (Mac-led; box confirms). **Runs:** Mac dev + box confirmation.

### WP3 — AVX2 NNUE kernels (speed arm)
- **Objective:** fill the declared-empty `simd-avx2` feature (Cargo.toml:44-47) for accumulator delta (network.rs:263-279), full refresh (117-134), ReLU-dot head (281-289).
- **TDD:** scalar-vs-SIMD parity tests (±1–2cp) — **run on the box only** (Mac is ARM; AVX2 cannot be validated locally); NPS benchmark harness.
- **Gate:** parity green + measured end-to-end NPS gain; fixed-movetime A/B for parity (CI within ±10). Honest expectation after Amdahl (eval ≤ ~40% of node cost; baseline already autovectorized under target-cpu=native): **1.2–1.6× NPS → +20–45 Elo**, plus faster relabel/selfplay.
- **Duration:** ~3 days inside days 3–10. **Runs:** box.

### WP4 — Gate rework: SPRT (C5) + gate opening book (C2) — **lands BEFORE the v2 root is minted**
- **Objective:** fix the under-powered 24/96 gate and the gate/actor opening-distribution mismatch at a single gate-identity boundary, then freeze gate knobs for the lineage's life (F8: gate_identity is re-derived from CLI on every restart, autopilot.py:2476-2477, 2734-2786 — the launcher is never edited mid-lineage).
- **Code:** GSPRT over per-pair deltas in `_paired_gate_statistics` / `_run_confirmed_gate_attempt` (autopilot.py:1537-1721, 2076-2329): H0 mean pair delta = 0, H1 = pair-delta equivalent of ~+8 self-play Elo (calibrated from historical gate JSONs), α=β=0.05, min 48 / max 300 pairs in 24-pair batches; the 24-game screen (mean>0) is retained as a cheap resource filter. Gate book: new CLI arg + `generate_paired_opening` (compare_play.rs:432-497, 561-575, 920-935) + pass-through in `_run_model_gate` (autopilot.py:1944-2004); FEN-derived opening_ids satisfy pair-equality validation (autopilot.py:1640-1644). Same curated book file as selfplay — training and gating distributions move together.
- **TDD:** SPRT unit tests replaying recorded compare_play evidence JSONs (schema strictly validated, autopilot.py:1585-1690, 2036-2040); accept/reject/continue boundary cases; gate-book pair-equality test.
- **Gate:** tests green; SPRT decisions reproduce LCB decisions on historical accepted cycles 94/98.
- **Elo:** +0 direct; enables promotion throughput. **Duration:** days 4–10 (must finish before v2 mint; if late, **slip the launch 1–3 days from reserve** — never launch on the old gate intending to swap later). **Runs:** Mac dev, box test.

### WP5 — Loop-restart pilots (sequential, one change each, data-shape metrics)
- **Pilot A — selfplay opening book (C1):** `selfplay_openings` key in `zen5_9755_7d_profile()` (autopilot.py:94-175), `--selfplay-openings` in `_parse_args` (~:362), entry in `_apply_cli_overrides` (:632-686); `_filter_run_pipeline_kwargs` (:2389-2392) forwards automatically; stage via `stage_verified_file`; **loaded-count>0 assertion** against the silent-empty fallback (selfplay/mod.rs:494-521). Live mini-root (`pilot_book_v1`, ~2000 games/cycle, retain 4), ~1 day. Metric: opening-FEN coverage in shards; unique-position rate up.
- **Pilot B — adjudication (C7):** in `generate_single_game` (mod.rs:161-203) using per-ply `value_cp`: resign |cp|≥900 for 8 consecutive plies with a **15% no-resign game fraction** (preserves won-position coverage); draw-adjudicate |cp|≤10 for 40 plies past move 40; new `GameTermination` variant (mod.rs:49-73) auto-serializes (prod uses `--skip-bin`, PIESP001 moot). Live mini-root, ~1 day. Metrics: threefold row share ≤20% (from 45%), truncation ≤5%, wall-time/game down.
- **Pilot C — actor budget (through TDD/A-B, closing the invariant gray zone):** real TT via `set_tt_capacity_mb` (alphabeta.rs:1732; default 4096 entries at :146-149) and CLI-exposed node caps replacing hard-coded 10k/20k (selfplay/mod.rs:346, 461). This **is** a search-config change: unit tests + `compare_play` A/B at selfplay settings + root-policy sharpness check (top-1 mass on the fixed 10k FENs must rise materially above the current 0.050–0.066 near-uniform). Cap values sized from the measured compute split (see Section 7 — the raise is cheap when **relabel** dominates). ~1 day, box A/B lane.
- **Pilot D — node-capped depth-7 teacher (offline, no root needed):** Rust `--max-nodes` **already exists at HEAD** (relabel_jsonl.rs:42, wired :81-84, tests :421-465) and run_pipeline plumbing exists (run_pipeline.py:296-330); remaining work = autopilot knob + launcher `RELABEL_MAX_NODES` + contract test. **Provenance fix (mandatory):** relabel currently stamps the REQUESTED depth (:227-228) — add achieved-depth stamping so `min_teacher_depth` filters honestly (update the `node_capped_teacher_still_produces_a_label` test at :443-465). Cap = **p95 of measured depth-5 node cost** (never the median — that truncates ~50% of searches below depth 5). `min_teacher_depth` stays 5 (no identity change; honest under achieved-depth stamping). Validate offline by relabeling a retained cycle-165 shard: achieved-depth distribution, wall cost (budget ≤ ~2× current relabel), agreement delta. ~0.5 day.
- **Pilot E — LR schedule (C4, offline on the idle 5090):** frozen-data optimizer ablation per AGENTS.md:334-339 (`train_torch.train_model` against an existing `jsonl_train`, outside autopilot): cross-cycle LR decay (hook at autopilot.py:2609-2613 — LR is not identity-locked), fresh-Adam restarts (`--no-continue-optimizer-state`, zero code change), epochs 2, beta sweep. Zero lineage risk. Winner's settings go into the v2 profile with `learning_rate` **explicitly pinned** (F2: omission falls back to code default 0.03).
- **Elo:** +0 direct (throwaway roots); informational. **Duration:** days 6–12, sequential/overlapped. **Runs:** box (A, B, C) + GPU (E) + offline CPU (D).

### WP6 — `campaign_v2` root launch
- **Objective:** mint the long lineage exactly once, correctly. Full config in Section 4.
- **Code:** launcher copy per C11 (fixed non-timestamped OUT_ROOT, `require_autopilot_flag` lines script:246-251, sibling supervisor conf with new program name); contract tests updated (pattern: scripts/tests/test_vast_self_teacher_deployment.py pins exact launcher text).
- **TDD:** contract tests for every new flag, the bootstrap-source path, explicit LR, explicit hidden_dim, book staging + assertion, HOURS value.
- **Gate (first 12 cycles):** openings-loaded count > 0 every cycle; threefold row share ≤20%; truncation ≤5%; root top-1 policy mass materially above uniform; measured cycle bytes match the retention model; cycle time ≤ 40 min; ≥1 SPRT gate acceptance within 15 cycles. Zero acceptances in 15 cycles with clean data-shape metrics → Section 7 pivot to C8.
- **Elo:** +80–200 across the following weeks. **Duration:** launch day ~12–14; runs continuously thereafter. **Runs:** box (32-slot autopilot lane).

### WP7 — Teacher decoupling (C8) — contingent (armed by the tripwire)
- **Objective:** break the self-distillation fixed point if it survives the restart. `teacher_lag_cycles` (script:330) is **near-no-op here** (promotions stopped at cycle 98 — lagged accepted ≈ current teacher); the real change is an **external-teacher autopilot knob**: pass `teacher_relabel_nnue_quant_file` (run_pipeline.py:1289) and bypass the overwrite at autopilot.py:2614-2633. External teacher is always a **PieBot-lineage net** (e.g., best h128 net or a stronger later checkpoint) — never Stockfish (invariant).
- **TDD:** unit test that the knob overrides state-derived teacher; provenance recorded in the completion marker.
- **Gate:** new-teacher lineage = new out_root (teacher provenance is a lineage property); adoption by promotion cadence + anchor checkpoint.
- **Duration:** ~2 days when triggered. **Runs:** Mac dev, box.

### WP8 — Training dedup (C6) — contingent
- Full-FEN-keyed dedup (stm/castling/EP/halfmove per AGENTS.md:349) inside sampling before quota fill (train_stub.py:487-751), preserving deterministic oversampling semantics so teacher-fraction enforcement (train_torch.py:714-725) does not raise. Trigger: measured duplicate rate rises materially post-book (baseline ~3% within-shard). TDD: dedup unit test + oversampling-semantics regression test. ~1 day.

### WP9 — Hidden-128 capacity arm — deferred until data health demonstrated
- **Preconditions:** v2 has ≥3 promotions and clean data-shape metrics; disk arithmetic fits two roots (else sequential with v2 paused at a cycle boundary); GPU pre-screen on frozen data (offline val loss at equal samples) is positive.
- **Mechanics:** own isolated root `h128_v1`; **fresh random weights mandatory** (warm start across widths impossible, train_torch.py:131-152); blend ramp resets to 25 by design; `hidden_dim` passed **explicitly at every entry point** (profile autopilot.py:119, `HIDDEN_DIM` env script:46, train_torch default 64, **run_pipeline CLI default 16** — F9 trap); re-measure int8 quant error (per-tensor absmax scales coarsen; tests/eval_blend_quant.rs pattern).
- **Adoption authority:** the AGENTS.md:357-360 equal-TIME 64-vs-128 controlled comparison + anchor mini-ladder — not the blend-handicapped gate. **F10 mechanism (the missing piece, now explicit):** if h128 wins the external-anchor mini-ladder but fails the incumbent-blend certification, the sanctioned path is to **relaunch a new out_root with the adjudicated h128 net as bootstrap incumbent** (the anchor hook autopilot.py:1824-1913 is an AND-veto, never an override; no state hand-edits).
- **Budget honestly:** a from-scratch net needs a real maturation window — allocate ≥7–10 days of meaningful compute share before the equal-time comparison is meaningful. **Elo:** +40–100 if adopted. **Runs:** box + GPU.

### WP10 — Search wave 2
- Arms (one change each, same power spec): singular extensions, probcut, staged movegen (kill the full Vec sort + board-clone-per-move per node), qsearch checks, razoring, TT sharding replacing Mutex-per-bucket (piebot/src/search/tt.rs:33), **time manager tuned on the anchor TC 60+0.5** (directly priced), SPSA-style constant tuning (each batch = one promotable diff). Single-thread strength strictly dominates (anchor runs Threads=1); SMP out of scope. Note: `target-cpu=native` is already the build default — only PGO is new.
- **Gate:** cumulative ≥ +80 vs wave-1 binary over 1000 paired games (LCB>0); acceptance suites green. **Elo:** +60–150. **Duration:** days ~30–42, Mac-led. 

### WP11 — Milestone ladders + attribution (continuous)
- 2-rung anchor checkpoints every 4–5 days on the arena lane; full milestone ladders at gates (Section 5); Elo/day attribution table committed at each milestone; 70/30 portfolio reallocation of the A/B lane + operator attention + GPU time, 30% floor (droppable only after M3 if a track's slope is <10 Elo/week for two consecutive milestones).

### WP12 — Certification + freeze
- Freeze best model + binary at a pinned commit; certification ladder at rungs spanning current strength ±400 (target 2200/2400/2500/2600, adding 2700/2800 only if the lower rung scores >40%), 100–200 games/rung, identical frozen conditions since Phase 0; pooled estimate + per-rung CIs, no multi-anchor fusion; wall-cap draw bias disclosed. Success = pooled ≥2400 with LCB ≥2300; the 2700 claim requires a 2700 rung with finite bounds and score-implied estimate ≥2700. Full evidence chain (arena JSONs, gate JSONs, attribution tables, disk audits) published to git. **Duration:** ~4 days at campaign end.

---

## 4. `campaign_v2` autopilot configuration (every knob, minted once)

| Knob | Value | Why / trap avoided |
|---|---|---|
| OUT_ROOT | fixed, non-timestamped, e.g. `/workspace/piebot_campaign_v2` | isolated lineage root (C11); resume-safe |
| HOURS | **1440 (60 days)** | deadline minted ONCE (autopilot.py:2426; C10); must cover campaign **+ 15-day reserve**; no extension tool; hand-edit forbidden by invariant |
| `--initial-checkpoint` + `--initial-checkpoint-weights-only` | dense checkpoint per Section 2.4 rule (cycle-98 dense if it exists, else the protected cycle-83 bootstrap artifact) | quants are not checkpoints (`_DIRECT_CHECKPOINT_FORMATS`, train_torch.py:63-66); weights-only ⇒ **fresh Adam** (mandatory at objective transitions) |
| INITIAL_ACTIVE_MODEL | cycle-98 accepted quant | protected forever; prod pattern (script:12-28) |
| INITIAL_ACTIVE_MODEL_BLEND_PERCENT | per Phase 0 ladder (50 if net beats PST-only; lower/0 otherwise) | F4: blend read from measurement, never docs |
| hidden_dim | **64, passed explicitly at every entry point** | F9: run_pipeline CLI default is 16 |
| learning_rate / warm_start_learning_rate | **0.002 / 0.001, explicit** (+ cross-cycle decay per Pilot E winner, hook autopilot.py:2609-2613) | F2: custom-profile omission falls back to 0.03 |
| continue_optimizer_state | per Pilot E (default True; fresh-Adam restarts if ablation wins) | existing flag, no code change |
| epochs / batch_size | 1 / 4096 (change only if Pilot E shows epochs-2 wins on frozen data) | one change at a time |
| teacher_relabel_depth | **7** (script pin edited at :204) | with node cap below; stronger labels |
| RELABEL_MAX_NODES | **p95 of measured depth-5 node cost** (Phase 0 measurement e) | never median (truncates ~50% below depth 5); achieved-depth stamping makes provenance honest |
| min_teacher_depth | **5 (unchanged)** | objective-identity field (script:312); no identity change; honest under achieved-depth stamping |
| teacher_relabel_every | 2 | keep label coverage (every=4 would halve it and stress the oversampling quota) |
| teacher_relabel_threads | **32** | pinned to the autopilot lane (fixes the slot-arithmetic bug); relabel determinism is (input, threads)-dependent (relabel_jsonl.rs:280-293) |
| SELFPLAY_PARALLEL_GAMES | **32** (script:34) | the unmentioned knob that would stomp the reserved lanes |
| selfplay games/cycle | 8000 | unchanged; cycle-time budget ≤40 min verified at cycle 1 |
| selfplay_openings | staged curated book (~2,500 FENs) + **loaded-count>0 assertion** | C1; silent-empty fallback (mod.rs:494-521) |
| adjudication | resign \|cp\|≥900 ×8 plies, 15% no-resign fraction; draw \|cp\|≤10 ×40 plies past move 40 | C7; attacks 45% threefold / 7.6% truncation |
| actor TT / node caps | values from Pilot C A/B (TT via `set_tt_capacity_mb`; caps CLI-exposed) | F11 resolved engine-first, then this as its own gated change |
| tau / dirichlet / temp knobs | unchanged (1.0→0.1/24; 0.30/0.25/12) | book supplies diversity; one change at a time; tau tightening is a later single-change arm |
| teacher_lag_cycles | 0 | C8 external-teacher knob is the designated pivot, not lag (near-no-op given promotion history) |
| Gate | 24-game screen (mean>0) + **SPRT confirmation** (α=β=0.05, min 48/max 300 pairs), 150ms, threads 1, gate_parallel 12 (inside the 32-slot lane; stages are sequential), noise 12/top-5, paired openings **from the curated gate book (C2)**, PST stage regression-veto, bootstrap 20k@95% retained for evidence | all gate knobs **frozen at mint for lineage life** (F8); launcher never edited mid-lineage |
| retain_full_cycles / replay_window | from measured bytes: `retain = min(8, floor(0.6 × verified_free / per_cycle_bytes))`; `replay = min(6, retain)` | D: replay silently shrinks if > retain; retain=0 disables all cleanup — never |
| Binary deploys | only at cycle boundaries, one per window, from pinned pushed commits, **binary SHA recorded in an out-of-band provenance log** | gate_identity contains no binary SHA — provenance must be explicit |
| Backups | daily off-box snapshot (Mac) of state, accepted quants, gate/anchor evidence; evidence dir in git | single-box SPOF |

---

## 5. Measurement protocol

- **Anchor conditions, frozen once in Phase 0 and never changed:** SF16 binary sha `00628bd9c9855c1b7ff93d7f8d51b413586cdc6336fe637f9a14a61531a05aca`; UCI_LimitStrength, Threads=1; TC 60+0.5; wall cap 900s (raised once); clamp rejection active (SF advertises ~1320–3190; ladder ceiling ~3190); SF sha + conditions recorded in every arena JSON; resumable atomic JSONs binding both binary SHAs + model.
- **Rung placement:** always within ~±400 of expected strength (single-anchor logistic; finite bounds). Start 1320/1500; walk upward with measured strength. Per-rung CIs reported; pooled multi-rung estimate via the wrapper; **no multi-anchor MLE fusion**.
- **Cadence:** 2-rung checkpoint every 4–5 days on the 6-slot arena lane (~10–17h, concurrent with cycles); full 3-rung milestone ladders at gates; certification ladder at the end. **Engine-delta claims always ladder BOTH binaries** at matched model/blend (~15–20h) — budgeted at consolidation checkpoints (≈weekly), not per-promotion. Total arena consumption <5% of compute by construction.
- **Gate design:** the 24/96 gate is replaced at the v2 boundary (WP4): screen kept as resource filter; confirmation becomes GSPRT on per-pair deltas (bounds calibrated from historical gate JSONs; α=β=0.05; min 48/max 300 pairs). Implemented + tested **before** the root mint; frozen thereafter. The one-time re-gating triggered by the identity change is trivial on a fresh root.
- **Promotion rules:** (a) search changes — CLAUDE.md workflow + the A/B power spec (1000-game LCB>0; speed = parity + NPS); (b) training candidates — autopilot SPRT gate + blend ladder, evidence JSONs retained; (c) binary deploys — cycle boundary, one per window, pinned pushed commit, provenance logged; anchor re-baselined at every binary swap (cross-binary Elo deltas valid only at matched model/blend/rung); (d) new lineages (h128, objective changes) — equal-TIME controlled match + anchor mini-ladder as adoption authority, F10 relaunch mechanism if gate and anchor disagree.
- **Attribution:** per-milestone table — search = binary deltas at fixed model; training = model deltas at fixed binary; compounding interaction left unattributed. Drives the 70/30 rule.
- **Diagnostics:** teacher-agreement on the frozen 10k FENs re-run every ~10 cycles (fixed-point tripwire, threshold in Section 7); data-shape metrics (threefold %, truncation %, unique-FEN rate, top-1 policy mass) per cycle from shard stats.

---

## 6. Disk / retention plan

1. **Verify capacity first (F6)** — all math uses the measured number (150GB claim vs 109GiB reference; ~120GB free reported pre-verification).
2. **Measure per-cycle bytes in Phase 0** (selfplay_jsonl + jsonl_relabel dominate; jsonl_train is hardlinks; nnue files are MBs). No retention number is set before this measurement.
3. **Budget:** steady-state footprint ≤ 60% of verified free space. `retain_full_cycles = min(8, floor(0.6 × free / B))`; `replay_window = min(6, retain)` — replay collection silently skips deleted dirs (autopilot.py:1174-1177). Pilot roots: retain 4, replay ≤4, deleted entirely after their verdicts (evidence copied to `evidence/` first).
4. **Never deleted:** accepted quants, bootstrap artifacts, gate/anchor evidence, `autopilot_state.json`. Anything worth preserving from non-accepted cycles is copied outside `cycles/` before retention reaches it.
5. **Two concurrent roots (h128 phase)** only if `2 × retention × B` fits the budget with the ≥30GiB launcher preflight margin; otherwise strictly sequential with v2 paused at a cycle boundary.
6. **Audits:** weekly `df`/`du` check against the model; retention failure aborts autopilot loudly (exit 2) — supervisor alerts on non-zero exit. Daily off-box snapshots (state, quants, evidence — all small).

---

## 7. Decision tree (post-deadline measurements → branches)

**Branch A — baseline ladder (model vs PST-only, absolute level):**
- Net (blend 100) > PST-only → weights-only bootstrap from the Section 2.4 dense source; blend per live-state value.
- PST-only ≥ net → bootstrap with low/zero initial blend; data-gen fixes outrank net scaling; weights-only bootstrap still used (cheap); fresh weights only if the net is clearly net-negative at all blends.
- Baseline ≥1300 → milestone dates hold; baseline ≤1100 → shift all milestone names down 100 and add 2 reserve days to wave 1.

**Branch B — compute split (corrected rule — the data-first inversion is fixed):**
- **Relabel dominates (≥60%)** → the actor-budget raise (Pilot C) is **cheap** — front-load it; teacher node cap sized tightly at p95; AVX2 prioritized (directly shrinks the dominant term).
- **Selfplay dominates** → actor-budget raise is expensive — size Pilot C caps conservatively and A/B the cycle-time impact; deeper teacher is comparatively cheap.

**Branch C — per-cycle bytes:** sets `retain_full_cycles`/`replay_window` per Section 6. If B > ~8GB/cycle: pilots at 2000 games/cycle; consider games/cycle reduction for v2 only as a last resort (single change, observed).

**Branch D — teacher agreement (fixed 10k FENs):**
- ≥90% best-move match → self-distillation ceiling binding: prioritize teacher-strength arms (engine deploys, depth-7 node-capped teacher) and diversity (book) over optimizer arms; **arm the C8 external-teacher pivot** (WP7).
- <90% → learner under-fit: Pilot E results (LR decay, fresh-Adam, epochs) move earlier in the v2 profile.
- **Standing tripwire (every ~10 cycles):** agreement creeping back toward ≥95% **and** promotion drought ≥15 cycles → execute WP7 before burning more cycles.

**Branch E — disk = ~109GiB:** single root only until h128 decision; retain sized to the smaller number; h128 runs sequential, never concurrent.

**Post-launch branches:**
- v2 first 15 cycles: promotions ≥1 and data-shape green → scale. Zero promotions + data-shape green → WP7 (C8). Data-shape red (book not loading, threefold still >30%) → fix the specific pilot regression before anything else — the pilots make the culprit identifiable.
- The defined middle (the data-first hole, closed): search deploys delivering Elo while training promotions lag → **not** a failure state; continue engine track at 70%, run WP7 on the 30%, re-evaluate at the next milestone.
- M-gate misses: one milestone missed by <100 Elo → extend that phase up to 3 days from reserve, re-run the tripwire diagnostics, continue. Missed by ≥100 or two consecutive misses → freeze the lagging track, reallocate per attribution table (floor dropped if post-M3).
- h128 gate/anchor disagreement → F10 relaunch mechanism (WP9).

---

## 8. Calendar — first month (dates; Day 1 = Aug 6), plus reserve

**Week 0/1 (Aug 5–12): wind-down, battery, wave-1 start.**
- Aug 5: F7 commit+push; branch + pin; Mac starts S1 (PVS) TDD; battery scripts + C9 patches prepared with tests; book curation.
- Aug 6: S1 screen on Mac; ladder wrapper + pooled estimation done; v2 launcher draft + contract-test skeleton.
- Aug 7: 03:54Z run expires untouched; snapshot; verify disk + CPU affinity; read blend; deploy measurement commit; build `uci`; install SF16 (verify sha); wall cap raised once; battery starts (ladder 1320/1500 ×3 configs, split, du, agreement, depth-5 node distribution).
- Aug 8: battery lands; **branch decisions recorded; slack day — no minting**; box: S1 1000-game confirmation; Mac: S2 (RFP).
- Aug 9–12: Mac arms S3–S6 sequentially with retune budget; box confirmations; AVX2 parity + bench on box (WP3); WP4 gate rework development; Pilot A (book) mini-root; Pilot D offline relabel validation.

**Week 2 (Aug 13–19): pilots, consolidation, launch.**
- Aug 13–15: Mac arms S7–S10 as time allows (honest target: 5–6 promoted total); Pilot B (adjudication) mini-root; Pilot C (actor budget) A/B; Pilot E GPU ablation; WP4 SPRT + gate book finished and tested.
- Aug 15–16: **consolidation:** promote surviving arms into `alphabeta.rs`, reset temp to re-export, full acceptance + perft, pin + push; engine-delta ladder **both binaries** at 1320/1500 (~15–20h) → **M1 gate: delta ≥ +150** (miss → 2 more days of arm iteration from reserve).
- Aug 16–17: **mint + launch `campaign_v2`** (Section 4 config; slips 1–3 days from reserve if WP4 is not green — never launches on an unfrozen gate).
- Aug 18–19: v2 cycle-1 verification (openings assert, adjudication rows, bytes, cycle time ≤40 min); slot partition holds (32/8/6).

**Week 3 (Aug 20–26): compounding.**
- v2 continuous; first binary-deploy window (wave-1 stragglers + AVX2 if gated) at a cycle boundary with provenance; anchor checkpoint mid-week; teacher-agreement re-run at cycle ~10; attribution table v1; **M2 gate (~Aug 24–26): pooled anchor ≥1600** → 70/30 reallocation applied; miss → Section 7 branches.

**Week 4 (Aug 27–Sep 2): scale + first contingencies.**
- v2 continuous; WP7 (C8) if tripwire fired; WP8 (dedup) if duplicate rate rose; wave-2 arms begin on Mac (staged movegen, singular extensions first); h128 GPU pre-screen on frozen data; anchor checkpoint; **M3 gate (~Sep 1–2): pooled ≥1900** (miss <100 → +3 days from reserve; miss ≥100 → freeze lagging track per attribution).

**Weeks 5–6 (Sep 3–19), outline:** h128 decision + maturation window if green (WP9, disk-gated); wave-2 deploys at boundaries; M4 (~Sep 10): pooled ≥2200; certification ladder + freeze + evidence publication (~Sep 15–19) → success ≥2400 pooled, LCB ≥2300.

**Reserve (Sep 20–Oct 4, 15 days — pre-planned base case):** continue the highest-measured-slope track under identical gates (most likely: more v2/h128 cycles + remaining wave-2 arms); re-run certification only if a promotion clears the internal gate by a wide margin; final honest report with per-rung CIs and per-track attribution. The 2700 claim is made only if a 2700 rung yields finite bounds and a score-implied estimate ≥2700 — otherwise the report states the measured number and the next bets (hidden-256, C8 ensemble teacher, remaining search backlog) for any follow-on budget.

---

### Trap-closure checklist (all named traps, owner = launch contract tests unless noted)
F1 depth pin + min_teacher_depth coherent (depth 7 / min 5 + achieved-depth stamping) · F2 explicit LR everywhere · F3 verified before any objective work (weights-only = new-root bootstrap; in-place reset = weights dropped) · F4 blend from live state · F5 compute split measured before P3 sizing · F6 disk verified before retention math · F7 handoff committed first · F8 gate knobs frozen at mint; launcher never edited mid-lineage · F9 hidden_dim explicit at every entry point · F10 relaunch-with-adjudicated-incumbent mechanism · F11 engine-first, actor raise separately gated · C10 HOURS = campaign + reserve (60d) minted once · openings loaded-count>0 assertion (mod.rs:494-521) · no invented "measured" numbers (cycle ≈16 min until re-measured) · one change per arm, everywhere, including composed launches (pilots make v2's composition attributable).