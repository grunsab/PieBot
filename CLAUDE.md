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
