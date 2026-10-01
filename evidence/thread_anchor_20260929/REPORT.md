# PieBot: eight threads versus one against Stockfish 18

Completed September 30, 2026. **Eight threads gained 72.2 Elo over one thread, with a joint paired-bootstrap 95% interval of +26.3 to +119.4 Elo.** The interval excludes zero and supports a gain under these test conditions.

| PieBot threads | Games | Wins | Draws | Losses | Score | Elo versus opponent | Nominal anchor-scale estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | 1 | 60 | 39 | 31.0% | −139.0 | 3051.0 |
| 8 | 100 | 5 | 71 | 24 | 40.5% | −66.8 | 3123.2 |

The opponent was Stockfish 18 with `UCI_LimitStrength=true` and `UCI_Elo=3190`. **3190 is its nominal UCI strength setting, not a published CCRL rating.** The last column simply adds the measured score-derived Elo offset to that setting; it is not a CCRL or Lichess rating estimate.

## Conditions

- This Mac (`Mac16,8`) has eight performance cores and four efficiency cores. macOS scheduled the engine's eight workers; they were not pinned to specific cores.
- Whole-game clocks: 60 seconds plus 0.5 seconds per move. One game ran at a time.
- Fifty matched openings, each played with reversed colors for both settings: four games per opening, 200 total. Execution order alternated in balanced ABBA blocks.
- Both settings used the same frozen PieBot Lazy SMP executable and `lc0_chunk_00034965.nnue`, with NNUE enabled, blend 75 and Hash 256 MiB. Only PieBot's `Threads` option differed.
- Stockfish stayed at one thread and Hash 256 MiB, with pondering and tablebases disabled.
- Games ran from 2026-09-29 22:37:43 UTC to 2026-09-30 07:30:54 UTC, about 8 hours 53 minutes.

## Estimate and uncertainty

For each arm, Elo relative to the opponent is `400 × log10(score / (1 − score))`, with a draw worth half a point. The reported gain subtracts the one-thread estimate from the eight-thread estimate.

The bootstrap samples 50 openings with replacement and keeps all four games for each sampled opening together. Each resample recalculates both arms and their Elo difference. The percentile interval uses 10,000 resamples with seed 20260929. This preserves pairing across colors and thread settings rather than treating the two arm estimates as independent.

The positive interval supports an eight-thread advantage on this Mac, model, opponent and time control. It does not establish the same gain at other time controls or against other opponents. Both settings used the new Lazy SMP implementation: this experiment does not measure new eight-thread Lazy SMP against the previous eight-thread root-splitting implementation.

## Final audit

All 200 games were replayed from their opening FENs. Every move was legal; final positions, results and recorded termination reasons matched. Each arm contained exactly 100 unique planned games with 50 games of each color. Opening/color plans, the sequential schedule and frozen input hashes passed verification. The point estimate and joint interval were independently reproduced.

| Termination | One thread | Eight threads |
| --- | ---: | ---: |
| Checkmate | 40 | 29 |
| Claimable threefold repetition | 60 | 71 |
| Crashes, time forfeits, or adjudications | 0 | 0 |

Both full Rust configurations finished before play: default 251 tests and all-features 241 tests, across 72 test/benchmark executables each. The all-features optional-backend `uci_stop` check failed initially and passed unchanged on a verified retry. Both attempts remain archived. macOS executable-startup holds were also observed during validation; no platform protections were disabled. See the [final Lazy SMP validation report](../lazy_smp_20260929.md).

## Evidence and reproduction

- `arm_1t.json` and `arm_8t.json`: unmodified game records, plans, configurations and per-arm summaries.
- `comparison.json`: original final aggregate results; `audit.json`: independent checks and SHA-256 hashes of the raw records.
- `manifest.json`: frozen engine, model, opening and source hashes. The original driver hash refers to `run.initial.py`; `recovery.json` records the validation-retry handling update in `run.py` before games began.
- `openings.json`, driver/recovery scripts and `test_runner.py`: archived reproduction sources. These scripts retain paths to the original working output directory.
- `validation.json` and `rust_validation.tar.gz`: accepted validation snapshot, original Rust result records, complete logs and unchanged retry evidence.
- `SHA256SUMS.json`: checksums of this evidence bundle.

Original frozen binaries and run workspace remain at `out/thread_anchor_20260929`. Re-run the independent audit from the repository root with `python3 out/thread_anchor_20260929/audit_final.py`; it verifies existing evidence without playing additional games. No completed game records were edited or replayed by the match runner.
