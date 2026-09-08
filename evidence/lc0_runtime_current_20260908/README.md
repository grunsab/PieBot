# Current-runtime encoder reconciliation

Qualified locally on 2026-09-08. No commit, transfer, remote action, or production change was performed by this qualification.

Base: `9c432db72bff8d406765caafc8fe0dfff096fa97` (current runtime with four parallel downloads). The only four source/test changes are copied byte-for-byte from reviewed candidate `a13ad857067891f33ee391e159531d183ed91af1`:

| File | SHA-256 |
| --- | --- |
| `training/nnue/features_v2.py` | `c000d828a9fd7f91611dc7009b46b6ebc95062d87d83152f1a8fe971cd214560` |
| `training/nnue/lc0_bin.py` | `c774f36dadd0582c4fc101a01875712d83cd955e3bf21872cb39265913b3e0a6` |
| `training/nnue/tests/test_features_v2_throughput.py` | `fbb960cf7cd18c8b7acb6eb795d8dc38422e30735f87e42b780cf8c2de8bf163` |
| `training/nnue/tests/test_lc0_fen_throughput.py` | `7aef5ece893b08a6a5cdda7efa7313a6bbaa3040f7c1d354b0c41b15d9e639e5` |

TDD: the two test files were copied first. Against unchanged base runtime, 11 tests ran with exactly two expected failures: feature collection scanned 20 times instead of once; FEN construction scanned 64 squares instead of zero calls to the legacy piece-mask scanner. After copying the two reviewed encoder files, all 11 passed. Exact RED/GREEN commands and log hashes are retained.

Fresh full suites ran sequentially using `/private/tmp/piebot-lc0-venv/bin/python`, with OMP/MKL/OpenBLAS threads set to 1:

| Suite | Passed | Skipped | Exit | Wall seconds | Log SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| nnue | 403 | 0 | 0 | 29.107 | `002ee61cabdae1317f531aa8abe9c70d015b06e11fba008586679d9c526e21cd` |
| scripts | 142 | 0 | 0 | 11.947 | `77b5ac28c3071c4da3d67b234ace1608ac6c3454833915a35ebb2ca0ecdbb90b` |

The full source/test snapshots before and after qualification match. Entire `PieBot`, `scripts`, and `deploy` contents match the current-runtime base, including launcher `DOWNLOAD_CONCURRENCY` default 4, 720 training hours, raw eviction, and the decimal 478 GB capacity ceiling. The protected arena and monitor paths remain unchanged; the separately hosted ranked monitor is absent from this production source tree. No corpus, model, optimizer, objective, or game protocol was changed.

The entire production `PieBot` tree remains `5c6360ddbead51609f71aa70737a57956531ffe2`. Rust was not rebuilt or rerun: verified inherited logs contain 218 default and 209 all-features passing tests, zero failures/ignored, plus 25 Criterion test-mode successes each. Their exact log hashes and source bindings are in `rust_reuse.json`.

`python -m training.nnue.features_v2` remains byte-identical to `PieBot/tests/data/halfkp_dp_fixture.json` (SHA-256 `4b7de7a8be3e378796e1f79e981ab71c63d9ff65bc02ff07b367733454578c43`); generated output was saved separately. These validation durations are test wall times, not throughput benchmarks.

`qualify.py` performs the scoped qualification using the unchanged inherited `validate_candidate.py` helpers. `validation.json`, baseline/before/after snapshots, selected source provenance, and logs contain the reproducible commands and hashes. All child test processes exited successfully. Root owns review, committing, and the approved production cutover.
