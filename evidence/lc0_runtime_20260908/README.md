# Minimal LC0 production runtime candidate

This candidate adds verified raw-archive eviction, a 478,000,000,000-byte capacity ceiling, and 720-hour training configuration to the isolated production-based trainer branch. It retains the previously tested exact-output trainer throughput patch. The existing LC0 pretraining migration helper now requires 720 hours; self-play migration behavior is unchanged.

Exactly 15 storage/720 source and test files were copied byte-for-byte from `5c60821`, plus the two-line LC0 migration/test correction. `selected_sources.json` records each origin hash; `scope.json` lists all 17 changed source/test paths from candidate base `a50a075` and the full 20-path source/test difference from production `b517c47`.

The default deployer still retains raw files unless explicitly enabled. The prospective Supervisor configuration selects `EVICT_RAW=1`, `DISK_CAPACITY_GB=478`, and `HOURS=720`. The learner budget starts after corpus preparation. The existing evaluation cadence remains in effect; this candidate does not enable the proposed 48/72-hour cadence.

Fresh candidate verification passed with no skips:

- NNUE: 385 tests, exit0, 26.57 seconds.
- Scripts: 139 tests, exit0, 10.87 seconds.

The suites ran independently with OMP/MKL/OpenBLAS limited to one thread each. `validation.json` binds exact commands, log hashes, and identical source/test hashes before and after execution. This minimal branch intentionally excludes later backup/ranked/relative-UCI tools and their tests, so its script count differs from the larger root branch.

The entire `PieBot` tree is unchanged from production, Git tree `5c6360ddbead51609f71aa70737a57956531ffe2`. The production UCI arena, ladder, and anchor-monitor scripts are byte-identical. The ranked monitor remains absent from this checkout, matching production's source tree; its separate checkout is untouched. No Rust or arena source was copied from the newer root branch, and no binary was built or replaced.

The original qualifying Rust logs are preserved here and their SHA-256 values reverified: 218 tests for locked all-targets, 209 for locked all-targets/all-features, and 25 Criterion cases in each. Production, candidate base, and the originally qualified implementation commit have the same complete `PieBot` tree. These Rust checks were reused, not rerun; `rust_reuse.json` records the binding.

The 720-hour migration correction has its own failing and passing test logs (28 relevant migration tests). Earlier historical proposal/audit records remain unchanged and cannot be reused as a new source-pin authorization.

This is a local source candidate only. No push, remote operation, source-pin change, or deployment occurred. Production cutover still requires its explicit source-pin approval and verified operational preflights; ongoing arenas retain their existing UCI binary hashes.

Source/test/evidence metadata whitespace checks pass. The two original qualifying Rust logs retain their final blank lines verbatim to preserve their previously recorded SHA-256 hashes.
