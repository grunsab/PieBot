Four concurrent downloads are implemented for the frozen LCZero TAR inventory.
The existing fetch CLI `--concurrency` now reaches snapshot mode; the supervised
launcher forwards `--download-concurrency`, default 4, configurable with
`DOWNLOAD_CONCURRENCY`. Valid values are 1–16. Direct Python callers retain the
serial default unless they explicitly pass `concurrency`.

The coordinator alone writes the manifest, saving completed files without waiting
for earlier slow files. It reserves every in-flight archive's full expected size
against the shared free-space budget before starting another download. This can
conservatively count a partly written archive twice; it never assumes that space
has already been freed. Low space waits for active jobs to settle before retrying
admission. Errors stop new admissions and drain/persist successful active jobs.
Resume rehashes completed archives, preserves the frozen inventory/order, and
retries unfinished files. Overlapping destination/staging/manifest paths fail
before publication. Existing SHA/size checks and atomic file publication remain.

Only one supervised acquisition process may own the live raw directory. Do not
start a second fetcher beside the current one or edit its manifest manually.
A stopped partial archive may restart from zero; HTTP range resume is unchanged.
Corpus preparation still begins after the download-all phase, and the complete
720-hour training budget still begins after preparation. Architecture, targets,
checkpoints, model selection and match cadence are unchanged by this feature.

Tests prove actual worker overlap, the worker limit, out-of-order persistence,
aggregate space admission, error draining, selective resume, option forwarding,
and destination collision rejection. Test network transfers are mocked; no
live throughput or speedup is claimed. The final full Python batteries and exact
unchanged Rust/source/log reuse are recorded in validation.json. The initial
RED/GREEN logs and the later manifest-overlap regression are preserved separately.

This is an isolated local candidate, not deployed. The production-compatible
variant is based on a02fd3eb09c9e179b25353f0fd6e5575441ad189, which includes the
previously prepared storage guard/eviction and exact-output trainer optimization.
Its entire production Rust tree and existing arena/anchor scripts stay unchanged.
The new parallelism delta comprises three runtime files and two test files.
No transfer, production stop, source-pin/config edit, download, or service restart
was performed from this side conversation. The main working branch is untouched.
