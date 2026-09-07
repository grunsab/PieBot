# LC0 pre-training source migration: local review candidate

This change adds `--mode lc0-pretraining` to the sanctioned source migration
utility. It is local preparation only: no supervisor was stopped, no remote
source or pin was changed, and no deployment is authorized by this document.
The default self-play mode and its state migration behavior remain unchanged.

The LC0 mode requires explicit old/new 40-character commits, a clean repository
at the new commit, and verified fast-forward ancestry. It acquires the existing
`OUT_ROOT/launcher.lock` nonblockingly, plus existing corpus preparation and
training locks. It rejects changed lock paths, new optional locks, directory
aliases below approved roots, every training state file (including a zero-chunk
state), and learner output. It never creates self-play state or a training
clock. A frozen acquisition manifest with queued downloads and no corpus is
supported.

Before replacing the pin, the utility fsyncs an immutable prepared audit, then
checks the same snapshot and the saved audit again. The audit contains exact
bootstrap/checkpoint and incumbent hashes, exact raw manifest bytes and date
bounds, the source delta hash/name-status list, and any corpus identity,
manifest, SQLite cache files, archive metadata, and validation hashes. Raw
archives and compressed corpus chunks are **not rehashed**; their existing
checksum commitments are preserved and labelled accordingly. Uncommitted
`.part` corpus directories are excluded. Existing launcher/corpus checks still
validate payloads when resumed. Both before-pin and after-pin crash retries are
tested; a changed snapshot or missing audit is refused.

`--supervisor-stop-verified` is the caller's explicit attestation that supervisor
is stopped and its descendants have exited. The tool does not query or control
supervisor. Similarly, `required_budget_hours: 336` records a deployment
requirement; `live_budget_verified_by_tool: false` makes clear that the utility
does not inspect the live supervisor configuration. It leaves configuration,
data paths, manifests and the deferred training clock untouched.

After explicit user approval, an operator must separately verify the stopped
boundary, unchanged 336-hour launcher configuration, and absence of learners;
then stage the tested descendant source and invoke the tool with these arguments:

```text
--mode lc0-pretraining
--repo-root <approved checkout>
--out-root <existing LC0 output root>
--expected-old-commit <explicit old 40-character SHA>
--expected-new-commit <explicit new 40-character SHA>
--bootstrap-root <preserved self-play root>
--bootstrap-checkpoint <original checkpoint inside that root>
--bootstrap-checkpoint-sha256 <expected checkpoint SHA-256>
--active-model <original incumbent inside that root>
--active-model-sha256 <expected incumbent SHA-256>
--supervisor-stop-verified
```

For this acquisition lineage the eventual source delta must be restricted to
the reviewed trainer throughput change and this migration support, descended
from live commit `b517c47cb86b9f524118df6469b29d7d76495898`. The tool audits the
delta rather than enforcing that deployment-specific allowlist. Active ranked
and anchor matches use the existing UCI binary path: its bytes must remain
unchanged, without a rebuild or replacement. This is a separate operator check;
the migration utility does not certify engine binary equivalence. Stop at a
completed archive boundary where feasible because an unfinished `.tar.part`
download restarts from zero. Any training state makes this mode inapplicable;
it is not a general LC0 state identity migration.

Deterministic tests were run RED before each implementation change. The new
tests cover lifecycle refusal, held/missing/replaced/late locks, bootstrap root
and checksum validation, aliases, frozen dates, source drift, audit tampering,
and both atomic-pin crash windows. Exact final counts and log hashes are in
`pretraining_source_migration_tests.json`. These script-only changes do not
alter architecture, objective, optimizer, training duration or checkpoint
formats. Full NNUE and Rust checks for any eventual integrated deployment
candidate remain an integration gate; prior trainer-branch checks are recorded
separately in `evidence/lc0_throughput_20260907/`.
