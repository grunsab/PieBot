The isolated trainer patch reuses the final evaluation in V2 LC0 latest-checkpoint
mode and skips unused legacy indices while retaining FEN validation and every
training row. For a warm 700,000-position chunk with 100,000 held-out positions,
batch size 16,384 and one epoch, evaluation batches fall from 150 to 100; all
43 update batches remain. This is an operation count, not a measured speedup.

[trainer_parity.json](trainer_parity.json) records source fingerprints, failing
tests before implementation, exact metric/weight/Adam/quant parity, and the passing
53 targeted tests and 313-test full NNUE suite.

Before merging, the broader full battery remains pending on this branch:

- `python3 -m unittest discover -v scripts/tests`
- `cargo test --locked --all-targets --manifest-path PieBot/Cargo.toml`
- `cargo test --locked --all-targets --all-features --manifest-path PieBot/Cargo.toml`

This branch has not been merged or deployed. No production source pin changed.
Deployment is deferred to an explicitly approved boundary compatible with LC0's
source identity. The existing source-migration utility supports self-play only;
do not manually change the LC0 pin or state to bypass that limitation.
