NNUE training scaffold

File formats
- Dense dev format (PIENNUE1): simple f32 layers for bootstrapping and tests.
  - Header: magic 'PIENNUE1', u32 version, u32 input_dim, u32 hidden_dim, u32 output_dim.
  - Payload: f32 w1[hidden*input], f32 b1[hidden], f32 w2[out*hidden], f32 b2[out].
- Quantized format (PIENNQ01): int8/i16 with per-layer scales.
  - Header: magic 'PIENNQ01', u32 version, u32 input_dim, u32 hidden_dim, u32 output_dim, f32 w1_scale, f32 w2_scale.
  - Payload: i8 w1[hidden*input], i16 b1[hidden], i8 w2[out*hidden], i16 b2[out].

Scripts
- fetch_lc0_pgns.py: downloads LCZero self-play PGNs (latest per suite).
- fetch_lc0_bins.py: downloads recent LCZero training BIN archives with date filtering.
- process_bins.py: converts BIN/BIN.ZST files (and tar bundles) into JSONL NNUE shards.
- process_pgns.py: converts PGNs to JSONL shards with {fen, result}.
- dataloader.py: reads self-play shards and JSONL; LCZero JSONL helpers.
- exporter.py: writes PIENNUE1 dense and PIENNQ01 quantized NNUE files.

Trainer stub
- train_stub.py: reads JSONL, featurizes HalfKP active indices, trains a small ReLU NNUE-style model with Adam, and writes checkpoint/metrics.
  - Supports blended value targets:
    - outcome target from `result_q` (or `result` fallback)
    - teacher target from `value_cp` (if present)
    - blend ratio via `--teacher-mix` (0..1)
  - Useful knobs:
    - `--max-teacher-cp` teacher clipping
    - `--outcome-decay` optional ply-based discount on outcome target
    - `--adam-beta1`, `--adam-beta2`, `--adam-eps`, `--grad-clip`

End-to-end pipeline
- run_pipeline.py: one-command ingest/train/export flow.
  - Inputs:
    - Existing JSONL shards via `--jsonl-dir`, or
    - LC0 BIN/BIN.ZST/tar inputs via `--bin-inputs ...` (ingested to JSONL first).
  - Outputs:
    - Training artifacts: `train/checkpoint.json`, `train/metrics.json`
    - Runtime artifacts: `nnue_dense.nnue` (PIENNUE1), `nnue_quant.nnue` (PIENNQ01)
    - `pipeline_summary.json`
  - Optional stronger-teacher relabel pass before training:
    - `--teacher-relabel-depth <N>` enables relabeling with PieBot search at depth `N`
    - `--teacher-relabel-every <K>` relabel every `K` plies
    - `--teacher-relabel-threads`, `--teacher-relabel-hash-mb`, `--teacher-relabel-max-records`
  - Optional NNUE bootstrap for search-driven data stages:
    - self-play: `--selfplay-nnue-quant-file <path>` and `--selfplay-nnue-blend-percent <0..100>`
    - relabel: `--teacher-relabel-nnue-quant-file <path>` and `--teacher-relabel-nnue-blend-percent <0..100>`
  - Optional resume:
    - `--resume` reuses existing self-play/relabel/train/export artifacts if present (fault-tolerant reruns)
  - Optional trainer backend:
    - `--trainer-backend stub|torch|auto`
    - `--trainer-device auto|cuda|cpu`

Example:
```bash
python -m training.nnue.run_pipeline \
  --jsonl-dir data/nnue_jsonl/test80 \
  --out out/nnue_pipeline \
  --epochs 8 --batch-size 4096 --val-split 0.1 --learning-rate 0.05
```

Self-play integrated example (generates data first, then trains/exports):
```bash
python -m training.nnue.run_pipeline \
  --out out/nnue_selfplay_pipeline \
  --selfplay-games 200 \
  --selfplay-depth 4 \
  --selfplay-threads 1 \
  --selfplay-parallel-games 0 \
  --teacher-relabel-depth 8 \
  --teacher-relabel-every 4 \
  --trainer-backend auto \
  --trainer-device cuda \
  --teacher-mix 0.8 \
  --max-teacher-cp 1200 \
  --epochs 8 --batch-size 4096 --val-split 0.1 --learning-rate 0.05
```


BIN workflow
-----------
- Use `fetch_lc0_bins.py` to download recent self-play archives (optionally filtered by suite/date).
- Convert the downloaded BIN/BIN.ZST files into JSONL shards with `python -m training.nnue.process_bins --inputs <paths> --out <dir>`.
  - The converter understands raw `.bin`, `.bin.gz`, `.bin.zst`, and `.tar` bundles containing BIN files.
  - Each JSONL entry captures the canonical FEN, best move, WDL target, policy top moves, and metadata required by training.
- Training utilities (`dataloader.py`, `train_stub.py`) consume the new JSON schema while remaining backwards-compatible with legacy PGN-derived shards.

Self-play JSONL labels
----------------------
- Current self-play JSONL includes:
  - `fen`, `ply`, `result`, `result_q`
  - `played_move` (actual sampled move used in game)
  - `target_best_move` (teacher move target)
  - `best_move` (compatibility alias for `target_best_move`)
  - `value_cp` (white-perspective teacher value from search, when available)
  - `policy_top` (top root moves with probabilities, when available)

Set-and-forget autopilot
------------------------
- `autopilot.py` runs repeated self-play -> relabel -> train -> export cycles with:
  - crash-safe state file (`autopilot_state.json`)
  - single-instance lock (`autopilot.lock`)
  - automatic retry on transient failures
  - resume-safe cycle execution (`run_pipeline --resume`)
  - bootstrap gate: until a candidate NNUE beats the default non-NNUE eval in same-search head-to-head, self-play and relabel stay on the default engine
  - automatic NNUE handoff after acceptance: once a candidate is accepted, later cycles use the accepted `nnue_quant.nnue` for self-play + relabel teacher search
  - gradual NNUE ramp after acceptance: accepted model generations are used at 25%, 50%, 75%, then 100% blend for later cycles
  - replay-window training: each cycle can merge JSONL from recent prior cycles
  - game-level parallel self-play: defaults to 1 search thread/game and auto fan-out to available cores (`--selfplay-parallel-games 0`)
  - lagged teacher option: relabel can use an older accepted model (reduces tight student-teacher coupling)
  - automatic model gate: candidate NNUE is promoted only after head-to-head `compare_play` passes
    - before the first acceptance, the candidate is compared against the default PST eval
    - after the first acceptance, the candidate is compared against the active accepted NNUE
    - gate runs in model-only mode (`compare_play --same-search`) to avoid search-code confounding

Zen5 9755 (7-day) profile:
```bash
python -m training.nnue.autopilot \
  --out-root out/autopilot_zen5_7d \
  --profile zen5_9755_7d \
  --hours 168
```

Notes:
- Profile defaults favor throughput with periodic stronger-teacher relabeling.
- Current default relabel depth in autopilot is 7.
- For high-core machines, leave `--selfplay-parallel-games 0` (auto) and keep `--selfplay-threads 1` unless you intentionally trade game count for deeper per-move search.
- If CUDA is unavailable, `trainer-backend=auto` falls back to the CPU stub trainer.
- For quick validation runs, you can disable promotion gating by setting `--gate-games 0`.

Windows 11 quick start
----------------------
- Current status: the Python NNUE pipeline is Windows-friendly, and autopilot single-instance locking now works on Windows too.
- Recommended shell: PowerShell from the repo root.
- Recommended Python: 64-bit Python 3.11 or newer.
- Install Python-side dependencies with:
  - `py -3.11 -m venv .venv`
  - `.venv\Scripts\Activate.ps1`
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r training\nnue\requirements.txt`
- Install PyTorch separately using the official Windows CUDA wheel that matches your NVIDIA driver. For GPU training you want a CUDA-enabled build; otherwise `run_pipeline` will fall back to the CPU stub trainer.

Two supported ways to run
-------------------------
- Python-only training from existing JSONL shards:
  - This path does not invoke Cargo or the Rust engine binaries.
  - Example:
    - `python -m training.nnue.run_pipeline --jsonl-dir data\nnue_jsonl\test80 --out out\nnue_pipeline --trainer-backend auto --trainer-device cuda --epochs 8 --batch-size 4096 --val-split 0.1 --learning-rate 0.05`
- Full pipeline with self-play and/or teacher relabel:
  - This path invokes PieBot Rust binaries via `cargo run`.
  - Install Rust and ensure `cargo` is on `PATH`.
  - Build the training-related binaries first:
    - `cargo build --release --manifest-path PieBot\Cargo.toml --bin selfplay --bin relabel_jsonl --bin compare_play`
  - Example:
    - `python -m training.nnue.run_pipeline --out out\nnue_selfplay_pipeline --selfplay-games 200 --selfplay-depth 4 --selfplay-threads 1 --selfplay-parallel-games 0 --teacher-relabel-depth 8 --teacher-relabel-every 4 --trainer-backend auto --trainer-device cuda --teacher-mix 0.8 --max-teacher-cp 1200 --epochs 8 --batch-size 4096 --val-split 0.1 --learning-rate 0.05`

Windows verification steps
--------------------------
- Verify Python imports:
  - `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
- Verify the Rust side only when you need self-play/relabel/autopilot:
  - `cargo run --release --manifest-path PieBot\Cargo.toml --bin selfplay -- --help`
  - `cargo run --release --manifest-path PieBot\Cargo.toml --bin relabel_jsonl -- --help`
  - `cargo run --release --manifest-path PieBot\Cargo.toml --bin compare_play -- --help`
- Verify the NNUE Python tests:
  - `python -m unittest discover training/nnue/tests`

Current caveat
--------------
- The top-level repository-wide `cargo test -q` is not the right validation command for this workflow right now because `PieBot/src/bin/bench.rs` still has unrelated compile drift. The training pipeline itself uses the targeted binaries above.
