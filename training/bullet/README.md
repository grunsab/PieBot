# Bullet trainer for arch-v3 networks

Trains PieBot's arch-v3 networks (the `PIENNQ03` file the engine loads) with
[bullet](https://github.com/jw1912/bullet) (MIT) on Stockfish-format
binpacks. The aim is to train on the ~200 billion Leela positions that have
been relabelled with Leela's BT4 network, rather than the 0.7 billion
positions `training/nnue/train_v3.py` reads from our own converted LCZero
archives.

The network is the same one `train_v3.py` trains:

- 16 king buckets (or 32), mirrored horizontally, with kings as features;
- 768 features per bucket;
- a 1024-wide SCReLU accumulator, with the side to move's half first;
- 8 output buckets chosen by piece count, `(pieces - 2) / 4`;
- quantised with QA = 255 and an evaluation scale of 400; the output weights
  are i16 at QB = 4096 (file version 4, the default) or i8 at QB = 64
  (version 1, with `export --file-version 1`).

Exported files are byte-for-byte what `train_v3.export` writes from the same
weights, so the engine, gating scripts and fixtures need no change.

This is a standalone crate, not part of the engine's workspace. bullet is
pinned to commit `c004ebf04025edf91c6ccac47e72d561062a46b5` (2026-09-27).

## Build

Build on a machine with an NVIDIA GPU and CUDA (tested with CUDA 13.2 and an
RTX 4080):

```bash
cd training/bullet
CUDA_PATH=/usr/local/cuda cargo build --release --features cuda
```

Without `--features cuda` the crate still builds, and `inspect`,
`compare-q`, `features` and `export` work on the CPU, but training does not.

## Commands

All commands take `--key value` options. `--data` accepts files and
directories, and reads every `.binpack` file in a directory.

```bash
B=target/release/piebot-bullet

# Train. A checkpoint is written every --save-rate superbatches to
# --out/<net-id>-<n>/, holding raw.bin, optimiser_state/ and a float
# quantised.bin.
$B train --data /workspace/bt4 --out /workspace/bt4_run --net-id b16 \
  --superbatches 2000 --threads 8

# Convert a checkpoint to the engine format (file version 4; add
# --file-version 1 for the legacy i8 head).
$B export --input /workspace/bt4_run/b16-200/quantised.bin --output b16-200.nnue

# Resume: weights and optimiser state from a checkpoint, with the learning
# rate schedule picked up at the given superbatch.
$B train ... --resume /workspace/bt4_run/b16-200 --start-superbatch 201

# Checks.
$B inspect --data file.binpack       # entry count, filter statistics, sample entries
$B compare-q --plain X.binpack --q X.q.binpack
$B features --data file.binpack      # bullet inputs vs train_v3's feature map
$B evalfens --checkpoint DIR --fens fens.txt --save net.nnue   # float evals as JSON
```

### Training options

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--buckets` | 16 | King buckets: 16 (train_v3's layout) or 32 (one per mirrored square) |
| `--hidden` | 1024 | Accumulator width |
| `--superbatches` | 2000 | Each superbatch is `--batches-per-superbatch` × `--batch` positions; the default is 6104 × 16384, about 100M, so 2000 superbatches is 200B positions |
| `--lr`, `--final-lr-frac` | 0.001, 0.03 | Cosine decay from `lr` to `lr × frac`, after `--warmup-batches` (1000) of linear warm-up |
| `--wdl` | 0.2 | Weight of the game result in the target, as in train_v3 |
| `--loss` | bce | `bce`: cross-entropy on the logit, as in train_v3. `mse`: bullet's usual squared error of the sigmoid |
| `--score` | q | How the binpack score becomes the target (see below) |
| `--min-ply`, `--keep-checks`, `--keep-captures` | 8, off, off | Filters |
| `--threads` | 8 | Data-loading threads |
| `--buffer-mpos` | 16 | Shuffle buffer, in millions of positions (32 bytes each) |
| `--no-factoriser` | off | Turn off the shared 768-input factoriser. It helps training and is folded into the bucketed weights on export, so it costs nothing in the engine |
| `--measure-loader` | off | Report data-loader throughput without training |

## Data

Only Leela games relabelled with BT4 are used. The Stockfish self-play sets
(`vondele/master-binpacks_relabel`) are excluded, and so are the original
`linrock/*` binpacks, whose scores come from Stockfish.

| Dataset | Files | GB |
| :--- | ---: | ---: |
| `vondele/linrock_relabel_1` (Leela test77–80, 2021–22) | 12 | 197.4 |
| `vondele/linrock_relabel_2` (Leela test80, 2023) | 12 | 141.8 |
| `vondele/from_kaggle_2_relabel` (T60/T70) | 5 | 110.7 |
| `vondele/from_kaggle_1_relabel` (`leela96`) | 5 | 97.7 |
| `xushawn/test80-bt4-relabel` (test80, 2024; ODbL) | 2 | 17.7 |
| **Total** | **36** | **565.2** |

```bash
training/bullet/download_leela_bt4.sh /workspace/bt4            # resumable; JOBS=4 at a time
DRY_RUN=1 training/bullet/download_leela_bt4.sh /workspace/bt4  # list files and total size only
```

The script resolves files through the Hugging Face API and takes one variant
of each. Every dataset holds two versions of the same positions; I checked
this on a 10.6M-entry sample of `test60-2021-11-nov`, where every position
was identical between the two:

- **`.binpack`**: the score is the relabeller's mapping of Q,
  `cp = 660.6 q / (1 - 0.9751875 q^10)` (`QToCentipawns` in
  [xu-shawn/lc0](https://github.com/xu-shawn/lc0/commit/6ca985b81afe85f90bd79687f653f601941fa223)).
  Inverting it recovers the `.q` file's Q within 0.00077 on every entry.
- **`.q.binpack`**: the score is Q × 32000. These files are about 26% larger
  (3.04 bytes per entry, against 2.41).

So the script takes the plain files.

It also skips files under 5 GB, as Triumviratus' recipe does, and the one
test80 month that exists both whole and split into `.part_N` files. That
leaves 565 GB at about 2.4–2.7 bytes per entry, roughly 220 billion entries.

**Filters**, the same as `lc0_to_v3bin.py`. These keep 62–69% of the
entries in the samples, so one pass over everything is about 140–150 billion
training positions. An entry is dropped if:

- its score is 32002, which minimised binpacks use to mark a position to skip;
- it is before ply 8;
- the side to move is in check;
- the recorded move is a capture.

**Target** (`--score q`): the binpack score is mapped back to Q, then to
`400 · ln((1 + q) / (1 − q))`. Bullet's target `sigmoid(score / 400)` then
equals `(1 + q) / 2`, the eval term `train_v3.py` uses. It is blended with
the game result at `--wdl` 0.2.

**Loading:** the loader opens every file at once. It reads each chunk
(about 1 MB of whole games) from a file picked at random in proportion to
its size, then shuffles in a 16M-position buffer. A file that ends, or is
cut off mid-chunk, is read again from the start.

## Recipe for ~200B positions

- **Network:** 16 buckets, 1024 wide, factoriser on, BCE, WDL 0.2.
- **Batch:** 16,384 positions, 6,104 batches per superbatch, 2,000
  superbatches.
- **Learning rate:** 1e-3 with 1,000 warm-up batches, then cosine decay to
  3e-5.
- **Checkpoints:** every 20 superbatches (2B positions). Export and gate each
  one against the current default with `scripts/experiments/uci_match.py`, on
  a local machine as before. Bullet's value trainer has no validation loss,
  so games are the measure.
- **Disk:** 565 GB of data plus about 0.3 GB per checkpoint.
- **Speed:** on the RTX 4080, while it was also running two other training
  jobs, training ran at 1.65–2.1M positions/s, so 200B would take about 26–34
  hours. The loader alone reached 3.7–5.1M positions/s on 6 threads with the
  box's CPUs saturated, so the GPU is the limit. A dedicated GPU should be
  faster, but that wasn't measured. Larger batches were also not tried.

## Parity

| Check | Result |
| :--- | :--- |
| `features`: bullet's `ChessBucketsMirrored` inputs vs train_v3's `piece_features` | 0 mismatches over 300k real positions (16 buckets) and 200k (32 buckets) |
| Exporter vs `train_v3.export` on the same random float weights | identical bytes (`training/nnue/tests/test_bullet_export.py`) |
| Bullet's float eval vs a numpy forward pass of the exported floats over train_v3's features | max 0.000 cp over 60 FENs |
| `train_v3.quant_eval_fen` (which the engine matches) vs the dequantised weights | max 0.96 cp |
| The engine loading an exported net | loads and searches |
| Unit tests (`cargo test --lib`) | Q mapping round trip; target equals `(1 + q) / 2`; layouts; numpy-style rounding |

Run the Python parity tests with:

```bash
PIEBOT_BULLET_BIN=training/bullet/target/release/piebot-bullet \
PIEBOT_BULLET_CHECKPOINT=/path/to/checkpoint python3 -m unittest -v training/nnue/tests/test_bullet_export.py
```

**Quantisation error.** With file version 1 the gap between a float net and
its PIENNQ03 file came almost entirely from the i8 output weights at QB = 64.
File version 4 stores them as i16 at QB = 4096. Measured on the current
default (`v3_full_lin`, 200 positions: 100 from random games, 100 from the
hold-out set):

| File version | Mean | Standard deviation | Mean absolute | Max |
| :--- | ---: | ---: | ---: | ---: |
| 1 (i8 output weights) | +1.5 cp | 23.6 cp | 18.7 cp | 74.7 cp |
| 4 (i16 output weights) | −0.8 cp | 4.0 cp | 3.2 cp | 10.3 cp |

What remains is the input layer's rounding at QA = 255 (4.3 cp standard
deviation on its own; the version-4 output weights alone add 0.3 cp).
