# NNUE

PieBot supports two quantised network formats, both loaded through the
`NNUEQuantFile` UCI option. The engine picks the implementation from the file
magic.

## Arch-v2 (`PIENNQ02`)

- Inputs: HalfKP, 41,024 features per perspective (king square × piece ×
  square).
- Feature transformer: 1024 wide, i16 weights; the first layer is about 84 MB.
- Activation: SCReLU on both perspectives' accumulators.
- Output: one linear head over the 2×1024 activations.

The previous default network, `models/lc0_chunk_00034965.nnue`, uses this format; the current default, `models/v3_lc0t91_b16.nnue`, is arch-v3.

## Arch-v3 (`PIENNQ03`)

- **King buckets.** Each perspective's king square is oriented (ranks flipped
  for Black) and mirrored to files a-d, then mapped to one of 16 buckets (32
  optional). The 16-bucket layout gives every square of the first two ranks
  its own bucket and groups the rest in pairs and quads.
- **Kings as pieces.** Every piece, kings included, contributes one feature
  within the bucket: `bucket × 768 + (relative colour × 6 + type) × 64 + square`,
  with the same orientation and mirror applied to the square. That is 12,288
  features at 16 buckets, and a 25 MB first layer at width 1024 instead of
  84 MB.
- **Output buckets.** 8 linear heads over the 2×1024 SCReLU activations,
  chosen by piece count: `(pieces − 2) / 4`.
- **Material experts (optional, `--experts 4`).** A mixture of experts on the
  input layer, following the description in the
  [Triumviratus](https://github.com/Tors3/Triumviratus) README: the feature
  transformer holds one full set of king-bucketed weights per material band
  (32-25, 24-17, 16-9 and 8-2 pieces), and the piece count selects the expert.
  Routing is a fixed rule, not a learned gate, and only one expert is active,
  so the layer has four times the weights at almost the cost of one. The
  implementation is our own; only the idea comes from Triumviratus.
- **Hidden stack (optional, off).** A 2×1024 → 16 → 32 → 1 stack per output
  bucket is implemented and tested. It improved hold-out loss slightly (0.6106
  against 0.6124 on a 128M-position comparison) but cut the engine from about
  2.07M to 0.67M nodes/s and lost 295 Elo in games, so released networks use
  the linear head.

Quantisation follows arch-v2: feature weights and biases ×QA (QA = 255,
i16), output weights ×QB (QB = 64, i8), output biases in the QA²·QB domain
(i32), and

```
cp = (sum + b2) × 400 / (QA × QA × QB)
```

### File format

All integers are little-endian.

```
magic            8 bytes  "PIENNQ03"
version          u32      1 = linear head, 2 = hidden stack, 3 = material experts
n_king_buckets   u32      16 or 32
bucket_layout    u8[32]   bucket of each oriented, mirrored king square (rank*4 + file)
-- version 3 only --
n_experts        u32
expert_of_pieces u8[33]   expert index by piece count
-- all versions --
hidden           u32
n_output_buckets u32
qa, qb, scale    i32 × 3
w1               i16[n_features × hidden]   feature-major
b1               i16[hidden]
w2               i8[n_output_buckets × 2·hidden]
b2               i32[n_output_buckets]
-- version 2 only --
l1, l2           u32 × 2
h1_w             i8[n_output_buckets·l1 × 2·hidden]
h1_b, h2_w, h2_b, h3_w, h3_b   f32
```

`n_features` is `n_experts × n_king_buckets × 768` (one expert unless the file
is version 3).

## Engine implementation

The arch-v3 code is in `PieBot/src/eval/nnue/network_v3.rs`, dispatched from
`network.rs`.

- **Lazy accumulators.** Making a move records the feature changes and
  prefetches the weight rows; an accumulator is built only when an evaluation
  needs it. Positions answered from the transposition table never pay for an
  update.
- **Refresh cache.** Each perspective's accumulator is keyed by expert, king
  bucket and mirror. When a move changes the key (a king crossing a bucket or
  the mirror line, or a capture crossing a material band), that perspective
  is rebuilt from a cached accumulator for the new key, patched with the
  pieces that differ since it was cached. All other moves are incremental.
- **Fused updates.** A quiet move is one pass computing `src − from + to`; a
  capture one pass over `src − mover − victim + mover'`, so the memory
  accesses for the weight rows overlap.
- **Kernels.** The update loops are plain Rust loops that the compiler
  vectorises (to AVX-512 under `target-cpu=native`); they measured faster than
  earlier hand-written AVX2 intrinsics. The SCReLU head has explicit AVX-512,
  AVX2 and NEON versions.

These are exact: tests compare incremental and lazy evaluation against full
recomputation.

## Training pipeline

1. **Data.** LCZero test91 training archives (Open Database License /
   Database Contents License). The first full run used the 1,800 most recent
   hourly archives (August-October 2026, about 205 GB).
2. **Conversion.** `training/nnue/lc0_to_v3bin.py` reads LCZero V6 records and
   drops deleted records, records with invalid Q, invalid boards, the first 8
   plies of each game, positions in check, and positions whose LCZero best move
   is a capture (about 70% of positions are kept). Each kept position is a
   32-byte record: occupancy bitboard, 4-bit piece codes, side to move, best Q,
   game result and ply. 1% of games, chosen by a stable hash, go to a hold-out
   split. The first run produced 725M training positions.
3. **Training.** `training/nnue/train_v3.py` computes features on the GPU from
   the compact records (about 600k positions/s on an RTX 4080). The target is

   ```
   0.8 × (1 + Q)/2 + 0.2 × (1 + result)/2
   ```

   (Q alone when the game outcome is not valid), with BCE loss on
   `sigmoid(cp / 400)`. Defaults: batch 16384, Adam at 1e-3 decaying
   exponentially to 3% of that. The trainer exports a `PIENNQ03` file at each
   evaluation step.

   ```bash
   python3 training/nnue/train_v3.py --data /data/lc0/bin --out runs/v3 \
     --hidden 1024 --buckets 16 --positions 3e9
   ```

The earlier arch-v2 LCZero pipeline (`fetch_lc0_*`, `lc0_corpus.py`,
`lc0_filter.py`) is described in
[history/LCZeroTraining.md](history/LCZeroTraining.md).

Stockfish and other engines are never used as label sources; LCZero's own
search values and game results are the only targets.

## Results

First full run: 16 king buckets, linear head, width 1024, 3B training
positions. Each checkpoint played the arch-v2 `lc0_chunk_00034965` net with
the same binary, 200 games at 10+0.1, one thread:

| Positions trained | Elo vs arch-v2 net [95% CI] |
| ---: | :--- |
| 250M | −59.6 [−96.8, −23.8] |
| 1.0B | +31.4 [−2.4, +65.7] |
| 2.0B | +96.2 [+62.6, +131.7] |
| 2.75B | +135.0 [+101.7, +170.8] |
| 3.0B | +123.0 [+87.1, +161.6] |
| final | +109.5 [+74.7, +146.5] |
| final, 1000-game confirmation (W409 D449 L142) | +95.1 [+80.1, +110.3] |

Against full-strength Stockfish 19 (60+0.5, 280 games, Syzygy, `Contempt 40`)
the final net scored 0 / 92 / 188, 32.9% win or draw, against 19.3% for the
arch-v2 net under the same conditions (Elo −282.6 [−313.8, −254.8]).

Variants played against `v3_lc0t91_b16` with the same binary, 10+0.1
(`evidence/nnue_v3_20261019/`):

| Variant | Games | W / D / L | Elo [95% CI] |
| :--- | ---: | :---: | :--- |
| 32 king buckets, same data and schedule | 1000 | 153 / 676 / 171 | −6.3 [−18.7, +6.2] |
| 4 material experts trained as independent weight sets | 400 | 49 / 233 / 118 | −60.5 [−82.7, −38.8] |
| Same net, file version 4 (precise output weights) | 1000 | 142 / 711 / 147 | −1.7 [−12.5, +9.1] |

The independent experts each learn from about a quarter of the data;
Consilium trains a shared base plus zero-initialised per-expert deltas
instead, which `train_v3.py --factorised` now implements. File version 4
cuts the quantisation error from 24 cp to 4 cp SD without a measurable
change in strength.

Hold-out loss reached 0.6013 at 2.75B positions. Checkpoints after about
1.25B positions are within noise of each other at 200 games.

## Tests

- `PieBot/tests/nnue_arch_v3.rs`: random games with captures, castling, en
  passant, promotions and king moves across buckets, checking eager and lazy
  incremental evaluation against full recomputation, with and without
  material experts; null moves; and exact agreement with the pure-Python
  integer reference evaluator (`train_v3.py::quant_eval_fen`) on exported
  fixtures in `PieBot/tests/data/` (`v3_fixture`, `v3e_fixture`; the hidden-stack
  fixture `v3h_fixture` allows ±1 cp for f32 rounding).
- `PieBot/tests/nnue_arch_v2.rs`: the same parity checks for arch-v2.
- `training/nnue/tests/test_lc0_to_v3bin.py` and `test_train_v3.py`: record
  encoding, GPU features against the reference, mirroring, bucket layouts,
  experts and exact export.
