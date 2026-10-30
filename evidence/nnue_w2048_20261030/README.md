# Lineage w2048: 2048-wide pairwise net on BT4-relabelled data

Goal: at least 75% win-or-draw against full-strength Stockfish 19 (60+0.5,
280 games, seed 1901, Syzygy 3-5, PieBot `Contempt 40`), confirmed by a second
280-game run (seed 1902) with the pooled 560 games at 75% or more.

The previous best, `v3pw_bt4_1500` (1024 wide), scored 0 / 152 / 128 (54.3%)
on the Ryzen 7940HX. All measurements here are on one Vast box, and they are
not comparable with the 7940HX numbers:

| Item | Value |
| :--- | :--- |
| Box | Vast, `ssh -p 40194 root@222.239.174.10` |
| CPU | i9-13900KF: 8 P-cores (CPUs 0-15, SMT pairs), 16 E-cores (CPUs 16-31); AVX2 and AVX-VNNI, no AVX-512; microcode 0x133 |
| Memory | 62 GB, no ECC |
| GPU | RTX 4090, driver 595.91.07, CUDA 13.2 |
| Disk | 2.7 TB container disk; `/workspace` is not a persistent volume |

Core layout, fixed for every match: one engine per physical P-core, with its
SMT sibling idle, paired by top clock: `--cpu-pairs 0:2,4:6,8:10,12:14`
(cores 4 and 5, CPUs 8 and 10, boost to 5.8 GHz, the others to 5.5 GHz). So 4
games run at once. The trainer runs on the E-cores (CPUs 16-31).

Binaries (`binaries_sha256.txt`):
- `uci_w2048`: the engine built from this branch. It has the new kernel and is used by every match.
- Stockfish 19: the official `stockfish-linux-x86-64-universal` build (tarball sha256 `9defc0d4…`).

## Data (`corpus_files.tsv`)

46 files, 705.6 GB. Every file is checked against the sha256 that the Hugging
Face API reports for it.

| Source | Positions from | Labels |
| :--- | :--- | :--- |
| `vondele/linrock_relabel_1`, `_2`, `from_kaggle_1_relabel`, `_2_relabel` | Leela test60-80 games | BT4-tf13tune Q |
| `xushawn/test80-bt4-relabel` 2024-01..06, `-histfix` versions only | Leela test80 games | BT4-tf13tune Q, evaluated with real move history |
| `vondele/master-binpacks_relabel` without `dfrc_n5000` | Stockfish games (fishpack32, multinet, nodes5000pv2_UHO, wrongIsRight) | BT4-tf13tune Q |

**Label check for the Stockfish-position sets.** I compared the first 300 MB
of each plain file with its `.q` twin, which stores raw BT4 Q × 32000, using
`piebot-bullet compare-q`:

| File | Entries compared | Position mismatches | Q diff > 0.002 | Worst Q diff |
| :--- | ---: | ---: | ---: | ---: |
| fishpack32 | 97,340,470 | 0 | 0 | 0.00077 |
| multinet_pv-2_diff-100_nodes-5000 | 86,676,977 | 0 | 0 | 0.00077 |
| nodes5000pv2_UHO | 87,314,615 | 0 | 0 | 0.00077 |
| wrongIsRight_nodes5000pv2 | 81,490,318 | 0 | 0 | 0.00077 |

So the scores are the relabeller's mapping of BT4 Q, not Stockfish
evaluations. The worst difference, 0.00077, is the mapping's rounding. The
same samples keep 72-76% of entries under the usual filters (ply ≥ 8, not in
check, best move not a capture).

## Engine: AVX2 / AVX-VNNI sparse first layer

Without AVX-512, `sparse_l1::product` used to fall back to a dense AVX2
product, so the pairwise net's sparsity was wasted on this CPU.
`v3pw_bt4_1500` has 9.4% nonzero activations, and 32% of its 4-byte blocks
contain at least one nonzero.

The new path:
1. lists the nonzero blocks branch-free, using a 256-entry offset table as Stockfish's `find_nnz` does;
2. multiplies only those blocks, with `vpdpbusd ymm` when AVX-VNNI is compiled in and exact i16 widening otherwise;
3. still takes the dense pass when more than 75% of blocks are nonzero.

| Measurement (one P-core) | Before | After |
| :--- | ---: | ---: |
| L1 product, n = 1024, 32% nonzero blocks | 302 ns (dense) | 147 ns |
| Engine knps, `v3pw_bt4_1500` (mean of 5 positions) | 2008 | 2232 (+11%) |
| Engine knps, 2048-wide smoke net | 1356 | 1502 |

The 2048-wide smoke net searches 32% fewer nodes per second than
`v3pw_bt4_1500` with the new kernel, inside the plan's 40% limit. A smoke net's
evaluations are meaningless, so this is only an estimate; it will be
re-measured on a trained checkpoint.

Exactness checks:
- matein3 at depths 7-14 with `v3pw_bt4_1500` loaded: 91 cases, 121,918 nodes on both builds, so the node counts match exactly;
- the kernel unit tests pass on the AVX-VNNI build and on the plain AVX2 build (`x86-64-v3`);
- engine against `train_v3.quant_eval_fen` on the 2048-wide smoke export: 300 of 300 positions exact on both builds (evaluations range from −1842 to +2022 cp).

## Trainer

- Bullet at width 2048 (pairwise 16/32, `--act-l1 0.1`) runs at about 2.55M positions/s on the 4090 with 8 loader threads on the E-cores.
- Two same-seed runs are not bit-identical. Their logged losses start identical, and the weights then drift by 9e-5 on average, which matches the GPU summing in a different order on each run. So bit-identity can't be the hardware check here.
- Instead:
  - every checkpoint's float weights are checked (finite, |w| < 64) before export and resume;
  - every export is hashed into `manifest.jsonl`;
  - the Mac copies are verified against those hashes;
  - the CPU side is checked with deterministic engine node counts.

## Threat inputs (Phase 6b), built while the plain run trains

Design and research: `threats_design.md`. The format is file version 6, as
documented in `documents/NNUE.md`.

| Part | Where | Check |
| :--- | :--- | :--- |
| Reference features (59,808, Stockfish FullThreats rules) | `training/nnue/threats.py` | 14 tests, including an independent ray-walking restatement on 400 random positions, colour-flip and file-mirror symmetry, and 28 threats per side at the start position |
| Engine features, incremental update, loader | `PieBot/src/eval/nnue/threats.rs`, `network_v3.rs`, `loader.rs` | 8 tests: features and evaluations equal the Python reference on 124 positions (`v3t_fixture.json`); along 60 random games of 200 plies, incremental changes equal the difference of full feature sets; eager and lazy evaluation equal full recomputation |
| Trainer | `train --pairwise --threats` (`training/bullet/src/threats_train.rs`), using the engine's feature map | the export equals Python's version-6 writer byte for byte; on a 64-wide GPU smoke checkpoint, bullet's float evaluation equals a numpy pass over the checkpoint floats within 0.0002 cp (60 positions) |
| Plain nets unaffected | the same binary | matein3 at depths 7-14 with `v3pw_bt4_1500`: 121,918 nodes, identical to the frozen `uci_w2048` |

### NPS decision gate (one P-core, CPU 8, 2M nodes × 2 rounds, 5 positions)

Raw output: `threat_nps/results.txt`. Binaries: `uci_w2048`
(sha 25514834…) and `uci_threats` (sha 19e6b447…).

| Net | Plain, old binary | Plain, threats binary | + threats, zero weights | + threats, random weights |
| :--- | ---: | ---: | ---: | ---: |
| `v3pw_bt4_1500` (1024) | 2203 | 2197 | 1666 | 1594 |
| `w2048-320` (2048) | 1819 | 1832 | 1373 | 1375 |

All values are mean knps.

- **Plain nets:** the threats binary costs them nothing.
- **Mean cost of threats:** −27% at 1024 and −25% at 2048.
- **Per position:** the mean leans on one fast endgame. The geometric mean of the per-position ratios is −34% at both widths. The two middlegames lose 42-48%.
- **Width doesn't change the share:** threats add about 0.18 µs per node at both widths. So a 1024 threats run would be just as much slower than plain 1024, and the width-1024 fallback in the plan buys nothing. The threats run stays at 2048.

Cost breakdown per move at 2048 (Mac, `threats::bench`, random games, about 5 raw threat changes per move):

| Part | ns |
| :--- | ---: |
| Finding changed threats | 220 |
| Mapping to features, both sides | 60 |
| Accumulator update, pieces only, one side | 81 |
| Pieces + threats, threat rows in cache, one side | 180 |
| Pieces + threats, threat rows from memory, one side | 378 |

Most of the cost is reading about 5 threat rows of 2 KB each per side, from a 122 MB matrix.

Re-measured on the P-core (`threat_nps/results2.txt`, `w2048-320`, random threat weights):

| Binary | Plain | + threats |
| :--- | ---: | ---: |
| first threats binary | 1848 (old binary) | 1381 |
| square-to-piece table, in-place diff, whole-row prefetch | 1819 | 1344 |

The whole-row prefetch was reverted, because it was slower. The other two changes are kept: they are exact, and they make no difference within noise. The remaining cost is the threat-row reads themselves.

## Phase 3: SF19 baseline on this box

`v3pw_bt4_1500` against Stockfish 19 using the reference protocol: 60+0.5, 280 games, seed 1901, Syzygy 3-5, Contempt 40, 4 games at a time on P-cores 0/2, 4/6, 8/10, 12/14, binary `uci_w2048`.

| W | D | L | Win or draw | Elo (95% CI) |
| ---: | ---: | ---: | ---: | :--- |
| 0 | 155 | 125 | **55.4%** | −167 (−190, −145) |

- **Pentanomial:** [21, 83, 36, 0, 0].
- **Draws:** 121 threefold, 17 insufficient material, 15 fifty-move, 2 stalemate.
- **Wall time:** 3.0 h.
- **Comparison:** the 7940HX measured 54.3%, so the two boxes agree within noise. This 55.4% is the baseline for the 75% target.
- **Raw results:** `pw1500_vs_sf19_s1901.json`.

## Phase 5: Gates against `v3pw_bt4_1500` (10+0.1, 200 games, `books/openings_v1.fen`, seed 77)

| Checkpoint | W | D | L | Score | Elo (95% CI) | Pentanomial |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| w2048-200 | 9 | 122 | 69 | 35.0% | −108 (−136, −80) | [11, 43, 41, 5, 0] |
| w2048-400 | 9 | 116 | 75 | 33.5% | −119 (−149, −91) | |
| w2048-600 | 11 | 127 | 62 | 37.3% | −91 (−117, −65) | |
| w2048-800 | 9 | 137 | 54 | 38.8% | −80 (−105, −55) | |
| w2048-1000 | 12 | 134 | 54 | 39.5% | −74 (−102, −47) | |

The gap closes by about 15 Elo per 200 superbatches after w2048-400. The learning rate is still high at this point, so the 2048 net isn't expected to match the finished 1024 net yet. Pentanomials for w2048-400 onward are in the gate JSONs on the box.
