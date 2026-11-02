# Threat inputs: research notes and proposed design (Phase 6b)

Research collected 2026-10-30. The sources are the engines' public code and
PRs, listed at the end. Anything marked *measured* is mine, not from a source.

## What the strong engines do

| | Stockfish v10 (Nov 2025) | Stockfish master (SFNNv19) | PlentyChess | Reckless |
| :--- | :--- | :--- | :--- | :--- |
| Threat features | 79,856 | 59,808 + 4,560 pawn pairs | Same as SF master | 66,864 |
| Pairs involving kings | K→K and K→Q excluded | All king pairs excluded | — | Piece→king excluded (Elo-neutral, −6.4 MB) |
| King buckets on threats | None; horizontal mirror plus perspective flip only | Same | Same | Same |
| Threat weights | i8 | i8 | i8 (+9.9 STC over i16) | i8 |
| Accumulator | Separate threat accumulator | Merged with piece features (+2.5% speed) | Merged | Separate, summed at activation |
| Width when threats were added | 1024 | — | 1792 → 640 | 1024 → 384, later 768 |
| Elo | +13.9 LTC | PP_3Wide passed LTC | +1.8 LTC (+45 at fixed nodes) | +6.9 LTC (+53 at fixed nodes) |

Common to all of them:
- **Feature:** (attacker piece and colour, attacker square, attacked piece and colour, attacked square). It counts only when the target square is occupied, so defences of friendly pieces are included.
- **Pruning:** same-type pairs are kept in one direction only. Pawn→B/Q/K and B/R→Q are excluded.
- **Index compression:** lookup tables built from empty-board attacks: a pair base, plus a per-square offset, plus the popcount of attacks below the target.
- **Incremental update:** a per-move "dirty threats" list (capacity 80) is built inside put/remove piece. It covers:
  - the moved piece's own attacks;
  - every slider that sees the square, including the x-ray target past it, which is how blocking and unblocking is handled;
  - incoming knight, pawn and king attacks.
  - Castling is at most 36 entries.
- **Refresh:** only when a king crosses the mirror line. Refresh caches hold piece state only, and threats are recomputed.
- **Active count:** *measured* on SF's 305 bench positions under SF-master rules: mean 15, max 53. The start position has 28. Buffers are sized at 128.

## Implications for PieBot at width 2048

- The fixed-node gains are large (+45 to +53). The timed gains are small (+2 to +14), so most of the benefit is spent on speed. Every engine kept its accumulator at 1024 or below after adding threats.
- At 2048 the threat matrix alone is 122 MB of i8 (59,808 × 2048). That's on top of the 50 MB of piece weights, and it will hurt the cache more than at 1024.
- **Decision gate:** before any GPU run, measure engine NPS with a random-weight threat net at 2048, and also at 1024 as a fallback.
  - If 2048 with threats is more than 25% slower than plain 2048, the second run should use threats at width 1024 and compare against the plain 2048 net in games.
  - **This needs the owner's approval, because it changes the mandated width.**

## Proposed design

1. **Feature set:** Stockfish master's 59,808 threat features (no king pairs, pawn diagonal threats only onto N and R), oriented with the existing mirror: own king file and perspective flip, no king buckets. The PP_3Wide pawn-pair set is left out of the first version.
2. **Weights:** i8 threat weights at QA = 255, clipped to ±127/255 in training, added into the existing i16 accumulator.
   - The existing piece weights stay i16 with 16 king buckets.
   - The exporter's worst-case accumulator check must add up to 128 threat rows.
3. **File format:** a new PIENNQ03 version (6). It holds the threat feature count and the i8 threat block after w1.
   - The loader gets a new branch.
   - The engine must match `train_v3.quant_eval_fen`, extended with threats, exactly.
4. **Engine:**
   - A threat module with the lookup tables.
   - A dirty-threat list built in the move-making path from the board before and after the move. It needs no changes to cozy-chess: attacks are recomputed for the changed squares and the rays through them.
   - Lazy application, matching the existing lazy accumulators.
5. **Tests (red first):**
   - feature parity with a full-board Python reference on 300k positions;
   - incremental threats equal full recomputation along random games (captures, castling, en passant, promotions, mirror crossings);
   - exact engine-vs-Python parity on an exported fixture.
6. **Trainer:**
   - Port bullet's `examples/advanced/inputs.rs` threat input (PR #530) into `training/bullet` as a second sparse input with max 128 active.
   - The `features` check must show 0 mismatches against the Python reference.
   - The exporter must be byte-identical to the Python export.

## Sources

- Stockfish SFNNv10 threat inputs: commit 8e5392d79a, PR https://github.com/official-stockfish/Stockfish/pull/6411; `src/nnue/features/full_threats.{h,cpp}`, `src/position.cpp` (`update_piece_threats`).
- Stockfish follow-ups: PR #6415 (threat speed), #6890 (merged accumulators), #6694 and #6982 (pawn threats, PP_3Wide), commit ff7cee4244 (SFNNv19).
- PlentyChess: https://github.com/Yoshie2000/PlentyChess, PRs #400 and #411; `src/threat-inputs.{h,cpp}`.
- Reckless: https://github.com/codedeliveryservice/Reckless, PRs #608, #653, #703, #710; `src/nnue/accumulator/threats.rs`.
- bullet threat input example: https://github.com/jw1912/bullet/blob/main/examples/advanced/inputs.rs (PR #530).
