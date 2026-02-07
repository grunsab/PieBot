# Null-Move Pruning Improvement Plan

## Current Status

### What Works
- **Simple alphabeta**: Fixed TT bound calculation bug → all mate-in-3 tests pass (90/90)
- **Regular alphabeta**: Disabled null-move at depth ≤ 7 → all mate-in-3 tests pass (86/86)

### Current Implementation Issues
```rust
// Current: alphabeta.rs:361
if self.use_nullmove && depth >= 8 {
    if (board.checkers()).is_empty() {
        let nb = board.clone();
        let null_ok = std::panic::catch_unwind(/*...*/);
        if null_ok {
            let r = 2 + (depth / 4) as u32;  // R = 2..5
            let score = -self.alphabeta(&nb, depth - 1 - r, -beta, -beta + 1, ply + 1, usize::MAX);
            if score >= beta { return score; }
        }
    }
}
```

**Problems:**
1. **Hardcoded depth threshold**: `depth >= 8` is a band-aid that:
   - Disables null-move for all mate-in-3 searches (depth 7)
   - Will fail on mate-in-4+ at similar depths
   - Removes speedup even in non-tactical positions

2. **No tactical position detection**: Null-move fails in:
   - Zugzwang positions (king+pawns endgames)
   - Mate threat positions (forced sequences)
   - Low material positions (few pieces, every tempo matters)

3. **Aggressive reduction**: R = 2 + depth/4 gives R=3 at depth 9-12, which is very deep
   - Misses short tactical sequences
   - R should be lower in tactical positions

---

## Research: Null-Move Pruning Best Practices

### Core Principle
**Null-move observation**: In most positions, even giving the opponent a free move doesn't change the evaluation drastically. If we're so far ahead that even after giving a free move we still exceed beta, we can safely prune this branch.

### Known Failure Modes

1. **Zugzwang**: Positions where being forced to move worsens your position
   - Common in pawn endgames, king+pawn vs king
   - Null-move assumes "not moving is okay" → false in zugzwang

2. **Mate Threats**: Forced tactical sequences
   - Attacker has mate in N moves
   - Null-move gives defender extra tempo → breaks the mating attack
   - Example: mate-in-3 becomes mate-in-4 after null-move → search misses it at reduced depth

3. **Low-piece positions**: When few pieces remain
   - Each tempo is critical
   - Single move can swing evaluation massively

4. **Very shallow depths**: At depth 2-3
   - Reduction R=2 means depth becomes 0-1
   - Too shallow to detect threats

### Industry Best Practices (from Stockfish, Ethereal, etc.)

**Guards to prevent null-move:**
1. **Never in check** (already implemented ✓)
2. **Minimum depth**: depth >= 3 (we have depth >= 8, too conservative)
3. **Not in PV nodes** (when doing full-window PVS)
4. **Zugzwang detection**:
   - Disable if side-to-move has only king+pawns (no pieces)
   - Some engines: disable if eval < beta by margin
5. **Mate distance**: Disable if within ~10 plies of mate (either side)
6. **Verification search**: After null-move cutoff, verify with reduced search
7. **Adaptive R**:
   - R = 3 at high depth (>6)
   - R = 2 at medium depth (3-6)
   - Increase R if eval >> beta (very winning)
   - Decrease R in endgames or tactical positions

**Double null-move prevention:**
- Never do null-move if parent was null-move (parent_move_idx == usize::MAX)

---

## Proposed Improved Implementation

### Phase 1: Better Guards (Conservative)

```rust
fn should_try_null_move(&self, board: &Board, depth: u32, beta: i32, parent_move_idx: usize) -> bool {
    // 1. Never if disabled
    if !self.use_nullmove { return false; }

    // 2. Minimum depth for null-move to be useful
    if depth < 3 { return false; }

    // 3. Never in check
    if !(board.checkers()).is_empty() { return false; }

    // 4. Never if parent was null-move (double null-move)
    if parent_move_idx == usize::MAX { return false; }

    // 5. Zugzwang-prone: king + pawns only (no pieces)
    if is_zugzwang_prone(board) { return false; }

    // 6. Mate threat detection: if eval suggests nearby mate
    let eval = self.eval_current(board);
    let mate_distance = MATE_SCORE - eval.abs();
    if mate_distance < 1000 { return false; }  // within ~3 moves of mate

    true
}

fn is_zugzwang_prone(board: &Board) -> bool {
    let stm = board.side_to_move();
    let our_pieces = board.colors(stm);
    let our_king = board.pieces(Piece::King) & our_pieces;
    let our_pawns = board.pieces(Piece::Pawn) & our_pieces;

    // If we only have king + pawns (no other pieces), zugzwang is likely
    (our_pieces ^ our_king ^ our_pawns).is_empty()
}
```

### Phase 2: Adaptive Reduction

```rust
fn null_move_reduction(&self, depth: u32, eval: i32, beta: i32) -> u32 {
    let mut r = 2;

    // Increase R at high depth for more pruning
    if depth >= 7 { r = 3; }
    if depth >= 10 { r = 4; }

    // Increase R if we're very far ahead
    if eval > beta + 200 { r += 1; }

    // Cap R to avoid searching at depth 0
    r.min(depth - 1)
}
```

### Phase 3: Verification Search (Advanced)

After null-move cutoff, verify with a shallow search:

```rust
if score >= beta && depth >= 6 {
    // Verification search at reduced depth to confirm cutoff
    let v_score = self.alphabeta(board, depth - r, beta - 1, beta, ply, parent_move_idx);
    if v_score >= beta { return score; }
    // Verification failed, continue normal search
}
```

---

## Testing Strategy

### Test Suites
1. **mate-in-3** (depth 7): 86 positions ✓ passing
2. **mate-in-4** (depth 9): 100 positions (generated)
3. **mate-in-5** (depth 11): 100 positions (generated)

### Baseline Metrics (Current: depth >= 8)
- [ ] mate-in-3 @ depth 7: pass rate, avg nodes
- [ ] mate-in-4 @ depth 9: pass rate, avg nodes
- [ ] mate-in-5 @ depth 11: pass rate, avg nodes

### Experiments

**Experiment 1: Better guards (Phase 1)**
- Implement improved guards
- Test on all three mate suites
- Compare: pass rate, nodes searched, time

**Experiment 2: Adaptive R (Phase 2)**
- Add adaptive reduction
- Test on all three suites
- Compare: pass rate, nodes, time

**Experiment 3: Verification search (Phase 3)**
- Add verification
- Test on all three suites
- Compare: correctness vs performance tradeoff

### Success Criteria
1. **Correctness**: Pass ≥95% of all mate suites
2. **Performance**: Node count reduction >30% vs null-move disabled
3. **No regressions**: mate-in-3 still passes 100%

---

## Implementation Steps

### Step 1: Add Helper Functions
- [x] `is_zugzwang_prone(board)` - detect king+pawns only
- [ ] `get_mate_distance(eval)` - distance from mate score
- [ ] `null_move_reduction(depth, eval, beta)` - adaptive R

### Step 2: Refactor Null-Move Logic
- [ ] Extract null-move decision into `should_try_null_move()`
- [ ] Replace hardcoded `depth >= 8` with proper guards
- [ ] Use adaptive R instead of fixed formula

### Step 3: Add Tests
- [ ] Unit test: `is_zugzwang_prone()` on known positions
- [ ] Integration test: null-move disabled in zugzwang positions
- [ ] Regression test: mate-in-3 still passes

### Step 4: Measure and Tune
- [ ] Run all mate suites with metrics
- [ ] Tune thresholds (mate distance, eval margins)
- [ ] Document final parameters

### Step 5: Optional Verification Search
- [ ] Implement verification if needed for correctness
- [ ] Measure performance impact
- [ ] Keep only if worthwhile

---

## Expected Outcomes

### Conservative Approach (Phases 1-2)
- Null-move enabled at depth >= 3 (vs current depth >= 8)
- Proper guards prevent tactical failures
- Should solve mate-in-3/4/5 correctly
- Significant speedup on non-tactical positions

### With Verification (Phase 3)
- Higher correctness guarantee
- Slight performance cost
- Worth it if Phase 1-2 still has failures

### Fallback
If null-move still causes issues:
- Keep depth-based threshold but make it smarter
- E.g., `depth >= 3 && (depth >= 8 || !is_tactical_position())`
- Where `is_tactical_position()` combines multiple heuristics

---

## References

Chess programming wiki:
- Null Move Pruning: https://www.chessprogramming.org/Null_Move_Pruning
- Zugzwang: https://www.chessprogramming.org/Zugzwang
- Verification Search: https://www.chessprogramming.org/Null_Move_Pruning#Verification_Search

Modern engines for reference:
- Stockfish: uses verification search, mate distance, zugzwang detection
- Ethereal: adaptive R based on depth and eval
- Laser: simpler guards, relies on verification
