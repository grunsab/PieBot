use cozy_chess::{Board, Move};
use rand::Rng;
use rand::rngs::SmallRng;

/// Choose a move uniformly from the top-K ordered moves, but filter out
/// obviously losing captures using SEE (< see_thresh_cp).
/// If all candidates are filtered, fall back to the best available move.
pub fn choose_noisy_from_order_filtered(
    board: &Board,
    order: &[Move],
    topk: usize,
    rng: &mut SmallRng,
    see_thresh_cp: i32,
) -> Option<Move> {
    if order.is_empty() { return None; }
    let k = topk.max(1).min(order.len());
    // Build candidate pool with SEE filter
    let mut pool: Vec<Move> = Vec::with_capacity(k);
    // Quiet hanging threshold: avoid obviously hanging quiet moves
    const QUIET_HANGING_THRESH_CP: i32 = 200;
    // Quiet checking sacrifice: avoid if hanging by a large margin (e.g., queen sac)
    const CHECK_SAC_THRESH_CP: i32 = 500;
    for &m in order.iter().take(k) {
        // Use SEE when applicable; if SEE says strongly negative, skip.
        // Additionally, even if SEE is acceptable, skip moves that expose a heavy
        // immediate loss elsewhere (e.g., queen hang via a tactical reply).
        if let Some(gain) = crate::search::see::see_gain_cp(board, m) {
            if gain < see_thresh_cp { continue; }
            if crate::search::safety::exposes_heavy_loss_after_move(board, m, CHECK_SAC_THRESH_CP) { continue; }
            pool.push(m);
            continue;
        }
        // Non-capture (SEE not applicable). Avoid quiet moves that leave the piece hanging by SEE.
        // Skip if the move gives check (tactical), only filter quiet blunders.
        let mut child = board.clone();
        child.play(m);
        let gives_check = !(child.checkers()).is_empty();
        if !gives_check {
            if crate::search::safety::is_hanging_after_move(board, m, QUIET_HANGING_THRESH_CP) { continue; }
        } else {
            if crate::search::safety::is_hanging_after_move(board, m, CHECK_SAC_THRESH_CP) { continue; }
        }
        pool.push(m);
    }
    // If everything filtered, fall back to the very first move (best ordering)
    if pool.is_empty() { return Some(order[0]); }
    let idx = rng.gen_range(0..pool.len());
    Some(pool[idx])
}
