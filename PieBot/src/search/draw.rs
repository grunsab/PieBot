use cozy_chess::{Board, Piece};

#[inline]
pub fn is_fifty_move_draw(board: &Board) -> bool {
    board.halfmove_clock() >= 100
}

/// Threefold repetition over the game history, which INCLUDES `board` itself,
/// so three matches means the position has genuinely occurred three times.
///
/// Only the plies since the last irreversible move can match: a capture changes
/// material and a pawn push changes pawn structure, so no earlier position can
/// ever equal the current one. `halfmove_clock` counts exactly those plies, so
/// scanning further back is provably wasted work -- and this runs at every node
/// (alphabeta.rs `rule_draw`), where it was measured at ~11% of NPS.
///
/// The window is deliberately `halfmove_clock + 1` rather than `halfmove_clock`:
/// erring one ply long can only cost a comparison, whereas erring one ply short
/// could miss a real draw.
pub fn is_threefold(board: &Board, history: &[Board]) -> bool {
    let window = usize::from(board.halfmove_clock()).saturating_add(1);
    let start = history.len().saturating_sub(window);
    history[start..]
        .iter()
        .rev()
        .filter(|previous| previous.same_position(board))
        .take(3)
        .count()
        >= 3
}

pub fn is_insufficient_material(board: &Board) -> bool {
    if !(board.pieces(Piece::Pawn) | board.pieces(Piece::Rook) | board.pieces(Piece::Queen))
        .is_empty()
    {
        return false;
    }

    let bishops: Vec<_> = board.pieces(Piece::Bishop).into_iter().collect();
    let total_knights = board.pieces(Piece::Knight).len() as usize;
    let total_minors = bishops.len() + total_knights;

    if total_minors <= 1 {
        return true;
    }

    if total_knights == 0 {
        let first_color = square_color(bishops[0]);
        return bishops
            .iter()
            .all(|&square| square_color(square) == first_color);
    }

    false
}

#[inline]
fn square_color(square: cozy_chess::Square) -> usize {
    let index = square as usize;
    (index / 8 + index % 8) & 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::Move;

    /// The pre-optimisation implementation: scan the ENTIRE history. Any faster
    /// version must agree with this everywhere, or it has changed which games
    /// are drawn.
    fn naive_is_threefold(board: &Board, history: &[Board]) -> bool {
        history
            .iter()
            .rev()
            .filter(|previous| previous.same_position(board))
            .take(3)
            .count()
            >= 3
    }

    /// Play `ucis` from the start position, pushing every position (including
    /// the start and the current one) into a history, and assert the optimised
    /// and naive detectors agree at every single ply.
    fn assert_agrees_along(ucis: &[&str]) {
        let mut board = Board::default();
        let mut history = vec![board.clone()];
        assert_eq!(
            naive_is_threefold(&board, &history),
            is_threefold(&board, &history),
            "disagreement at the start position"
        );
        for (ply, uci) in ucis.iter().enumerate() {
            let mv: Move = uci.parse().expect("test move must parse");
            board.play(mv);
            history.push(board.clone());
            assert_eq!(
                naive_is_threefold(&board, &history),
                is_threefold(&board, &history),
                "disagreement after ply {} ({uci}), halfmove_clock {}",
                ply + 1,
                board.halfmove_clock()
            );
        }
    }

    #[test]
    fn bounded_scan_matches_the_full_scan_through_a_knight_shuffle() {
        // Reversible moves only: the halfmove clock keeps climbing, so the
        // window stays wide and real repetitions must still be seen.
        assert_agrees_along(&[
            "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6",
        ]);
    }

    #[test]
    fn bounded_scan_matches_the_full_scan_across_irreversible_moves() {
        // Pawn moves reset the halfmove clock to 0, collapsing the window to a
        // single ply. Nothing before them can repeat, and the bound must not
        // invent a draw or lose one.
        assert_agrees_along(&[
            // Shuffle to build repetitions, then pawn moves (clock -> 0), then
            // captures (clock -> 0 again) so the window collapses repeatedly.
            "g1f3", "g8f6", "f3g1", "f6g8", "e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "f6e4",
        ]);
    }

    #[test]
    fn a_real_threefold_is_still_detected() {
        // Same shuffle repeated until the start-of-shuffle position occurs a
        // third time; the optimised detector must say so.
        let mut board = Board::default();
        let mut history = vec![board.clone()];
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"] {
            let mv: Move = uci.parse().expect("test move must parse");
            board.play(mv);
            history.push(board.clone());
        }
        assert!(
            is_threefold(&board, &history),
            "the start position has now occurred three times and must be a draw"
        );
        assert!(naive_is_threefold(&board, &history));
    }

    #[test]
    fn a_fresh_position_is_not_a_draw() {
        let board = Board::default();
        let history = vec![board.clone()];
        assert!(!is_threefold(&board, &history));
    }
}
