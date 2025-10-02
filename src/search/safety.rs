use cozy_chess::{Board, Move};

/// Returns true if making `mv` (assumed quiet) leaves the moved piece hanging:
/// in the resulting position, the opponent has a capture on `mv.to` with
/// non-negative Static Exchange Evaluation (SEE) above a threshold.
///
/// Threshold is in centipawns (e.g., 150..250). Higher means stricter filter.
pub fn is_hanging_after_move(board: &Board, mv: Move, loss_thresh_cp: i32) -> bool {
    // Only consider quiets; for captures, SEE is handled elsewhere.
    let mut child = board.clone();
    child.play(mv);
    // Check if opponent can profitably capture on mv.to
    let mut blunder = false;
    child.generate_moves(|ml| {
        for m2 in ml {
            if m2.to == mv.to {
                if let Some(gain) = crate::search::see::see_gain_cp(&child, m2) {
                    if gain >= loss_thresh_cp {
                        blunder = true;
                        break;
                    }
                }
            }
        }
        blunder
    });
    blunder
}

/// Returns true if making `mv` allows the opponent, in the resulting position,
/// to immediately win a large amount of material via any capture (not just on
/// the destination square). This is useful to detect moves that expose a heavy
/// piece elsewhere (e.g., queen) to a winning tactical reply like ...Nxf3+.
///
/// Threshold is in centipawns; use a high value (e.g., 500) to focus on heavy losses.
pub fn exposes_heavy_loss_after_move(board: &Board, mv: Move, loss_thresh_cp: i32) -> bool {
    let mut child = board.clone();
    child.play(mv);
    let mut severe = false;
    child.generate_moves(|ml| {
        for m2 in ml {
            if let Some(gain) = crate::search::see::see_gain_cp(&child, m2) {
                if gain >= loss_thresh_cp {
                    severe = true;
                    break;
                }
            }
        }
        severe
    });
    severe
}

/// Returns true if the given position is stalemate (no legal moves and not in check).
pub fn is_stalemate(board: &Board) -> bool {
    let mut has_legal = false;
    board.generate_moves(|_| {
        has_legal = true;
        true
    });
    if has_legal {
        return false;
    }
    (board.checkers()).is_empty()
}

/// Returns true if applying `mv` results in a stalemate for the opponent.
pub fn is_stalemate_after_move(board: &Board, mv: Move) -> bool {
    let mut child = board.clone();
    child.play(mv);
    is_stalemate(&child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::{Board, Square};

    #[test]
    fn detects_a_stalemating_move_from_user_fen() {
        // From user: there exists a move that stalemates; discover and confirm.
        let fen = "7R/4k1p1/6B1/1PN5/3PN3/P3BP2/6P1/R3K3 w Q - 3 33";
        let board = Board::from_fen(fen, false).unwrap();
        let mut found = None;
        board.generate_moves(|ml| {
            for m in ml {
                if is_stalemate_after_move(&board, m) {
                    found = Some(m);
                    break;
                }
            }
            found.is_some()
        });
        assert!(
            found.is_some(),
            "expected at least one stalemating move from this position"
        );
    }
}
