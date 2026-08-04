use cozy_chess::{Board, Piece};

#[inline]
pub fn is_fifty_move_draw(board: &Board) -> bool {
    board.halfmove_clock() >= 100
}

pub fn is_threefold(board: &Board, history: &[Board]) -> bool {
    history
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
