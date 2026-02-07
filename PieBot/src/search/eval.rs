use cozy_chess::{Board, Color, Piece, Square};

const PAWN: i32 = 100;
const KNIGHT: i32 = 320;
const BISHOP: i32 = 330;
const ROOK: i32 = 500;
const QUEEN: i32 = 900;

fn count_piece(board: &Board, color: Color, piece: Piece) -> i32 {
    let bb = board.colors(color) & board.pieces(piece);
    bb.into_iter().count() as i32
}

// Side-agnostic material in centipawns: positive means White has more material.
pub fn material_eval_cp_side_agnostic(board: &Board) -> i32 {
    let w = Color::White;
    let b = Color::Black;
    let score = (count_piece(board, w, Piece::Pawn) - count_piece(board, b, Piece::Pawn)) * PAWN
        + (count_piece(board, w, Piece::Knight) - count_piece(board, b, Piece::Knight)) * KNIGHT
        + (count_piece(board, w, Piece::Bishop) - count_piece(board, b, Piece::Bishop)) * BISHOP
        + (count_piece(board, w, Piece::Rook) - count_piece(board, b, Piece::Rook)) * ROOK
        + (count_piece(board, w, Piece::Queen) - count_piece(board, b, Piece::Queen)) * QUEEN;
    score
}

// Material from side-to-move perspective (negamax-friendly)
pub fn material_eval_cp(board: &Board) -> i32 {
    let base = material_eval_cp_side_agnostic(board);
    if board.side_to_move() == Color::White {
        base
    } else {
        -base
    }
}

// Mate scoring helpers
pub const MATE_SCORE: i32 = 30_000;
pub const DRAW_SCORE: i32 = 0;

// Simple PSTs (from white's perspective); values in centipawns
// Lightweight, hand-rolled to encourage centralization/development
// Indexing: 0..63 = rank*8 + file, with rank/file from 0..7 for white's POV.
const PST_PAWN: [i16; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0,
    20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50,
    50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0,
];
const PST_KNIGHT: [i16; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];
const PST_BISHOP: [i16; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10, -10, 10, 10, 10, 10, 10,
    10, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];
const PST_ROOK: [i16; 64] = [
    0, 0, 5, 10, 10, 5, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0,
    0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0,
    0, 0, 0, 0, 0, 0, 0,
];
const PST_QUEEN: [i16; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0,
    0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];
const PST_KING: [i16; 64] = [
    20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20, -10, -20, -20, -20, -20, -20, -20,
    -10, -20, -30, -30, -40, -40, -30, -30, -20, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40,
    -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50,
    -40, -40, -30,
];

#[inline]
fn square_index_fast(sq: Square) -> usize {
    // Cozy-chess implements From<Square> for u8
    sq as usize
}

fn pst_value_for(board: &Board, color: Color, piece: Piece) -> i32 {
    let bb = board.colors(color) & board.pieces(piece);
    let mut sum = 0i32;
    for sq in bb {
        // iterate over squares
        let mut idx = square_index_fast(sq);
        // mirror rank for black
        if color == Color::Black {
            let r = idx / 8;
            let f = idx % 8;
            idx = (7 - r) * 8 + f;
        }
        let v = match piece {
            Piece::Pawn => PST_PAWN[idx],
            Piece::Knight => PST_KNIGHT[idx],
            Piece::Bishop => PST_BISHOP[idx],
            Piece::Rook => PST_ROOK[idx],
            Piece::Queen => PST_QUEEN[idx],
            Piece::King => PST_KING[idx],
        } as i32;
        sum += v;
    }
    sum
}

// Combined material + PST (side-to-move perspective)
pub fn eval_cp(board: &Board) -> i32 {
    let mat = material_eval_cp_side_agnostic(board);
    let pst = pst_value_for(board, Color::White, Piece::Pawn)
        - pst_value_for(board, Color::Black, Piece::Pawn)
        + pst_value_for(board, Color::White, Piece::Knight)
        - pst_value_for(board, Color::Black, Piece::Knight)
        + pst_value_for(board, Color::White, Piece::Bishop)
        - pst_value_for(board, Color::Black, Piece::Bishop)
        + pst_value_for(board, Color::White, Piece::Rook)
        - pst_value_for(board, Color::Black, Piece::Rook)
        + pst_value_for(board, Color::White, Piece::Queen)
        - pst_value_for(board, Color::Black, Piece::Queen)
        + pst_value_for(board, Color::White, Piece::King)
        - pst_value_for(board, Color::Black, Piece::King);
    let total = mat + pst;
    if board.side_to_move() == Color::White {
        total
    } else {
        -total
    }
}
