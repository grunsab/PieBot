use cozy_chess::{Board, Color, Piece, Square};

pub const HALFKP_PIECE_ORDER: [Piece; 5] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
];

#[inline]
pub fn halfkp_dim() -> usize {
    2 * 64 * HALFKP_PIECE_ORDER.len() * 64
}

#[inline]
fn square_to_index(sq: Square) -> usize {
    sq as usize
}

#[inline]
fn idx_for(side: Color, k_idx: usize, piece_idx: usize, sq_idx: usize) -> usize {
    let side_off = if side == Color::White { 0 } else { 1 };
    (((side_off * 64 + k_idx) * HALFKP_PIECE_ORDER.len() + piece_idx) * 64) + sq_idx
}

/// HalfKP(A) feature extractor: active indices for non-king pieces keyed by each side's king square.
pub struct HalfKpA;

impl HalfKpA {
    pub fn dim(&self) -> usize {
        halfkp_dim()
    }

    pub fn active_indices(&self, board: &Board) -> Vec<usize> {
        let mut out = Vec::with_capacity(64);
        // King squares
        let wk_sq = (board.colors(Color::White) & board.pieces(Piece::King))
            .into_iter()
            .next()
            .unwrap();
        let bk_sq = (board.colors(Color::Black) & board.pieces(Piece::King))
            .into_iter()
            .next()
            .unwrap();
        let wk_idx = square_to_index(wk_sq);
        let bk_idx = square_to_index(bk_sq);
        // For each non-king piece on both sides
        for (side, k_idx) in [(Color::White, wk_idx), (Color::Black, bk_idx)] {
            for (pi, p) in HALFKP_PIECE_ORDER.iter().enumerate() {
                let bb = board.colors(side) & board.pieces(*p);
                for sq in bb {
                    let sq_idx = square_to_index(sq);
                    let idx = idx_for(side, k_idx, pi, sq_idx);
                    out.push(idx);
                }
            }
        }
        out
    }
}
