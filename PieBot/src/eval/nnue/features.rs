use cozy_chess::{Board, Color, Piece, Square};

pub const HALFKP_PIECE_ORDER: [Piece; 5] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
];

const KING_SQUARES: usize = 64;
const PIECE_SQUARES: usize = 64;
const COLORS: usize = 2;
const NON_KING_PIECES: usize = HALFKP_PIECE_ORDER.len();

/// Original PieBot HalfKP layout. Each color's non-king pieces are keyed only
/// to its own king, yielding one active feature per non-king piece.
pub const HALFKP_DIM: usize = COLORS * KING_SQUARES * NON_KING_PIECES * PIECE_SQUARES;

/// Full perspective-aware layout. From each king's perspective, every white
/// and black non-king piece is represented, yielding two active features per
/// non-king piece.
pub const HALFKP_V2_DIM: usize = COLORS * KING_SQUARES * (COLORS * NON_KING_PIECES) * PIECE_SQUARES;

#[inline]
pub fn halfkp_dim() -> usize {
    HALFKP_DIM
}

#[inline]
pub fn halfkp_v2_dim() -> usize {
    HALFKP_V2_DIM
}

#[inline]
fn square_to_index(sq: Square) -> usize {
    sq as usize
}

#[inline]
fn legacy_idx_for(side: Color, k_idx: usize, piece_idx: usize, sq_idx: usize) -> usize {
    let side_off = if side == Color::White { 0 } else { 1 };
    (((side_off * 64 + k_idx) * HALFKP_PIECE_ORDER.len() + piece_idx) * 64) + sq_idx
}

#[inline]
fn full_idx_for(
    perspective: Color,
    k_idx: usize,
    piece_color: Color,
    piece_idx: usize,
    sq_idx: usize,
) -> usize {
    let perspective_off = if perspective == Color::White { 0 } else { 1 };
    let piece_color_off = if piece_color == Color::White { 0 } else { 1 };
    let colored_piece_idx = piece_color_off * NON_KING_PIECES + piece_idx;
    (((perspective_off * KING_SQUARES + k_idx) * (COLORS * NON_KING_PIECES) + colored_piece_idx)
        * PIECE_SQUARES)
        + sq_idx
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfKpSchema {
    Legacy,
    FullPerspective,
}

impl HalfKpSchema {
    #[inline]
    pub fn from_input_dim(input_dim: usize) -> Option<Self> {
        match input_dim {
            HALFKP_DIM => Some(Self::Legacy),
            HALFKP_V2_DIM => Some(Self::FullPerspective),
            _ => None,
        }
    }

    #[inline]
    pub fn dim(self) -> usize {
        match self {
            Self::Legacy => HALFKP_DIM,
            Self::FullPerspective => HALFKP_V2_DIM,
        }
    }

    pub fn active_indices(self, board: &Board) -> Vec<usize> {
        match self {
            Self::Legacy => legacy_active_indices(board),
            Self::FullPerspective => full_perspective_active_indices(board),
        }
    }

    /// Return only the active feature(s) contributed by one non-king piece.
    ///
    /// The legacy schema contributes one feature, keyed by the piece owner's
    /// king.  The full-perspective schema contributes two, one keyed by each
    /// king.  Keeping this mapping beside the full extractors prevents the
    /// incremental evaluator from duplicating the schema arithmetic.
    #[inline]
    pub(super) fn piece_indices(
        self,
        white_king: usize,
        black_king: usize,
        piece_color: Color,
        piece: Piece,
        square: Square,
    ) -> PieceFeatureIndices {
        let piece_idx = non_king_piece_index(piece)
            .expect("HalfKP features are defined only for non-king pieces");
        let square_idx = square_to_index(square);
        match self {
            Self::Legacy => {
                let king = if piece_color == Color::White {
                    white_king
                } else {
                    black_king
                };
                PieceFeatureIndices::one(legacy_idx_for(piece_color, king, piece_idx, square_idx))
            }
            Self::FullPerspective => PieceFeatureIndices::two(
                full_idx_for(Color::White, white_king, piece_color, piece_idx, square_idx),
                full_idx_for(Color::Black, black_king, piece_color, piece_idx, square_idx),
            ),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PieceFeatureIndices {
    indices: [usize; 2],
    len: u8,
}

impl PieceFeatureIndices {
    #[inline]
    const fn one(index: usize) -> Self {
        Self {
            indices: [index, 0],
            len: 1,
        }
    }

    #[inline]
    const fn two(first: usize, second: usize) -> Self {
        Self {
            indices: [first, second],
            len: 2,
        }
    }

    #[inline]
    pub(super) fn len(self) -> usize {
        self.len as usize
    }

    #[inline]
    pub(super) fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len()]
    }
}

#[inline]
fn non_king_piece_index(piece: Piece) -> Option<usize> {
    match piece {
        Piece::Pawn => Some(0),
        Piece::Knight => Some(1),
        Piece::Bishop => Some(2),
        Piece::Rook => Some(3),
        Piece::Queen => Some(4),
        Piece::King => None,
    }
}

fn king_square_index(board: &Board, color: Color) -> usize {
    let square = (board.colors(color) & board.pieces(Piece::King))
        .into_iter()
        .next()
        .expect("legal HalfKP position must contain both kings");
    square_to_index(square)
}

fn legacy_active_indices(board: &Board) -> Vec<usize> {
    let wk_idx = king_square_index(board, Color::White);
    let bk_idx = king_square_index(board, Color::Black);
    let mut out = Vec::with_capacity(32);
    for (side, k_idx) in [(Color::White, wk_idx), (Color::Black, bk_idx)] {
        for (piece_idx, piece) in HALFKP_PIECE_ORDER.iter().enumerate() {
            let pieces = board.colors(side) & board.pieces(*piece);
            for square in pieces {
                out.push(legacy_idx_for(
                    side,
                    k_idx,
                    piece_idx,
                    square_to_index(square),
                ));
            }
        }
    }
    out
}

fn full_perspective_active_indices(board: &Board) -> Vec<usize> {
    let wk_idx = king_square_index(board, Color::White);
    let bk_idx = king_square_index(board, Color::Black);
    let mut out = Vec::with_capacity(64);
    for (perspective, k_idx) in [(Color::White, wk_idx), (Color::Black, bk_idx)] {
        for piece_color in [Color::White, Color::Black] {
            for (piece_idx, piece) in HALFKP_PIECE_ORDER.iter().enumerate() {
                let pieces = board.colors(piece_color) & board.pieces(*piece);
                for square in pieces {
                    out.push(full_idx_for(
                        perspective,
                        k_idx,
                        piece_color,
                        piece_idx,
                        square_to_index(square),
                    ));
                }
            }
        }
    }
    out
}

/// HalfKP(A) feature extractor: active indices for non-king pieces keyed by each side's king square.
pub struct HalfKpA;

impl HalfKpA {
    pub fn dim(&self) -> usize {
        halfkp_dim()
    }

    pub fn active_indices(&self, board: &Board) -> Vec<usize> {
        HalfKpSchema::Legacy.active_indices(board)
    }
}

/// Full perspective-aware HalfKP v2 feature extractor.
pub struct HalfKpV2;

impl HalfKpV2 {
    pub fn dim(&self) -> usize {
        halfkp_v2_dim()
    }

    pub fn active_indices(&self, board: &Board) -> Vec<usize> {
        HalfKpSchema::FullPerspective.active_indices(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_piece_indices_match_full_extractors() {
        let board = Board::from_fen("4k3/8/8/8/8/7r/4P3/4K3 w - - 0 1", false).unwrap();
        let white_king = king_square_index(&board, Color::White);
        let black_king = king_square_index(&board, Color::Black);

        for schema in [HalfKpSchema::Legacy, HalfKpSchema::FullPerspective] {
            let all = schema.active_indices(&board);
            for (color, piece, square) in [
                (Color::White, Piece::Pawn, Square::E2),
                (Color::Black, Piece::Rook, Square::H3),
            ] {
                let direct = schema.piece_indices(white_king, black_king, color, piece, square);
                assert_eq!(
                    direct.len(),
                    if schema == HalfKpSchema::Legacy { 1 } else { 2 }
                );
                assert!(direct.as_slice().iter().all(|idx| all.contains(idx)));
            }
        }
    }
}
