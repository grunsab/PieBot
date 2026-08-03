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
