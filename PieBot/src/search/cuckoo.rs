//! Upcoming-repetition detection (Stockfish's cuckoo tables, after Marcel van
//! Kervinck): for every reversible piece move, the XOR of the position keys
//! before and after it is a fixed value. Storing those values lets the search
//! ask, in O(plies), whether the side to move has a move that returns to a
//! position already on the current line.
//!
//! This needs the exact Zobrist keys cozy-chess uses. They are private but
//! generated deterministically, so they are regenerated here in the same order
//! (`hash_matches_cozy_chess` checks the reconstruction).

use cozy_chess::{BitBoard, Board, Color, Piece, Square};
use std::sync::LazyLock;

struct Keys {
    /// [color][piece][square]
    piece: [[[u64; 64]; 6]; 2],
    castle: [[u64; 8]; 2],
    en_passant: [u64; 8],
    black_to_move: u64,
}

static KEYS: LazyLock<Keys> = LazyLock::new(|| {
    // cozy-chess: Pcg64Mcg, en passant first, then per colour castling rights
    // and pieces, then the side-to-move key.
    let mut state: u128 = 0x7369787465656E2062797465206E756Du128 | 1;
    let mut rand = || {
        state = state.wrapping_mul(0x2360ED051FC65DA44385DF649FCCF645);
        let rot = (state >> 122) as u32;
        let xsl = (state >> 64) as u64 ^ state as u64;
        xsl.rotate_right(rot)
    };
    let mut en_passant = [0u64; 8];
    for k in &mut en_passant {
        *k = rand();
    }
    let mut castle = [[0u64; 8]; 2];
    let mut piece = [[[0u64; 64]; 6]; 2];
    for c in 0..2 {
        for k in &mut castle[c] {
            *k = rand();
        }
        for p in 0..6 {
            for k in &mut piece[c][p] {
                *k = rand();
            }
        }
    }
    let black_to_move = rand();
    Keys {
        piece,
        castle,
        en_passant,
        black_to_move,
    }
});

const SIZE: usize = 8192;

struct Table {
    keys: Vec<u64>,
    moves: Vec<(Square, Square)>,
    len: usize,
}

#[inline]
fn h1(k: u64) -> usize {
    (k & 0x1fff) as usize
}

#[inline]
fn h2(k: u64) -> usize {
    ((k >> 16) & 0x1fff) as usize
}

fn attacks(piece: Piece, sq: Square) -> BitBoard {
    match piece {
        Piece::Knight => cozy_chess::get_knight_moves(sq),
        Piece::Bishop => cozy_chess::get_bishop_moves(sq, BitBoard::EMPTY),
        Piece::Rook => cozy_chess::get_rook_moves(sq, BitBoard::EMPTY),
        Piece::Queen => {
            cozy_chess::get_bishop_moves(sq, BitBoard::EMPTY) | cozy_chess::get_rook_moves(sq, BitBoard::EMPTY)
        }
        Piece::King => cozy_chess::get_king_moves(sq),
        Piece::Pawn => BitBoard::EMPTY,
    }
}

static TABLE: LazyLock<Table> = LazyLock::new(|| {
    let z = &*KEYS;
    let mut keys = vec![0u64; SIZE];
    let mut moves = vec![(Square::A1, Square::A1); SIZE];
    let mut len = 0;
    for c in 0..2 {
        for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            for s1 in 0..64 {
                for s2 in s1 + 1..64 {
                    let (a, b) = (Square::index(s1), Square::index(s2));
                    if !attacks(piece, a).has(b) {
                        continue;
                    }
                    let mut mv = (a, b);
                    let mut key = z.piece[c][piece as usize][s1] ^ z.piece[c][piece as usize][s2] ^ z.black_to_move;
                    let mut i = h1(key);
                    loop {
                        std::mem::swap(&mut keys[i], &mut key);
                        std::mem::swap(&mut moves[i], &mut mv);
                        if key == 0 {
                            break;
                        }
                        i = if i == h1(key) { h2(key) } else { h1(key) };
                    }
                    len += 1;
                }
            }
        }
    }
    Table { keys, moves, len }
});

/// The reversible move whose key change is `diff`, if any.
#[inline]
pub fn lookup(diff: u64) -> Option<(Square, Square)> {
    let t = &*TABLE;
    let i = h1(diff);
    if t.keys[i] == diff {
        return Some(t.moves[i]);
    }
    let j = h2(diff);
    if t.keys[j] == diff {
        return Some(t.moves[j]);
    }
    None
}

/// Force table construction (call once at startup to keep it off the clock).
pub fn init() {
    let _ = TABLE.len;
}

/// Recompute a position key from scratch with the reconstructed constants.
pub fn full_hash(board: &Board) -> u64 {
    let z = &*KEYS;
    let mut h = 0;
    for (ci, c) in [Color::White, Color::Black].into_iter().enumerate() {
        for p in Piece::ALL {
            for sq in board.colored_pieces(c, p) {
                h ^= z.piece[ci][p as usize][sq as usize];
            }
        }
        let r = board.castle_rights(c);
        if let Some(f) = r.short {
            h ^= z.castle[ci][f as usize];
        }
        if let Some(f) = r.long {
            h ^= z.castle[ci][f as usize];
        }
    }
    if let Some(f) = board.en_passant() {
        h ^= z.en_passant[f as usize];
    }
    if board.side_to_move() == Color::Black {
        h ^= z.black_to_move;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_cozy_chess() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/8/4k3/8/8/3K4/8/7R b - - 7 40",
        ] {
            let b = Board::from_fen(fen, false).unwrap();
            assert_eq!(full_hash(&b), b.hash(), "{fen}");
        }
    }

    #[test]
    fn table_holds_every_reversible_piece_move_once() {
        assert_eq!(TABLE.len, 3668);
        let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 3 10", false).unwrap();
        let mut a = b.clone();
        a.play("a1a5".parse().unwrap());
        let found = lookup(b.hash() ^ a.hash()).expect("Ra1-a5 is a reversible rook move");
        assert_eq!(found, (Square::A1, Square::A5));
    }
}
