use cozy_chess::{Board, Color, Piece};
use std::sync::OnceLock;

fn square_index_from_str(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() != 2 {
        return None;
    }
    let f = b[0];
    let r = b[1];
    if !(b'a'..=b'h').contains(&f) || !(b'1'..=b'8').contains(&r) {
        return None;
    }
    let file = (f - b'a') as usize;
    let rank = (r - b'1') as usize;
    Some(rank * 8 + file)
}

fn piece_index(color: Color, piece: Piece) -> usize {
    let p = match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    };
    let c = if color == Color::White { 0 } else { 1 };
    c * 6 + p
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

static TABLE: OnceLock<[u64; 12 * 64]> = OnceLock::new();
static SIDE_KEY: OnceLock<u64> = OnceLock::new();

fn init_table() -> &'static [u64; 12 * 64] {
    TABLE.get_or_init(|| {
        let mut t = [0u64; 12 * 64];
        let mut seed = 0xF00D_F00D_DEAD_BEEF;
        for v in &mut t {
            seed = splitmix64(seed);
            *v = seed;
        }
        t
    })
}

fn init_side() -> u64 {
    *SIDE_KEY.get_or_init(|| splitmix64(0xABCDEF1234567890))
}

pub fn compute(board: &Board) -> u64 {
    let table = init_table();
    let mut key = 0u64;
    for &color in &[Color::White, Color::Black] {
        for &piece in &[
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            let bb = board.colors(color) & board.pieces(piece);
            for sq in bb {
                let s = format!("{}", sq);
                if let Some(idx) = square_index_from_str(&s) {
                    let pi = piece_index(color, piece);
                    key ^= table[pi * 64 + idx];
                }
            }
        }
    }
    if board.side_to_move() == Color::Black {
        key ^= init_side();
    }
    key
}
