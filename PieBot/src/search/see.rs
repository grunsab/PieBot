use cozy_chess::{Board, Color, Piece, Square, BitBoard};

// Helper to create a single-square bitboard
#[inline]
fn square_bb(sq: Square) -> BitBoard {
    BitBoard::EMPTY | sq.into()
}

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

#[inline]
fn bb_contains(bb: BitBoard, target: Square) -> bool {
    for sq in bb {
        if sq == target {
            return true;
        }
    }
    false
}

fn piece_at_square(board: &Board, sq: Square) -> Option<(Color, Piece)> {
    // Check which color occupies this square
    let color = if bb_contains(board.colors(Color::White), sq) {
        Color::White
    } else if bb_contains(board.colors(Color::Black), sq) {
        Color::Black
    } else {
        return None; // Empty square
    };

    // Find which piece type
    for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop,
                    Piece::Rook, Piece::Queen, Piece::King] {
        if bb_contains(board.pieces(piece), sq) {
            return Some((color, piece));
        }
    }
    None
}

// Bitboard-based attack generation for SEE
#[inline]
fn get_bishop_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    cozy_chess::get_bishop_moves(sq, occupied)
}

#[inline]
fn get_rook_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    cozy_chess::get_rook_moves(sq, occupied)
}

#[inline]
fn get_queen_attacks(sq: Square, occupied: BitBoard) -> BitBoard {
    get_bishop_attacks(sq, occupied) | get_rook_attacks(sq, occupied)
}

#[inline]
fn get_knight_attacks(sq: Square) -> BitBoard {
    cozy_chess::get_knight_moves(sq)
}

#[inline]
fn get_king_attacks(sq: Square) -> BitBoard {
    cozy_chess::get_king_moves(sq)
}

#[inline]
fn get_pawn_attacks(sq: Square, color: Color) -> BitBoard {
    cozy_chess::get_pawn_attacks(sq, color)
}

// Find all pieces of a given color that attack a target square
fn get_attackers(board: &Board, target: Square, color: Color, occupied: BitBoard) -> BitBoard {
    let our_pieces = board.colors(color);

    let mut attackers = BitBoard::EMPTY;

    // Pawns
    let pawns = board.pieces(Piece::Pawn) & our_pieces;
    for sq in pawns {
        if bb_contains(get_pawn_attacks(sq, color), target) {
            attackers |= square_bb(sq);
        }
    }

    // Knights
    let knights = board.pieces(Piece::Knight) & our_pieces;
    for sq in knights {
        if bb_contains(get_knight_attacks(sq), target) {
            attackers |= square_bb(sq);
        }
    }

    // Bishops and Queens (diagonal attacks)
    let bishops_queens = (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) & our_pieces;
    let bishop_atks = get_bishop_attacks(target, occupied);
    attackers |= bishops_queens & bishop_atks;

    // Rooks and Queens (straight attacks)
    let rooks_queens = (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) & our_pieces;
    let rook_atks = get_rook_attacks(target, occupied);
    attackers |= rooks_queens & rook_atks;

    // King
    let king = board.pieces(Piece::King) & our_pieces;
    for sq in king {
        if bb_contains(get_king_attacks(sq), target) {
            attackers |= square_bb(sq);
        }
    }

    attackers
}

// Find the least valuable attacker of target square for given color
fn least_valuable_attacker(board: &Board, target: Square, color: Color, occupied: BitBoard) -> Option<(Square, Piece)> {
    let attackers = get_attackers(board, target, color, occupied) & occupied;  // Mask with occupied!

    // Check each piece type in order of value (cheapest first)
    for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        let piece_attackers = board.pieces(piece) & attackers;
        if let Some(sq) = piece_attackers.into_iter().next() {
            return Some((sq, piece));
        }
    }

    None
}

pub fn see_gain_cp(board: &Board, mv: cozy_chess::Move) -> Option<i32> {
    // Bitboard-based SEE: compute static exchange evaluation without move generation
    let stm = board.side_to_move();
    let to_sq = mv.to;
    let from_sq = mv.from;

    // Get the captured piece value (or 0 if moving to empty square)
    let captured_val = piece_at_square(board, to_sq)
        .map(|(_, p)| piece_value(p))
        .unwrap_or(0);

    // Get the attacker piece
    let (_, attacker_piece) = piece_at_square(board, from_sq)?;

    // Track material gains in the exchange sequence
    // gains[0] = value captured by initial move
    let mut gains: Vec<i32> = vec![captured_val];

    // Simulate the exchange: track occupied squares as pieces are captured
    let mut occupied = board.occupied();

    // Remove the initial attacker from occupied
    occupied ^= square_bb(from_sq);

    // The target square is now occupied by the initial attacker
    let mut current_occupant_val = piece_value(attacker_piece);
    let mut side = if stm == Color::White { Color::Black } else { Color::White };

    // Continue the exchange until no more attackers
    loop {
        // Find least valuable attacker from current side
        if let Some((sq, piece)) = least_valuable_attacker(board, to_sq, side, occupied) {
            let attacker_val = piece_value(piece);

            // Calculate gain: we capture current_occupant_val, then subtract what we gained so far
            // This represents the material swing from the perspective of alternating sides
            let gain = current_occupant_val - *gains.last().unwrap();
            gains.push(gain);

            // Remove this attacker from occupied
            occupied ^= square_bb(sq);

            // Update state for next iteration: attacker is now the occupant
            current_occupant_val = attacker_val;
            side = if side == Color::White { Color::Black } else { Color::White };
        } else {
            // No more attackers for this side
            break;
        }
    }

    // Minimax fold from the end: each player chooses whether to stop or continue
    // Stockfish-style fold: gains[i] = -max(-gains[i], gains[i+1])
    for i in (0..gains.len().saturating_sub(1)).rev() {
        let a = -gains[i];
        let b = gains[i + 1];
        let m = if a > b { a } else { b };
        gains[i] = -m;
    }

    Some(gains[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::{Board, Square};

    #[test]
    fn see_detects_bad_exchange_rook_x_pawn_on_h7() {
        // FEN from user: after Rxh7, ...Kxh7 wins the rook; SEE must be negative.
        let fen = "6k1/2R4p/6p1/8/6K1/6P1/8/8 w - - 3 38";
        let board = Board::from_fen(fen, false).unwrap();
        let mut rxh7 = None;
        board.generate_moves(|ml| {
            for m in ml {
                if m.from == Square::C7 && m.to == Square::H7 {
                    rxh7 = Some(m);
                    break;
                }
            }
            rxh7.is_some()
        });
        let m = rxh7.expect("Rxh7 must be legal in this position");
        let see = see_gain_cp(&board, m).expect("SEE must return some");
        assert!(
            see < 0,
            "SEE should be negative for losing exchange, got {}",
            see
        );
    }
}
