use cozy_chess::{BitBoard, Board, Color, Piece, Square};

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

fn piece_at_square(board: &Board, sq: Square) -> Option<(Color, Piece)> {
    let piece = board.piece_on(sq)?;
    let color = board.color_on(sq)?;
    Some((color, piece))
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

    // Pawns: a pawn of `color` at `sq` attacks `target` iff `sq` is attacked by a pawn of `!color` at `target`
    let pawns = board.pieces(Piece::Pawn) & our_pieces;
    let pawn_attackers = get_pawn_attacks(target, !color) & pawns;

    // Knights
    let knights = board.pieces(Piece::Knight) & our_pieces;
    let knight_attackers = get_knight_attacks(target) & knights;

    // Bishops and Queens (diagonal attacks through current occupied)
    let bishops_queens = (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) & our_pieces;
    let bishop_atks = get_bishop_attacks(target, occupied) & bishops_queens;

    // Rooks and Queens (straight attacks through current occupied)
    let rooks_queens = (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) & our_pieces;
    let rook_atks = get_rook_attacks(target, occupied) & rooks_queens;

    // King
    let king = board.pieces(Piece::King) & our_pieces;
    let king_attackers = get_king_attacks(target) & king;

    pawn_attackers | knight_attackers | bishop_atks | rook_atks | king_attackers
}

// Find the least valuable attacker of target square for given color
fn least_valuable_attacker(
    board: &Board,
    target: Square,
    color: Color,
    occupied: BitBoard,
) -> Option<(Square, Piece)> {
    let attackers = get_attackers(board, target, color, occupied) & occupied; // Mask with occupied!
    if attackers.is_empty() {
        return None;
    }

    // Check each piece type in order of value (cheapest first)
    for &piece in &[
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
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

    // Get the attacker piece
    let (_, attacker_piece) = piece_at_square(board, from_sq)?;

    // Get the captured piece value (or 0 if moving to empty square)
    let mut captured_val = piece_at_square(board, to_sq)
        .map(|(_, p)| piece_value(p))
        .unwrap_or(0);

    // Simulate the exchange: track occupied squares as pieces are captured
    let mut occupied = board.occupied();

    // En passant: the victim is not on the destination square, so reading the
    // destination alone scores the capture as winning nothing. The pawn sits
    // on the destination file at the mover's original rank, and clearing it
    // matters beyond the material -- it can open an x-ray onto `to_sq`.
    if attacker_piece == Piece::Pawn && captured_val == 0 && from_sq.file() != to_sq.file() {
        let victim_sq = Square::new(to_sq.file(), from_sq.rank());
        if piece_at_square(board, victim_sq).map(|(_, p)| p) == Some(Piece::Pawn) {
            captured_val = piece_value(Piece::Pawn);
            occupied ^= square_bb(victim_sq);
        }
    }

    // Track material gains in the exchange sequence on the stack (max 32 captures, zero heap allocations)
    let mut gains = [0i32; 32];
    gains[0] = captured_val;
    let mut gain_count: usize = 1;

    // Remove the initial attacker from occupied
    occupied ^= square_bb(from_sq);

    // The target square is now occupied by the initial attacker -- or, on a
    // promotion, by the promoted piece. Valuing it as a pawn both understates
    // the gain and makes a recapture look far too cheap.
    let mut current_occupant_val = piece_value(attacker_piece);
    if let Some(promoted) = mv.promotion {
        gains[0] += piece_value(promoted) - piece_value(Piece::Pawn);
        current_occupant_val = piece_value(promoted);
    }
    let mut side = if stm == Color::White {
        Color::Black
    } else {
        Color::White
    };

    // Continue the exchange until no more attackers
    loop {
        // Find least valuable attacker from current side
        if let Some((sq, piece)) = least_valuable_attacker(board, to_sq, side, occupied) {
            let attacker_val = piece_value(piece);

            // Calculate gain: we capture current_occupant_val, then subtract what we gained so far
            // This represents the material swing from the perspective of alternating sides
            let gain = current_occupant_val - gains[gain_count - 1];
            if gain_count < 32 {
                gains[gain_count] = gain;
                gain_count += 1;
            }

            // Remove this attacker from occupied
            occupied ^= square_bb(sq);

            // Update state for next iteration: attacker is now the occupant
            current_occupant_val = attacker_val;
            side = if side == Color::White {
                Color::Black
            } else {
                Color::White
            };
        } else {
            // No more attackers for this side
            break;
        }
    }

    // Minimax fold from the end: each player chooses whether to stop or continue
    // Stockfish-style fold: gains[i] = -max(-gains[i], gains[i+1])
    for i in (0..gain_count.saturating_sub(1)).rev() {
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
