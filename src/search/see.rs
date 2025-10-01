use cozy_chess::{Board, Color, Piece};

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

fn piece_at_str(board: &Board, sq: &str) -> Option<(Color, Piece)> {
    for &color in &[Color::White, Color::Black] {
        for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let bb = board.colors(color) & board.pieces(piece);
            for s in bb {
                if format!("{}", s) == sq { return Some((color, piece)); }
            }
        }
    }
    None
}

fn move_from_to_str(mv: cozy_chess::Move) -> (String, String) {
    let s = format!("{}", mv);
    (s[0..2].to_string(), s[2..4].to_string())
}

pub fn see_gain_cp(board: &Board, mv: cozy_chess::Move) -> Option<i32> {
    // Swap-off list SEE using only legal moves to the target square.
    // Returns net material gain in centipawns from the side-to-move perspective.
    let stm = board.side_to_move();
    let (from_s, to_s) = move_from_to_str(mv);
    let captured0 = piece_at_str(board, &to_s)?;
    let attacker0 = piece_at_str(board, &from_s)?;
    let mut gains: Vec<i32> = vec![piece_value(captured0.1)];

    let mut cur = board.clone();
    cur.play(mv);
    let mut side = if stm == Color::White { Color::Black } else { Color::White };
    let mut current_occ_val = piece_value(attacker0.1);

    loop {
        // Find least valuable attacker from 'side' that captures back on to_s
        let mut best_mv: Option<cozy_chess::Move> = None;
        let mut best_attacker_val = i32::MAX;
        cur.generate_moves(|ml| {
            for m in ml {
                let (_, to2) = move_from_to_str(m);
                if to2 == to_s {
                    let (src, _) = move_from_to_str(m);
                    if let Some((c, p)) = piece_at_str(&cur, &src) {
                        if c == side {
                            let v = piece_value(p);
                            if v < best_attacker_val { best_attacker_val = v; best_mv = Some(m); }
                        }
                    }
                }
            }
            false
        });
        if let Some(m2) = best_mv {
            // Next gain is the value of the piece captured on to_s (current occupant) minus previous gain
            let next_gain = current_occ_val - *gains.last().unwrap();
            gains.push(next_gain);
            cur.play(m2);
            side = if side == Color::White { Color::Black } else { Color::White };
            current_occ_val = best_attacker_val;
        } else {
            break;
        }
    }

    // From the end, choose optimal stopping point
    for i in (0..gains.len().saturating_sub(1)).rev() {
        // Stockfish-style fold: gains[i] = -max(-gains[i], gains[i+1])
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
                if m.from == Square::C7 && m.to == Square::H7 { rxh7 = Some(m); break; }
            }
            rxh7.is_some()
        });
        let m = rxh7.expect("Rxh7 must be legal in this position");
        let see = see_gain_cp(&board, m).expect("SEE must return some");
        assert!(see < 0, "SEE should be negative for losing exchange, got {}", see);
    }
}
