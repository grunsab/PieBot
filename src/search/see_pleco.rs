#![cfg(feature = "board-pleco")]
use pleco::{Board as PlecoBoard, BitMove as PMove, Player, PieceType, Piece};

#[inline]
fn piece_value(pt: PieceType) -> i32 {
    match pt {
        PieceType::P => 100,
        PieceType::N => 320,
        PieceType::B => 330,
        PieceType::R => 500,
        PieceType::Q => 900,
        PieceType::K => 20000,
        _ => 0,
    }
}

/// Minimal, robust SEE using legal replies swap list on target square.
/// Returns net material gain from the side-to-move perspective in centipawns.
pub fn see_gain_cp(board: &PlecoBoard, m: PMove) -> Option<i32> {
    // If destination has no piece, consider it zero-gain capture (or promotion capture elsewhere);
    // We'll still run through as long as first move is a capture.
    if !m.is_capture() {
        // Not a capture; SEE not defined here
        return None;
    }

    let stm = board.turn();
    let to = m.get_dest();
    let from = m.get_src();

    let victim_piece = board.piece_at_sq(to);
    if victim_piece == Piece::None {
        // Destination empty (shouldn't happen for capture in Pleco), treat as zero
        return Some(0);
    }
    let victim_val = piece_value(victim_piece.type_of());

    let attacker_piece = board.piece_at_sq(from);
    if attacker_piece == Piece::None {
        return None;
    }
    let mut gains: Vec<i32> = vec![victim_val];

    // Work on a clone so we can play/unplay safely
    let mut cur = board.clone();
    cur.apply_move(m);
    let mut side = match stm { Player::White => Player::Black, Player::Black => Player::White };
    let mut current_occ_val = piece_value(attacker_piece.type_of());

    // Keep capturing back with least valuable legal attacker onto 'to'
    loop {
        let moves = cur.generate_moves();
        let mut best_mv: Option<PMove> = None;
        let mut best_attacker_val = i32::MAX;
        for mv in moves.iter() {
            if mv.get_dest() == to {
                let src = mv.get_src();
                let p = cur.piece_at_sq(src);
                if p != Piece::None && cur.turn() == side {
                    let val = piece_value(p.type_of());
                    if val < best_attacker_val { best_attacker_val = val; best_mv = Some(*mv); }
                }
            }
        }
        if let Some(mv2) = best_mv {
            // Next gain equals value of the piece currently on 'to' minus previous gain
            let next_gain = current_occ_val - *gains.last().unwrap();
            gains.push(next_gain);
            cur.apply_move(mv2);
            side = match side { Player::White => Player::Black, Player::Black => Player::White };
            current_occ_val = best_attacker_val;
        } else {
            break;
        }
    }

    for i in (0..gains.len().saturating_sub(1)).rev() {
        let alt = -gains[i + 1];
        if alt > gains[i] { gains[i] = alt; }
    }
    Some(gains[0])
}
