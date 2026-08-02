use cozy_chess::{Color, Piece, Square};
use piebot::board::cozy::Position;

#[test]
fn apply_startpos_moves_sequence() {
    let moves = vec!["e2e4".to_string(), "e7e5".to_string(), "g1f3".to_string()];
    let pos = Position::set_from_start_and_moves(&moves).expect("legal move sequence");
    assert_eq!(
        pos.side_to_move(),
        Color::Black,
        "expected black to move after 3 plies"
    );
}

#[test]
fn apply_moves_relative_to_a_fen_position() {
    let fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1";
    let moves = vec!["e2e4".to_string()];
    let pos = Position::set_from_fen_and_moves(fen, &moves).expect("legal FEN move sequence");

    assert_eq!(pos.side_to_move(), Color::Black);
    assert_eq!(pos.board().piece_on(Square::E4), Some(Piece::Pawn));
    assert!(pos.board().colors(Color::White).has(Square::E4));
}
