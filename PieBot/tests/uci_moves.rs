use cozy_chess::Color;
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
