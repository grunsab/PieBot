use piebot::board::cozy::Position;

#[test]
fn position_tracks_every_reachable_board_for_repetition_search() {
    let moves = [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();

    let position = Position::set_from_start_and_moves(&moves).expect("legal repetition");

    assert_eq!(position.history().len(), moves.len() + 1);
    assert_eq!(
        position
            .history()
            .iter()
            .filter(|board| board.same_position(position.board()))
            .count(),
        3,
        "the current position must be visible as a threefold repetition"
    );
}

#[test]
fn fen_history_starts_at_the_loaded_position() {
    let fen = "8/8/8/8/8/8/4K3/7k w - - 87 44";
    let position = Position::from_fen(fen).expect("valid FEN");

    assert_eq!(position.history().len(), 1);
    assert!(position.history()[0].same_position(position.board()));
    assert_eq!(position.board().halfmove_clock(), 87);
}
