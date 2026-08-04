use cozy_chess::Board;
use piebot::search::zobrist;

fn board(fen: &str) -> Board {
    Board::from_fen(fen, false).expect("test FEN must be valid")
}

#[test]
fn castling_rights_are_part_of_the_position_key() {
    let with_rights = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let without_rights = board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");

    assert_ne!(
        zobrist::compute(&with_rights),
        zobrist::compute(&without_rights)
    );
}

#[test]
fn legal_en_passant_state_is_part_of_the_position_key() {
    // This is the position after 1. e4 a6 2. e5 d5. White may legally play exd6 e.p.
    let with_en_passant = board("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    let without_en_passant = board("rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3");

    assert_ne!(
        zobrist::compute(&with_en_passant),
        zobrist::compute(&without_en_passant)
    );
}

#[test]
fn equivalent_positions_have_the_same_key() {
    let early_clock = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let late_clock = board("4k3/8/8/8/8/8/8/4K3 w - - 47 91");

    assert_eq!(
        zobrist::compute(&early_clock),
        zobrist::compute(&late_clock)
    );
    assert_eq!(
        zobrist::compute(&early_clock),
        zobrist::compute(&early_clock.clone())
    );
}

#[test]
fn side_to_move_is_part_of_the_position_key() {
    let white_to_move = board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let black_to_move = board("4k3/8/8/8/8/8/8/4K3 b - - 0 1");

    assert_ne!(
        zobrist::compute(&white_to_move),
        zobrist::compute(&black_to_move)
    );
}
