use cozy_chess::Board;
use piebot::search::draw::{is_fifty_move_draw, is_insufficient_material, is_threefold};

#[test]
fn recognizes_fifty_move_claim_at_one_hundred_halfmoves() {
    let before = Board::from_fen("8/8/8/8/8/8/4K3/7k w - - 99 50", false).unwrap();
    let at = Board::from_fen("8/8/8/8/8/8/4K3/7k w - - 100 50", false).unwrap();
    assert!(!is_fifty_move_draw(&before));
    assert!(is_fifty_move_draw(&at));
}

#[test]
fn recognizes_only_dead_minor_piece_endings() {
    let bare = Board::from_fen("k7/8/8/8/8/8/4K3/8 w - - 0 1", false).unwrap();
    let bishop = Board::from_fen("k7/8/8/8/8/8/4KB2/8 w - - 0 1", false).unwrap();
    let two_knights = Board::from_fen("k7/8/8/8/8/8/3NKN2/8 w - - 0 1", false).unwrap();
    let opposing_knights = Board::from_fen("k7/8/8/8/8/7n/4KN2/8 w - - 0 1", false).unwrap();
    let pawn = Board::from_fen("k7/8/8/8/8/8/4KP2/8 w - - 0 1", false).unwrap();

    assert!(is_insufficient_material(&bare));
    assert!(is_insufficient_material(&bishop));
    assert!(
        !is_insufficient_material(&two_knights),
        "two knights cannot force mate, but mating positions are still legally reachable"
    );
    assert!(!is_insufficient_material(&opposing_knights));
    assert!(!is_insufficient_material(&pawn));
}

#[test]
fn requires_three_occurrences_of_the_same_rule_position() {
    let current = Board::default();
    let one = vec![current.clone()];
    let two = vec![current.clone(), current.clone()];
    let three = vec![current.clone(), current.clone(), current.clone()];

    assert!(!is_threefold(&current, &one));
    assert!(!is_threefold(&current, &two));
    assert!(is_threefold(&current, &three));
}
