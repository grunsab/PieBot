use cozy_chess::Board;

#[test]
fn knight_center_better_than_rim() {
    use piebot::search::eval::eval_cp;
    // White: Kh1, Nd4; Black: Ka8. White to move.
    let center = Board::from_fen("k7/8/8/8/3N4/8/8/7K w - - 0 1", false).unwrap();
    let rim = Board::from_fen("k7/8/8/8/8/8/8/N6K w - - 0 1", false).unwrap();
    let c = eval_cp(&center);
    let r = eval_cp(&rim);
    assert!(c > r, "center eval {c} should be greater than rim {r}");
}

#[test]
fn pawn_advanced_better_than_back() {
    use piebot::search::eval::eval_cp;
    // White pawn on e4 vs e2; kings only otherwise.
    let advanced = Board::from_fen("k7/8/8/8/4P3/8/8/7K w - - 0 1", false).unwrap();
    let back = Board::from_fen("k7/8/8/8/8/8/4P3/7K w - - 0 1", false).unwrap();
    let a = eval_cp(&advanced);
    let b = eval_cp(&back);
    assert!(a > b, "advanced pawn eval {a} should exceed back pawn {b}");
}
