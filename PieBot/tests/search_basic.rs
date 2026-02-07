use cozy_chess::Board;

#[test]
fn eval_material_startpos_is_zeroish() {
    use piebot::search::eval::material_eval_cp;
    let b = Board::default();
    // Startpos material should be equal; score ~ 0 from white perspective.
    let cp = material_eval_cp(&b);
    assert!(cp.abs() < 5, "startpos material not near zero: {cp}");
}

#[test]
fn eval_material_known_advantage() {
    use piebot::search::eval::material_eval_cp;
    // White: Kh1, Qe2; Black: Ka8, Qd2 (legal). Material equal.
    let fen = "k7/8/8/8/8/8/3qQ3/7K w - - 0 1";
    let b = Board::from_fen(fen, false).expect("valid fen");
    let cp = material_eval_cp(&b);
    // Material equal pre-move => near 0
    assert!(cp.abs() < 5, "material should be equal: {cp}");
}

#[test]
fn search_returns_legal_move_startpos() {
    use piebot::search::alphabeta::Searcher;
    let b = Board::default();
    let mut searcher = Searcher::default();
    let res = searcher.search_depth(&b, 1);
    assert!(res.bestmove.is_some(), "no move found at depth 1");
}

#[test]
fn search_prefers_winning_queen_capture() {
    use piebot::search::alphabeta::Searcher;
    // Position where Qe2xd2 wins a queen (and is legal)
    let fen = "k7/8/8/8/8/8/3qQ3/7K w - - 0 1";
    let b = Board::from_fen(fen, false).expect("valid fen");
    let mut searcher = Searcher::default();
    let res = searcher.search_depth(&b, 1);
    let bm = res.bestmove.expect("expected a best move");
    assert_eq!(bm, "e2d2", "expected Qe2xd2 as best move, got {bm}");
}
