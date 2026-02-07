use cozy_chess::Board;

#[test]
fn qsearch_improves_tactical_position() {
    use piebot::search::alphabeta::Searcher;
    let fen = "4k3/8/8/8/5Q2/8/8/2b4K b - - 0 1"; // hanging queen vs bishop
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let stand = piebot::search::eval::eval_cp(&b);
    let qs = s.qsearch_eval_cp(&b);
    // For black to move, eval is from side-to-move perspective; capturing queen should improve score over stand pat.
    assert!(
        qs > stand,
        "qsearch should improve eval: qs {qs} vs stand {stand}"
    );
}

#[test]
fn qsearch_equals_standpat_without_captures() {
    use piebot::search::alphabeta::Searcher;
    let fen = "k7/8/8/8/8/8/8/7K w - - 0 1"; // bare kings, no captures
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let stand = piebot::search::eval::eval_cp(&b);
    let qs = s.qsearch_eval_cp(&b);
    assert_eq!(qs, stand, "qsearch should equal stand pat without captures");
}
