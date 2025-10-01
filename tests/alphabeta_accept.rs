use cozy_chess::Board;

#[test]
fn alphabeta_finds_simple_mate_in_one() {
    let board = Board::from_fen("k7/8/6Q1/8/8/8/8/K7 w - - 0 1", false).expect("valid FEN");
    let mut searcher = piebot::search::alphabeta::Searcher::default();
    searcher.set_tt_capacity_mb(16);

    let mut params = piebot::search::alphabeta::SearchParams::default();
    params.depth = 3;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.use_killers = true;
    params.use_nullmove = true;
    params.use_aspiration = true;
    params.use_lmr = true;
    params.aspiration_window_cp = 35;
    params.threads = 1;
    params.deterministic = true;

    let result = searcher.search_with_params(&board, params);
    assert_eq!(result.bestmove.as_deref(), Some("qe8"));
    assert!(result.score_cp > 24_000, "expected mate score, got {}", result.score_cp);
}

