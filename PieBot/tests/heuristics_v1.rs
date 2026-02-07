use cozy_chess::Board;

#[test]
fn aspiration_retains_score_and_can_reduce_nodes() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 4;
    p1.use_tt = true;
    p1.order_captures = true;
    p1.use_history = true;
    p1.threads = 1;
    p1.use_aspiration = false;
    p1.use_lmr = false;
    p1.use_killers = false;
    p1.aspiration_window_cp = 50;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.use_aspiration = true;
    let r2 = s2.search_with_params(&b, p2);
    assert_eq!(r2.score_cp, r1.score_cp, "aspiration changed score");
    assert!(
        r2.nodes <= r1.nodes,
        "aspiration did not reduce nodes: {} vs {}",
        r2.nodes,
        r1.nodes
    );
}

#[test]
fn killers_lmr_reduce_nodes() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 4;
    p1.use_tt = true;
    p1.order_captures = true;
    p1.use_history = true;
    p1.threads = 1;
    p1.use_aspiration = true;
    p1.aspiration_window_cp = 50;
    p1.use_lmr = false;
    p1.use_killers = false;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.use_lmr = true;
    p2.use_killers = true;
    let r2 = s2.search_with_params(&b, p2);
    assert!(
        (r2.score_cp - r1.score_cp).abs() <= 100,
        "heuristics changed score too much: {} vs {}",
        r2.score_cp,
        r1.score_cp
    );
    assert!(
        r2.nodes <= r1.nodes,
        "heuristics did not reduce nodes: {} vs {}",
        r2.nodes,
        r1.nodes
    );
}
