use cozy_chess::Board;

#[test]
fn root_parallel_bestmove_equals_single_thread() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    // A midgame-ish FEN with many legal moves
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();

    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 3;
    p1.use_tt = true;
    p1.order_captures = true;
    p1.use_history = true;
    p1.threads = 1;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.threads = 4;
    let r2 = s2.search_with_params(&b, p2);

    assert_eq!(
        r2.score_cp, r1.score_cp,
        "score differs between single and multi-thread at fixed depth"
    );
}

#[test]
fn root_parallel_returns_move_with_movetime() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    use std::time::Duration;
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = 10;
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = 4;
    p.movetime = Some(Duration::from_millis(5));
    let r = s.search_with_params(&b, p);
    assert!(
        r.bestmove.is_some(),
        "no move returned under movetime with threads"
    );
}
