use cozy_chess::Board;

#[test]
fn split_points_score_equal_multi_vs_single() {
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
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.threads = 4;
    let r2 = s2.search_with_params(&b, p2);

    assert_eq!(
        r2.score_cp, r1.score_cp,
        "split points multi-thread score differs from single-thread"
    );
}
