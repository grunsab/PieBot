use cozy_chess::Board;

#[test]
fn history_table_reduces_nodes() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    // Tactical-ish position where history heuristics should help ordering
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();

    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 4;
    p1.use_tt = true;
    p1.order_captures = true;
    p1.use_history = false;
    p1.threads = 1;
    p1.use_aspiration = false;
    p1.use_lmr = false;
    p1.use_killers = false;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.use_history = true;
    let r2 = s2.search_with_params(&b, p2);

    assert_eq!(r2.score_cp, r1.score_cp, "history should not change score");
    assert!(
        r2.nodes <= r1.nodes,
        "history should reduce nodes: {} vs {}",
        r2.nodes,
        r1.nodes
    );
}
