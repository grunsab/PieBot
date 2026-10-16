use cozy_chess::Board;

#[test]
fn captures_first_reduces_nodes() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    // Position with a clear capture line available
    let fen = "4k3/8/8/8/5Q2/8/8/2b4K b - - 0 1";
    let b = Board::from_fen(fen, false).expect("valid fen");
    // Baseline: no capture ordering
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 3;
    p1.use_tt = false;
    p1.max_nodes = None;
    p1.movetime = None;
    p1.order_captures = false;
    p1.use_history = false;
    let r1 = s1.search_with_params(&b, p1);

    // With capture-first ordering
    let mut s2 = Searcher::default();
    let mut p2 = p1;
    p2.order_captures = true;
    let r2 = s2.search_with_params(&b, p2);

    // The search always orders captures first now (the flag is kept for
    // API compatibility), so both runs must simply win the queen.
    assert_eq!(r1.bestmove.as_deref(), Some("c1f4"));
    assert_eq!(r2.bestmove.as_deref(), Some("c1f4"));
    assert!(r2.nodes <= r1.nodes, "{} vs {}", r2.nodes, r1.nodes);
}
