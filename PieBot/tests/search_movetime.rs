use cozy_chess::Board;

#[test]
fn movetime_only_searches_nodes() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    use std::time::Duration;
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.movetime = Some(Duration::from_millis(10));
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = 1;
    // Leave depth at default (0); engine should still search until deadline
    let r = s.search_with_params(&b, p);
    assert!(
        r.nodes > 0,
        "expected nodes>0 when depth=0 and movetime is set"
    );
}
