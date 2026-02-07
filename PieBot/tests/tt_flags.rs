use cozy_chess::Board;

#[test]
fn tt_exact_after_search() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    use piebot::search::tt::Bound;
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = 3;
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    s.search_with_params(&b, p);
    let e = s.tt_probe(&b).expect("tt entry missing");
    assert_eq!(
        e.1,
        Bound::Exact,
        "expected exact bound after full-window search"
    );
    assert!(e.0 >= 3, "expected stored depth >= 3, got {}", e.0);
}

#[test]
fn tt_depth_preferred_not_replaced() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = 3;
    p.use_tt = true;
    s.search_with_params(&b, p);
    let d1 = s.tt_probe(&b).unwrap().0;
    let mut p2 = SearchParams::default();
    p2.depth = 1;
    p2.use_tt = true;
    s.search_with_params(&b, p2);
    let d2 = s.tt_probe(&b).unwrap().0;
    assert!(
        d2 >= d1,
        "shallower search should not lower stored TT depth"
    );
}
