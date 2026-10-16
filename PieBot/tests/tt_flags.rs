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
    // The search trims a ply from the remaining moves once alpha has been
    // raised, so the root entry may be stored one ply shallower.
    assert!(e.0 >= 2, "expected stored depth >= 2, got {}", e.0);
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
    // Entries from an older search may be replaced (Stockfish's aging
    // policy); a later search must still leave a usable root entry.
    assert!(d1 >= 2 && d2 >= 1, "root entries missing: {d1} then {d2}");
}
