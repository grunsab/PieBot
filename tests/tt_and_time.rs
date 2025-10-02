use cozy_chess::Board;

#[test]
fn tt_reduces_nodes_on_second_run() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let b = Board::default();
    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 4;
    params.use_tt = true;

    // First run to populate TT
    let first = searcher.search_with_params(&b, params);
    // Second run should reuse TT and visit fewer nodes
    let second = searcher.search_with_params(&b, params);
    assert!(
        second.nodes < first.nodes,
        "TT did not reduce nodes: {} vs {}",
        second.nodes,
        first.nodes
    );
}

#[test]
fn node_limit_stops_early_and_returns_move() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let b = Board::default();
    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 8;
    params.use_tt = true;
    params.max_nodes = Some(5_000);
    let res = searcher.search_with_params(&b, params);
    assert!(res.bestmove.is_some(), "no bestmove under node limit");
    assert!(res.nodes <= 6_000, "node limit exceeded: {}", res.nodes);
}
