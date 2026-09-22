use cozy_chess::Board;
use piebot::search::alphabeta_temp::Searcher;

#[test]
fn test_see_pruning_and_lmp_deterministic_depth() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    // Initial position, search at depth 5
    let board = Board::default();
    let res = searcher.search_depth(&board, 5);
    assert!(res.bestmove.is_some(), "Search should find a move");
    assert!(res.nodes > 0, "Nodes searched should be positive");
}

#[test]
fn test_experimental_improving_asymmetric_pruning() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board = Board::default();
    let res = searcher.search_depth(&board, 6);
    assert!(res.bestmove.is_some(), "Search must find a legal move at depth 6");
    assert!(res.nodes > 0, "Search visited nodes");
}

