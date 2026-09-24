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

#[test]
fn test_probcut_search_depth() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board: Board = "r1bqk2r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5".parse().unwrap();
    let res = searcher.search_depth(&board, 7);
    assert!(res.bestmove.is_some(), "ProbCut search finds a valid move");
    assert!(res.nodes > 0, "Nodes searched");
}

#[test]
fn test_internal_iterative_reduction() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board: Board = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4".parse().unwrap();
    let res = searcher.search_depth(&board, 6);
    assert!(res.bestmove.is_some(), "Search with IIR finds a valid move");
    assert!(res.nodes > 0, "Nodes searched");
}

#[test]
fn test_see_guarded_check_extension_temp() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    // Position with a checking move
    let board: Board = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3".parse().unwrap();
    let res = searcher.search_depth(&board, 5);
    assert!(res.bestmove.is_some());
    assert!(res.nodes > 0);
}




