use cozy_chess::Board;
use piebot::search::alphabeta::{SearchParams, Searcher};

#[test]
fn test_lmp_improving_tuning_solves_and_runs() {
    let board = Board::default();
    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 6;
    params.use_nullmove = true;
    params.use_history = true;
    let res = searcher.search_with_params(&board, params);
    assert!(res.bestmove.is_some());
    assert!(res.nodes > 0);
}
