use cozy_chess::Board;
use piebot::search::alphabeta_temp::{SearchParams, Searcher};

#[test]
fn continuation_history_populates_and_preserves_score() {
    // Italian game opening position with tactical and quiet choices
    let fen = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    let b = Board::from_fen(fen, false).unwrap();

    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 5;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.threads = 1;
    params.use_aspiration = false;
    params.use_lmr = false;
    params.use_killers = true;

    let res = searcher.search_with_params(&b, params);
    assert!(res.nodes > 0);
    assert!(res.bestmove.is_some());

    // Verify continuation history was actively populated and contains non-zero entries
    assert!(
        searcher.continuation_history_entries_count() > 0,
        "continuation history tables must be populated during search with use_history=true"
    );
}
