use cozy_chess::Board;
use std::time::{Duration, Instant};

#[test]
fn movetime_returns_quickly_with_move() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let b = Board::default();
    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 10; // large depth
    params.use_tt = true;
    params.movetime = Some(Duration::from_millis(10));
    let t0 = Instant::now();
    let res = searcher.search_with_params(&b, params);
    let elapsed = t0.elapsed();
    assert!(res.bestmove.is_some(), "no bestmove under movetime");
    assert!(
        elapsed < Duration::from_millis(300),
        "search exceeded time: {:?}",
        elapsed
    );
}
