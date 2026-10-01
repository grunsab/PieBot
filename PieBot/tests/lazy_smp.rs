use cozy_chess::Board;
use piebot::search::alphabeta_temp::{SearchParams, Searcher};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

fn params(threads: usize) -> SearchParams {
    SearchParams {
        depth: 7,
        threads,
        use_tt: true,
        order_captures: true,
        use_history: true,
        use_killers: true,
        use_nullmove: true,
        use_lmr: true,
        use_aspiration: true,
        aspiration_window_cp: 35,
        ..SearchParams::default()
    }
}

#[test]
fn deterministic_search_is_identical_when_more_threads_are_requested() {
    let board = Board::default();
    let mut one = Searcher::default();
    let mut many = Searcher::default();
    let mut p = params(1);
    p.deterministic = true;
    let reference = one.search_with_params(&board, p);
    p.threads = 4;
    let result = many.search_with_params(&board, p);
    assert_eq!(
        (result.bestmove, result.score_cp, result.nodes, result.depth),
        (
            reference.bestmove,
            reference.score_cp,
            reference.nodes,
            reference.depth
        )
    );
    assert_eq!(many.get_threads(), 4);
}

#[test]
fn node_limit_is_global_with_multiple_threads_and_aspiration() {
    let mut searcher = Searcher::default();
    let mut p = params(4);
    p.depth = 99;
    p.max_nodes = Some(5000);
    let result = searcher.search_with_params(&Board::default(), p);
    assert!(
        result.nodes <= 5000,
        "global budget exceeded: {}",
        result.nodes
    );
    assert!(result.bestmove.is_some());
}

#[test]
fn stopping_parallel_search_joins_helpers_and_allows_another_search() {
    let mut searcher = Searcher::default();
    let stop = Arc::new(AtomicBool::new(false));
    searcher.set_stop_flag(Some(stop.clone()));
    let begin = Instant::now();
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(40));
            stop.store(true, Ordering::Relaxed);
        });
        let mut p = params(4);
        p.depth = 99;
        searcher.search_with_params(&Board::default(), p)
    });
    assert!(begin.elapsed() < Duration::from_secs(3));
    assert!(result.bestmove.is_some());
    searcher.clear_stop_flag();
    let mut p = params(2);
    p.depth = 5;
    let next = searcher.search_with_params(&Board::default(), p);
    assert_eq!(next.depth, 5);
    assert!(next.nodes > 0);
}
