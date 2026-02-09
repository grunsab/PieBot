#[test]
fn threads_param_propagates() {
    use cozy_chess::Board;
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let b = Board::default();
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = 2;
    p.use_tt = false;
    p.threads = 4;
    let _ = s.search_with_params(&b, p);
    assert_eq!(
        s.get_threads(),
        4,
        "threads param did not propagate to searcher"
    );
}

#[test]
fn multi_thread_search_produces_valid_result_under_time() {
    use cozy_chess::Board;
    use piebot::search::alphabeta::{SearchParams, Searcher};
    use std::time::Duration;
    let b = Board::default();
    // 1 thread (install a 1-thread pool)
    let r1 = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s1 = Searcher::default();
            let mut p1 = SearchParams::default();
            p1.movetime = Some(Duration::from_millis(150));
            p1.use_tt = true;
            p1.order_captures = true;
            p1.use_history = true;
            p1.threads = 1;
            s1.search_with_params(&b, p1)
        })
    };
    // 4 threads
    let r4 = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s4 = Searcher::default();
            let mut p4 = SearchParams::default();
            p4.movetime = Some(Duration::from_millis(150));
            p4.use_tt = true;
            p4.order_captures = true;
            p4.use_history = true;
            p4.threads = 4;
            s4.search_with_params(&b, p4)
        })
    };
    assert!(r1.nodes > 0, "single-thread search should visit nodes");
    assert!(r4.nodes > 0, "multi-thread search should visit nodes");
    assert!(
        r4.bestmove.is_some(),
        "multi-thread search should return a best move"
    );
}
