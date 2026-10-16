use piebot::search::alphabeta_temp::Searcher;

#[test]
fn null_move_reduction_grows_with_depth_and_eval_margin() {
    let searcher = Searcher::default();
    let mut prev = 0;
    for depth in 3..=20 {
        let r = searcher.null_move_reduction(depth, 0, 0);
        assert!(r >= prev, "reduction must not shrink with depth");
        assert!(r <= depth, "reduction cannot exceed the depth");
        prev = r;
    }
    assert!(searcher.null_move_reduction(12, 600, 0) > searcher.null_move_reduction(12, 0, 0));
    // The eval-margin term is capped.
    assert_eq!(
        searcher.null_move_reduction(30, 5_000, 0),
        searcher.null_move_reduction(30, 50_000, 0)
    );
}
