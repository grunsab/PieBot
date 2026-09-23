use piebot::search::alphabeta_temp::Searcher;

#[test]
fn test_dynamic_null_move_reduction() {
    let searcher = Searcher::default();

    // Shallow depth (<= 4): R = 1
    assert_eq!(searcher.null_move_reduction(3, 0, 0), 1);
    assert_eq!(searcher.null_move_reduction(4, 0, 0), 1);

    // Medium depth (5): R = 2
    assert_eq!(searcher.null_move_reduction(5, 0, 0), 2);

    // Depth 6+: R = 3
    assert_eq!(searcher.null_move_reduction(6, 0, 0), 3);
    assert_eq!(searcher.null_move_reduction(8, 0, 0), 3);

    // Depth 10+: R = 4
    assert_eq!(searcher.null_move_reduction(10, 0, 0), 4);

    // Eval margin bonus (+1 when eval - beta > 250)
    assert_eq!(searcher.null_move_reduction(8, 300, 0), 4);
    assert_eq!(searcher.null_move_reduction(8, 600, 0), 5);
}
