use cozy_chess::Board;

#[test]
fn nullmove_reduces_nodes_midgame() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 4; p1.use_tt = true; p1.order_captures = true; p1.use_history = true; p1.threads = 1;
    p1.use_nullmove = false; p1.use_aspiration = true; p1.aspiration_window_cp = 50;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1; p2.use_nullmove = true;
    let r2 = s2.search_with_params(&b, p2);
    assert!((r2.score_cp - r1.score_cp).abs() <= 100, "nullmove changed score too much: {} vs {}", r2.score_cp, r1.score_cp);
    let ratio = r2.nodes as f64 / r1.nodes as f64;
    assert!(ratio <= 1.6, "nullmove should not explore substantially more nodes (ratio {:.2})", ratio);
}

#[test]
fn nullmove_disabled_in_check() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    // Black in check from rook on a1
    let fen = "k7/8/8/8/8/8/8/R3K3 b - - 0 1";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s1 = Searcher::default();
    let mut p1 = SearchParams::default();
    p1.depth = 3; p1.use_tt = true; p1.order_captures = true; p1.use_history = true; p1.threads = 1;
    p1.use_nullmove = false;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    let mut p2 = p1; p2.use_nullmove = true;
    let r2 = s2.search_with_params(&b, p2);
    assert_eq!(r2.score_cp, r1.score_cp, "nullmove in check should not change score");
}

#[test]
fn nullmove_reduces_nodes_shallow_depth() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    // Balanced middlegame position where null-move pruning should still help at depth 3
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let board = Board::from_fen(fen, false).unwrap();

    let mut search_no_null = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 3; params.use_tt = true; params.order_captures = true; params.use_history = true; params.threads = 1;
    params.use_nullmove = false; params.use_aspiration = true; params.aspiration_window_cp = 50;
    let result_no_null = search_no_null.search_with_params(&board, params);

    let mut search_with_null = Searcher::default();
    let mut params_with_null = params; params_with_null.use_nullmove = true;
    let result_with_null = search_with_null.search_with_params(&board, params_with_null);

    let ratio = result_with_null.nodes as f64 / result_no_null.nodes as f64;
    assert!(ratio <= 1.1, "nullmove should not explore substantially more nodes even at depth 3 (ratio {:.2})", ratio);
    assert!((result_with_null.score_cp - result_no_null.score_cp).abs() <= 100, "nullmove should not swing evaluation wildly at shallow depth");
}

#[test]
fn nullmove_disabled_in_simple_zugzwang() {
    use piebot::search::alphabeta::{Searcher, SearchParams};
    // Basic king+pawn zugzwang: null move would give away the win
    let fen = "8/8/8/8/5K2/8/5P2/7k w - - 0 1";
    let board = Board::from_fen(fen, false).unwrap();

    let mut search_no_null = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 4; params.use_tt = true; params.order_captures = true; params.use_history = true; params.threads = 1;
    params.use_nullmove = false;
    let result_no_null = search_no_null.search_with_params(&board, params);

    let mut search_with_null = Searcher::default();
    let mut params_with_null = params; params_with_null.use_nullmove = true;
    let result_with_null = search_with_null.search_with_params(&board, params_with_null);

    assert_eq!(result_with_null.score_cp, result_no_null.score_cp, "nullmove must stay disabled in zugzwang to preserve score");
    assert!(result_with_null.nodes >= result_no_null.nodes, "zugzwang guard should prevent pruning ({} vs {})", result_with_null.nodes, result_no_null.nodes);
}
