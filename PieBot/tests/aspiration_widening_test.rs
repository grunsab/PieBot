use cozy_chess::Board;
use piebot::search::alphabeta_temp::{SearchParams, Searcher};
use std::time::Instant;

#[test]
fn test_single_legal_move_exits_early() {
    // White king in check, only 1 legal evasion move:
    // White: Kh1, pawns blocked or none. Black: Rh2 check.
    // e.g., 8/8/8/8/8/8/7r/6K1 w - - 0 1 -> White only has Kg1-f1 or Kh1 (if legal)
    // Construct single-legal-move position:
    // 7k/8/8/8/8/8/6r1/7K w - - 0 1
    // King on h1, black rook on g2.
    // White's only legal moves: Kh1xg2! (1 legal move)
    let fen = "7k/8/8/8/8/8/6r1/7K w - - 0 1";
    let board = Board::from_fen(fen, false).expect("valid fen");

    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 10;
    params.use_tt = true;
    params.order_captures = true;

    let t0 = Instant::now();
    let res = searcher.search_with_params(&board, params);
    let elapsed = t0.elapsed();

    assert_eq!(res.bestmove.as_deref(), Some("h1g2"));
    // With early exit on single legal move, depth stops early at d=1 or d=2 rather than grinding to depth 10
    assert!(res.depth <= 2, "Expected early exit at d<=2, got depth {}", res.depth);
    assert!(elapsed.as_millis() < 50, "Should return almost instantaneously");
}

#[test]
fn test_aspiration_window_resolves_fluctuation() {
    let board = Board::default();
    let mut searcher = Searcher::default();
    searcher.set_use_aspiration(true);
    let (bestmove, score, nodes) = searcher.search_movetime(&board, 50, 6);
    assert!(bestmove.is_some());
    assert!(nodes > 0);
    assert!(score.abs() < 100);
}

#[test]
fn test_aspiration_window_movetime_fluctuating_tactical() {
    let fens = [
        "r1bqk2r/pppp1ppp/2n5/4p3/1bB1n3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "r2q1rk1/ppp2ppp/2n1bn2/3pp3/1bPP4/2N1PN2/PP1BBPPP/R2QK2R w KQ - 4 8",
    ];
    for fen in fens {
        let board = Board::from_fen(fen, false).expect("valid fen");
        let mut searcher = Searcher::default();
        searcher.set_use_aspiration(true);
        searcher.set_order_captures(true);
        let (bestmove, score, nodes) = searcher.search_movetime(&board, 50, 6);
        assert!(bestmove.is_some());
        assert!(nodes > 0);
        assert!(score.abs() < 1500);
    }
}
