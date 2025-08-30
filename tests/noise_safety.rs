use cozy_chess::Board;
use rand::SeedableRng;
use rand::rngs::SmallRng;

// Helper to compute a noisy choice with SEE filtering using the library functions
fn choose_filtered(board: &Board, topk: usize, seed: u64, see_thresh: i32) -> Option<cozy_chess::Move> {
    let mut s = piebot::search::alphabeta::Searcher::default();
    // Enable ordering heuristics so top-K is meaningful
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_use_aspiration(true);
    let order = s.debug_order_for_parent(board, usize::MAX);
    let mut rng = SmallRng::seed_from_u64(seed);
    piebot::search::noise::choose_noisy_from_order_filtered(board, &order, topk, &mut rng, see_thresh)
}

#[test]
fn noisy_topk_filters_losing_queen_capture() {
    // Position: White to move. White queen on b1 can capture a2, but ...Rxa2 loses the queen badly.
    // FEN: k7/8/8/8/8/8/p6P/rQ2K3 w - - 0 1
    let fen = "k7/8/8/8/8/8/p6P/rQ2K3 w - - 0 1";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Try multiple seeds to simulate noisy sampling; ensure the losing capture is never returned
    for t in 0..64u64 {
        let m = choose_filtered(&board, 2, 0xABCDEF00u64 ^ t, -150).expect("some move");
        let uci = format!("{}", m);
        assert_ne!(uci, "b1a2", "filtered should avoid Qxa2 losing capture (seed={})", t);
    }
}

#[test]
fn noisy_topk_filters_hanging_quiet_move() {
    // From user: baseline blunder under noise. Nc3 hangs to ...Qxc3.
    let fen = "rn2kbnr/2qb1ppp/8/1P1Pp3/p7/P4N2/4PPPP/RNBQKB1R w KQkq - 1 10";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Try multiple seeds to simulate noisy sampling; ensure b1c3 is never chosen when in top-K
    for t in 0..64u64 {
        let m = choose_filtered(&board, 4, 0xBAD5EEDu64 ^ t, -150).expect("some move");
        let uci = format!("{}", m);
        assert_ne!(uci, "b1c3", "filtered should avoid hanging quiet move Nc3 (seed={})", t);
    }
}

#[test]
fn noisy_topk_filters_bxh7_exposing_queen() {
    // From user: White to move. Bxh7 allows ...Nxf3+ winning the white queen on f3.
    // FEN: r1bqkb1r/1pp1pppp/5n2/3Pn3/p7/P1NB1Q2/1PP2PPP/R1B1K2R w KQkq - 1 9
    let fen = "r1bqkb1r/1pp1pppp/5n2/3Pn3/p7/P1NB1Q2/1PP2PPP/R1B1K2R w KQkq - 1 9";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Compute the engine's root order, then find Bxh7 (d3h7) index.
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_use_aspiration(true);
    let order = s.debug_order_for_parent(&board, usize::MAX);
    let idx = order.iter().position(|&m| format!("{}", m) == "d3h7").expect("Bxh7 should be legal");
    // Limit top-K to include Bxh7 and test multiple seeds; the filter should never return Bxh7
    let topk = (idx + 1).min(order.len());
    for t in 0..64u64 {
        let mut rng = SmallRng::seed_from_u64(0xFEED_F00Du64 ^ t);
        let mv = piebot::search::noise::choose_noisy_from_order_filtered(&board, &order, topk, &mut rng, -150).expect("some move");
        let uci = format!("{}", mv);
        assert_ne!(uci, "d3h7", "filtered should avoid Bxh7 which exposes the queen (seed={})", t);
    }
}
