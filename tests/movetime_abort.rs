use cozy_chess::Board;

// When an iteration is aborted (via node limit), the experimental engine
// should not record an Exact TT entry at the root for the aborted depth.
// This test first estimates nodes needed for depth 1, then starts a depth-2
// search with a very small max_nodes budget to ensure the depth-2 iteration
// aborts. We then assert the TT does not contain an Exact bound at the
// root for depth >= 2.
#[test]
fn experimental_abort_does_not_store_exact_tt_for_aborted_depth() {
    // A moderately branching midgame FEN to ensure non-trivial node counts.
    let fen = "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2P2N2/PP1N1PPP/R1BQ1RK1 w - - 0 8";
    let board = Board::from_fen(fen, false).expect("valid FEN");

    // First, complete a depth-1 search to get a baseline node count.
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let mut p1 = piebot::search::alphabeta_temp::SearchParams::default();
    p1.depth = 1;
    p1.use_tt = true;
    p1.max_nodes = None;
    p1.movetime = None;
    p1.order_captures = true;
    p1.use_history = true;
    p1.use_killers = true;
    p1.use_lmr = true;
    p1.use_nullmove = true;
    p1.threads = 1;
    p1.deterministic = true;
    let r1 = s.search_with_params(&board, p1);

    // Now run depth-2 but cap nodes just above the depth-1 consumption to
    // ensure depth-1 completes and depth-2 iteration aborts early.
    let mut p2 = piebot::search::alphabeta_temp::SearchParams::default();
    p2.depth = 2;
    p2.use_tt = true;
    // Budget: comfortably complete depth-1, begin depth-2, but likely abort before finishing.
    p2.max_nodes = Some(r1.nodes.saturating_mul(3));
    p2.movetime = None;
    p2.order_captures = true;
    p2.use_history = true;
    p2.use_killers = true;
    p2.use_lmr = true;
    p2.use_nullmove = true;
    p2.threads = 1;
    p2.deterministic = true;
    let _ = s.search_with_params(&board, p2);

    // Probe TT: do not allow an Exact bound at depth >= 2 when the iteration
    // has been aborted.
    if let Some((depth, bound)) = s.tt_probe(&board) {
        if depth >= 2 {
            assert!(
                !matches!(bound, piebot::search::tt::Bound::Exact),
                "aborted iteration stored Exact TT at depth {}",
                depth
            );
        }
    }
}
