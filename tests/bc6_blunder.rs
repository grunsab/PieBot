use cozy_chess::Board;

// From user: baseline (White) blundered with Bc6 in this position.
// Ensure baseline avoids hanging bishop move e8c6.
// FEN: 4Brk1/1pp2pp1/8/p3qbbp/Q7/PPN1B3/2n1KPPP/R6R w - - 0 19
#[test]
fn baseline_avoids_bc6_hanging_bishop() {
    let fen = "4Brk1/1pp2pp1/8/p3qbbp/Q7/PPN1B3/2n1KPPP/R6R w - - 0 19";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove { assert_ne!(bm.as_str(), "e8c6", "baseline chose hanging bishop move Bc6"); }
}

