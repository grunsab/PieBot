use cozy_chess::Board;

// From user: Baseline (White) should capture the queen with Nxd4,
// but played Rf6+. Ensure baseline picks Nxd4.
// FEN: 2b2k1r/1pp1b2p/7R/5P2/r2q4/P4NQ1/2P5/R4K2 w - - 0 23
#[test]
fn baseline_prefers_nxd4_over_check() {
    let fen = "2b2k1r/1pp1b2p/7R/5P2/r2q4/P4NQ1/2P5/R4K2 w - - 0 23";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    let bm = res.bestmove.expect("expected best move");
    assert_eq!(bm.as_str(), "f3d4", "expected Nxd4 winning the queen; got {}", bm);
}

#[test]
fn baseline_avoids_hanging_check_rf6() {
    // Same FEN: the quiet checking move Rf6+ hangs to ...Bxf6. Ensure baseline does not choose it.
    let fen = "2b2k1r/1pp1b2p/7R/5P2/r2q4/P4NQ1/2P5/R4K2 w - - 0 23";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "h6f6", "baseline should avoid hanging check Rf6+; got {}", bm);
    }
}
