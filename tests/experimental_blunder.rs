use cozy_chess::Board;

// Regression: experimental search (alphabeta_temp) should not blunder Qxh2 in this FEN.
// FEN: rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7
#[test]
fn experimental_avoids_losing_queen_sac_on_h2() {
    let fen = "rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Depth-2 should be sufficient to see Rxh2 wins the queen.
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "d6h2", "experimental chose losing queen sac Qxh2 at depth 2");
    }
}

// Sanity check: SEE detects that Qxh2 is a losing capture for Black.
#[test]
fn see_flags_qxh2_as_losing() {
    let fen = "rnb1kbnr/4pppp/3q4/1P6/p7/P3P3/3P1PPP/RNBQKBNR b KQkq - 0 7";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Locate Qd6h2
    let mut qxh2 = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == "d6h2" { qxh2 = Some(m); break; } }
        qxh2.is_some()
    });
    let m = qxh2.expect("Qxh2 should be legal");
    let see = piebot::search::see::see_gain_cp(&board, m).expect("SEE gain must exist");
    assert!(see < -300, "expected Qxh2 to be a large losing capture (SEE), got {}", see);
}

// Baseline regression: avoid hanging quiet move Nc3 which loses to ...Qxc3.
#[test]
fn baseline_avoids_hanging_nc3() {
    let fen = "rn2kbnr/2qb1ppp/8/1P1Pp3/p7/P4N2/4PPPP/RNBQKB1R w KQkq - 1 10";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "b1c3", "baseline chose hanging quiet move Nc3 at depth 2");
    }
}

// Experimental regression: avoid queen sac Qxd3 in this FEN.
// FEN: rn2k1nr/3b1ppp/3b4/1P1Pp3/p3P3/P2B1N2/3Q1PPP/1qBK3R b kq - 4 14
#[test]
fn experimental_avoids_losing_qxd3() {
    let fen = "rn2k1nr/3b1ppp/3b4/1P1Pp3/p3P3/P2B1N2/3Q1PPP/1qBK3R b kq - 4 14";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "b1d3", "experimental chose losing queen capture Qxd3 at depth 2");
    }
}

// Experimental regression: avoid quiet checking queen sac Qf3+
// FEN: 6k1/8/rb6/8/5K2/2p5/2q5/7q b - - 3 66
#[test]
fn experimental_avoids_qf3_check_sac() {
    let fen = "6k1/8/rb6/8/5K2/2p5/2q5/7q b - - 3 66";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "h1f3", "experimental chose losing queen checking sac Qf3+ at depth 2");
    }
}

// Baseline regression: avoid queen sac Qc3 (hanging to ...Bxc3)
// FEN: 3k3r/5ppp/4pn2/r7/qb1PP3/P2Q4/5PPP/R1B1KBR1 w Q - 1 18
#[test]
fn baseline_avoids_losing_qc3() {
    let fen = "3k3r/5ppp/4pn2/r7/qb1PP3/P2Q4/5PPP/R1B1KBR1 w Q - 1 18";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "d3c3", "baseline chose losing queen quiet Qc3 at depth 2");
    }
}

// Baseline should prefer winning capture gxf3 over a slow move like Bb6
// FEN: k7/8/3P4/8/8/P3Bn1P/2P2PP1/3R1K2 w - - 5 41
#[test]
fn baseline_prefers_gxf3_over_bb6() {
    let fen = "k7/8/3P4/8/8/P3Bn1P/2P2PP1/3R1K2 w - - 5 41";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    // Slightly deeper to let SEE and ordering distinguish
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "e3b6", "baseline chose slow Bb6 instead of strong gxf3");
    }
}

// Baseline: mate in 1 must be chosen (Rb7#)
// FEN: R5NB/5R2/8/1k6/4B3/8/8/2R3K1 w - - 3 61
#[test]
fn baseline_plays_mate_in_one_rb7() {
    let fen = "R5NB/5R2/8/1k6/4B3/8/8/2R3K1 w - - 3 61";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 1);
    let bm = res.bestmove.expect("expected a best move");
    assert_eq!(bm.as_str(), "f7b7", "expected Rb7# (f7b7 in UCI)");
}

// Baseline: avoid quiet queen hang Qf3 in a losing position
// FEN: 2bqk2r/2p2pp1/r6p/4N3/3nN3/8/1bP2P1P/3Q1RK1 w k - 0 16
#[test]
fn baseline_avoids_qf3_hanging() {
    let fen = "2bqk2r/2p2pp1/r6p/4N3/3nN3/8/1bP2P1P/3Q1RK1 w k - 0 16";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove { assert_ne!(bm.as_str(), "d1f3", "baseline chose hanging queen move Qf3"); }
}
