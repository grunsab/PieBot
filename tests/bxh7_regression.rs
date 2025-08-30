use cozy_chess::Board;

// Baseline regression from user: in this FEN, White should NOT play Bxh7,
// because ...Nxf3+ wins the white queen on f3. Ensure baseline avoids Bxh7.
// FEN: r1bqkb1r/1pp1pppp/5n2/3Pn3/p7/P1NB1Q2/1PP2PPP/R1B1K2R w KQkq - 1 9
#[test]
fn baseline_avoids_bxh7_hanging_queen() {
    let fen = "r1bqkb1r/1pp1pppp/5n2/3Pn3/p7/P1NB1Q2/1PP2PPP/R1B1K2R w KQkq - 1 9";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    // Depth-2 is sufficient to see ...Nxf3+ wins the queen
    let res = s.search_depth(&board, 2);
    if let Some(bm) = res.bestmove {
        assert_ne!(bm.as_str(), "d3h7", "baseline chose Bxh7, hanging the queen to ...Nxf3+ at depth 2");
    }
}

