#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn see_ordering_improves_rank_of_best_capture() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    use piebot::search::see_pleco::see_gain_cp;

    // A position with several captures; not pathological, just enough to compare
    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = PBoard::from_fen(fen).expect("valid FEN");

    // Identify best SEE capture
    let moves: Vec<_> = b.generate_moves().iter().copied().filter(|m| m.is_capture()).collect();
    assert!(!moves.is_empty());
    let mut best_cap = moves[0];
    let mut best_gain = i32::MIN;
    for m in &moves {
        if let Some(g) = see_gain_cp(&b, *m) { if g > best_gain { best_gain = g; best_cap = *m; } }
    }

    // Rank with SEE ordering disabled
    let off_rank = {
        let mut s = PlecoSearcher::default(); s.set_see_ordering(false);
        let ordered = s.debug_order_no_parent(&b);
        ordered.iter().position(|&m| m == best_cap).unwrap_or(ordered.len())
    };

    // Rank with SEE ordering enabled
    let on_rank = {
        let mut s = PlecoSearcher::default(); s.set_see_ordering(true);
        let ordered = s.debug_order_no_parent(&b);
        ordered.iter().position(|&m| m == best_cap).unwrap_or(ordered.len())
    };

    assert!(on_rank <= off_rank, "best SEE capture should not rank worse with SEE ordering (on={} off={})", on_rank, off_rank);
}

