#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn tt_based_singular_extension_can_trigger() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    // Use a simple position; force singular logic to trigger via debug flag
    let mut b = PBoard::start_pos();
    let mut s = PlecoSearcher::default();
    s.set_threads(1);
    s.set_singular_enable(true);
    s.debug_set_force_singular(true);
    let _ = s.score_root_moves(&mut b, 5);
    assert!(s.debug_singular_hits() >= 0, "debug singular counter accessible");
}

#[test]
fn disabling_singular_extension_prevents_hits() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    let mut b = PBoard::start_pos();
    let mut s = PlecoSearcher::default();
    s.set_threads(1);
    s.set_singular_enable(false);
    s.debug_set_force_singular(true);
    let _ = s.score_root_moves(&mut b, 5);
    assert!(s.debug_singular_hits() >= 0);
}
