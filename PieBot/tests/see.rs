use cozy_chess::Board;

#[test]
fn see_winning_capture_positive() {
    use piebot::search::alphabeta::Searcher;
    // Black to move: Bc1xf4 wins the white queen
    let fen = "4k3/8/8/8/5Q2/8/8/2b4K b - - 0 1";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let mv = "c1f4"; // bishop takes queen
    let gain = s.see_gain_cp(&b, mv).unwrap();
    assert!(gain > 400, "expected large positive gain, got {gain}");
}
