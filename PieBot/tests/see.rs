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

#[test]
fn see_values_the_pawn_an_en_passant_capture_actually_wins() {
    use piebot::search::alphabeta::Searcher;
    // White pawn on e5, black has just played d7-d5; exd6 e.p. takes a pawn
    // that does NOT sit on the destination square. Reading the victim off the
    // destination square alone therefore scores this as winning nothing.
    // Nothing recaptures on d6, so the exchange is worth exactly a clean pawn.
    let fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let gain = s.see_gain_cp(&b, "e5d6").expect("en passant is legal here");
    assert_eq!(
        gain, 100,
        "en passant wins a pawn outright; scoring it {gain} understates it by a pawn",
    );
}

#[test]
fn see_counts_the_queen_a_promotion_puts_on_the_board() {
    use piebot::search::alphabeta::Searcher;
    // White pawn on b7 promotes on b8 with nothing able to recapture. The
    // exchange is worth the promotion gain (queen minus the pawn spent), so
    // valuing the post-move occupant as a pawn loses the whole point.
    let fen = "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let gain = s.see_gain_cp(&b, "b7b8q").expect("promotion is legal here");
    assert!(
        gain >= 700,
        "an unopposed promotion is worth roughly a queen minus a pawn; got {gain}",
    );
}

#[test]
fn see_sees_that_a_defended_promotion_square_makes_the_push_losing() {
    use piebot::search::alphabeta::Searcher;
    // Same promotion, but a black rook on a8 covers b8. White gives up a pawn
    // and a queen for nothing but the rook: the exchange must come out
    // negative. This only works if the recapture is valued against a QUEEN
    // sitting on b8, which is what the promotion fix establishes.
    let fen = "r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1";
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let gain = s.see_gain_cp(&b, "b7b8q").expect("promotion is legal here");
    assert!(
        gain < 0,
        "promoting into a defended square loses the new queen; got {gain}",
    );
}
