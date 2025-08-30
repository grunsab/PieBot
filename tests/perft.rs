use piebot::perft::perft;

#[test]
fn perft_startpos_small_depths() {
    #[cfg(feature = "board-pleco")]
    {
        let mut b = pleco::Board::start_pos();
        assert_eq!(perft(&mut b, 1), 20);
        assert_eq!(perft(&mut b, 2), 400);
        assert_eq!(perft(&mut b, 3), 8902);
        assert_eq!(perft(&mut b, 4), 197281);
    }
    #[cfg(not(feature = "board-pleco"))]
    {
        let b = cozy_chess::Board::default();
        assert_eq!(perft(&b, 1), 20);
        assert_eq!(perft(&b, 2), 400);
        assert_eq!(perft(&b, 3), 8902);
        assert_eq!(perft(&b, 4), 197281);
    }
}
