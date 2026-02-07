use cozy_chess::Board;
use piebot::perft::perft;

#[test]
fn perft_startpos_small_depths() {
    let b = Board::default();
    assert_eq!(perft(&b, 1), 20);
    assert_eq!(perft(&b, 2), 400);
    assert_eq!(perft(&b, 3), 8902);
    assert_eq!(perft(&b, 4), 197281);
}
