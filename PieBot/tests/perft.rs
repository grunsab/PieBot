use cozy_chess::Board;
use piebot::perft::perft;

#[test]
fn perft_startpos_small_depths() {
    let b = Board::default();
    assert_eq!(perft(&b, 1), 20);
    assert_eq!(perft(&b, 2), 400);
    assert_eq!(perft(&b, 3), 8902);
    assert_eq!(perft(&b, 4), 197281);
    assert_eq!(perft(&b, 5), 4_865_609);
}

#[test]
fn perft_kiwipete_exercises_castling_checks_and_pins() {
    let b = Board::from_fen(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        false,
    )
    .expect("valid Kiwipete FEN");
    assert_eq!(perft(&b, 1), 48);
    assert_eq!(perft(&b, 2), 2_039);
    assert_eq!(perft(&b, 3), 97_862);
    assert_eq!(perft(&b, 4), 4_085_603);
}
