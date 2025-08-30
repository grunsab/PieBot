use cozy_chess::Board;

#[test]
fn bishop_e5_is_hanging_blunder() {
    // FEN from report: baseline/experimental played Be5 hanging the bishop to ...Rxe5
    let fen = "8/6R1/7k/2p5/5p2/4r3/8/B6K w - - 8 64";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Find Be5 move (a1e5)
    let mut target = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == "a1e5" { target = Some(m); break; } }
        target.is_some()
    });
    let mv = target.expect("Be5 should be legal here");
    // The blunder guard should flag this as hanging by SEE
    assert!(piebot::search::safety::is_hanging_after_move(&board, mv, 200));
}

