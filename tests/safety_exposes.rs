use cozy_chess::Board;

#[test]
fn exposes_heavy_loss_after_bxh7() {
    // From user: After Bxh7, ...Nxf3+ wins the white queen on f3.
    let fen = "r1bqkb1r/1pp1pppp/5n2/3Pn3/p7/P1NB1Q2/1PP2PPP/R1B1K2R w KQkq - 1 9";
    let board = Board::from_fen(fen, false).expect("valid FEN");
    // Find Bxh7 (d3h7)
    let mut mv = None;
    board.generate_moves(|ml| { for m in ml { if format!("{}", m) == "d3h7" { mv = Some(m); break; } } mv.is_some() });
    let m = mv.expect("Bxh7 must be legal here");
    assert!(piebot::search::safety::exposes_heavy_loss_after_move(&board, m, 500),
        "expected Bxh7 to expose a heavy immediate loss (queen) via ...Nxf3+");
}

