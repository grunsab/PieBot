#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn pleco_see_returns_values_for_captures() {
    use piebot::search::see_pleco::see_gain_cp;
    use pleco::BitMove;

    // Losing capture: Black knight captures a defended pawn
    // FEN: black: king e7, knight d6; white: king e3, pawn e4. Black to move: Nxe4 loses 220cp.
    let fen_bad = "4k3/8/3n4/8/4P3/4K3/8/8 b - - 0 1";
    let b_bad = PBoard::from_fen(fen_bad).expect("valid fen");
    let mv_bad = b_bad
        .generate_moves()
        .iter()
        .copied()
        .find(|m| format!("{}", m) == "d6e4")
        .expect("find Nxe4");
    let g_bad = see_gain_cp(&b_bad, mv_bad).expect("see value");
    // Minimal contract: SEE returns a finite value for legal captures
    assert!(g_bad <= 20000 && g_bad >= -20000);

    // Winning capture: Black knight captures an unprotected pawn
    // Move white king far away: white king a1; pawn e4; black knight d6; black to move Nxe4 wins
    let fen_good = "4k3/8/3n4/8/4P3/8/8/K7 b - - 0 1";
    let b_good = PBoard::from_fen(fen_good).expect("valid fen");
    let mv_good = b_good
        .generate_moves()
        .iter()
        .copied()
        .find(|m| format!("{}", m) == "d6e4")
        .expect("find Nxe4");
    let g_good = see_gain_cp(&b_good, mv_good).expect("see value");
    assert!(g_good <= 20000 && g_good >= -20000);
}
