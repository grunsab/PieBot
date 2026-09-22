use cozy_chess::Board;
use piebot::search::alphabeta_temp::{SearchParams, Searcher};

#[test]
fn test_bad_captures_ordered_after_quiet_moves() {
    // Position where White has a suicidal capture (Qxd4 losing Queen for Knight)
    // and several healthy quiet moves.
    // FEN: 7k/8/8/2p5/3n4/8/8/Q3K3 w - - 0 1
    // In this position:
    // White Queen on a1 can capture d4 (Qxd4), but d4 is defended by pawn c5.
    // SEE for Qxd4 is 320 - 900 = -580 (strongly negative).
    let fen = "7k/8/8/2p5/3n4/8/8/Q3K3 w - - 0 1";
    let board = Board::from_fen(fen, false).expect("valid fen");

    let mut searcher = Searcher::default();
    let mut params = SearchParams::default();
    params.depth = 4;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.use_killers = true;
    params.use_nullmove = true;

    let res = searcher.search_with_params(&board, params);
    // Best move should NOT be Qxd4 (suicide)
    let best = res.bestmove.expect("should find a move");
    assert_ne!(best, "a1d4", "Engine should not play suicidal capture");
}
