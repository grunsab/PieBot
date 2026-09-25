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

#[test]
fn probcut_parity_between_baseline_and_temp() {
    let fens = [
        "r1bqk2r/pppp1ppp/2n5/4p3/1bB1n3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
    ];

    for fen in fens {
        let board = Board::from_fen(fen, false).expect("valid fen");
        let mut base_s = piebot::search::alphabeta::Searcher::default();
        let mut temp_s = piebot::search::alphabeta_temp::Searcher::default();

        let base_res = base_s.search_depth(&board, 6);
        let temp_res = temp_s.search_depth(&board, 6);

        assert_eq!(
            base_res.bestmove, temp_res.bestmove,
            "bestmove mismatch on FEN {fen}: base {:?} vs temp {:?}",
            base_res.bestmove, temp_res.bestmove
        );
        assert_eq!(
            base_res.score_cp, temp_res.score_cp,
            "score mismatch on FEN {fen}: base {} vs temp {}",
            base_res.score_cp, temp_res.score_cp
        );
        assert_eq!(
            base_res.nodes, temp_res.nodes,
            "nodes mismatch on FEN {fen}: base {} vs temp {}",
            base_res.nodes, temp_res.nodes
        );
    }
}
