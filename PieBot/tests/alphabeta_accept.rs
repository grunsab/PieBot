use cozy_chess::{Board, Move};

fn find_move_uci(board: &Board, uci: &str) -> Option<Move> {
    let mut found = None;
    board.generate_moves(|ml| {
        for m in ml {
            if format!("{}", m) == uci {
                found = Some(m);
                return true;
            }
        }
        false
    });
    found
}

#[test]
fn alphabeta_finds_simple_mate_in_one() {
    let board = Board::from_fen("k7/8/6Q1/8/8/8/8/K7 w - - 0 1", false).expect("valid FEN");
    let mut searcher = piebot::search::alphabeta::Searcher::default();
    searcher.set_tt_capacity_mb(16);

    let mut params = piebot::search::alphabeta::SearchParams::default();
    params.depth = 3;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.use_killers = true;
    params.use_nullmove = true;
    params.use_aspiration = true;
    params.use_lmr = true;
    params.aspiration_window_cp = 35;
    params.threads = 1;
    params.deterministic = true;

    let result = searcher.search_with_params(&board, params);
    let best_uci = result.bestmove.as_deref().expect("expected a best move");
    let best =
        find_move_uci(&board, best_uci).expect("best move should be legal in current position");
    let _ = best;
    assert!(
        result.score_cp > 500,
        "expected a clearly winning score, got {}",
        result.score_cp
    );
}

#[test]
fn temp_search_keeps_forcing_checking_sacrifice_at_root() {
    // matein3 suite index 15: Qxf7+ has negative static exchange value, but is
    // the forcing move. Root safety filters must not discard checking moves.
    let board = Board::from_fen(
        "r1b2rk1/p4ppp/2p5/6q1/6P1/3p1Q1P/PPP5/1K2RR2 w - - 0 18",
        false,
    )
    .expect("valid FEN");
    let mut searcher = piebot::search::alphabeta_temp::Searcher::default();
    searcher.set_tt_capacity_mb(128);
    let see = searcher
        .see_gain_cp(&board, "f3f7")
        .expect("Qxf7+ is a legal capture");
    assert!(
        see <= -500,
        "test requires a negative-SEE sacrifice, got {see}"
    );

    let mut params = piebot::search::alphabeta_temp::SearchParams::default();
    params.depth = 7;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.use_killers = true;
    params.use_nullmove = true;
    params.use_aspiration = true;
    params.use_lmr = true;
    params.threads = 1;
    params.deterministic = true;

    let result = searcher.search_with_params(&board, params);
    assert_eq!(
        result.bestmove.as_deref(),
        Some("f3f7"),
        "forcing Qxf7+ must survive root filtering (score={}, nodes={})",
        result.score_cp,
        result.nodes
    );
}
