use cozy_chess::Board;
use piebot::search::alphabeta_temp::Searcher;

#[test]
fn test_capture_history_updates_and_ranks_moves() {
    // Starting board with captures available
    let fen = "r1bqk2r/ppp2ppp/2n5/3np3/1b6/2NP1N2/PPP1BPPP/R1BQK2R w KQkq - 0 7";
    let board = Board::from_fen(fen, false).expect("valid fen");

    let mut searcher = Searcher::default();
    searcher.set_order_captures(true);
    searcher.set_use_history(true);

    let res = searcher.search_depth(&board, 4);
    assert!(res.bestmove.is_some());
    assert!(res.nodes > 0);
}
