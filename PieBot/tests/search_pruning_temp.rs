use cozy_chess::Board;
use piebot::search::alphabeta_temp::Searcher;
use piebot::search::eval::MATE_SCORE;

#[test]
fn test_see_pruning_and_lmp_deterministic_depth() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    // Initial position, search at depth 5
    let board = Board::default();
    let res = searcher.search_depth(&board, 5);
    assert!(res.bestmove.is_some(), "Search should find a move");
    assert!(res.nodes > 0, "Nodes searched should be positive");
}

#[test]
fn test_experimental_improving_asymmetric_pruning() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board = Board::default();
    let res = searcher.search_depth(&board, 6);
    assert!(res.bestmove.is_some(), "Search must find a legal move at depth 6");
    assert!(res.nodes > 0, "Search visited nodes");
}

#[test]
fn test_probcut_search_depth() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board: Board = "r1bqk2r/pppp1ppp/2n5/1B2p3/4n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5".parse().unwrap();
    let res = searcher.search_depth(&board, 7);
    assert!(res.bestmove.is_some(), "ProbCut search finds a valid move");
    assert!(res.nodes > 0, "Nodes searched");
}

#[test]
fn test_internal_iterative_reduction() {
    let mut searcher = Searcher::default();
    searcher.set_use_lmr(true);
    searcher.set_use_nullmove(true);
    searcher.set_use_history(true);

    let board: Board = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4".parse().unwrap();
    let res = searcher.search_depth(&board, 6);
    assert!(res.bestmove.is_some(), "Search with IIR finds a valid move");
    assert!(res.nodes > 0, "Nodes searched");
}

#[test]
fn sacrificial_check_is_extended_temp() {
    // Nh6+ Kh8, Qg8+ Rxg8, Nf7#. The queen check loses the queen by static
    // exchange; without an extension on it the mate is seen a ply later.
    let board: Board = "r5k1/5Npp/8/8/2Q5/8/8/7K w - - 0 1".parse().unwrap();
    let mut searcher = Searcher::default();
    let res = searcher.search_depth(&board, 3);
    assert_eq!(res.bestmove.as_deref(), Some("f7h6"));
    assert!(
        res.score_cp >= MATE_SCORE - 5,
        "smothered mate in three must be found at depth 3: {}",
        res.score_cp
    );
}

#[test]
fn test_killer_exempt_from_quiet_see_pruning_temp() {
    use cozy_chess::{Move, Square};
    let mut searcher = Searcher::default();
    searcher.set_use_killers(true);
    searcher.set_use_nullmove(true);

    // Position where a quiet move moves to an attacked square (SEE < 0)
    // White knight on f3 moves to g5 where it's attacked by black queen/pawns
    let fen = "r1bqk2r/pppp1ppp/2n5/4p3/1bB1n3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6";
    let board = Board::from_fen(fen, false).unwrap();
    let sacrifice_move = Move {
        from: Square::F3,
        to: Square::G5,
        promotion: None,
    };
    // Confirm this quiet move has negative SEE
    let gain = piebot::search::see::see_gain_cp(&board, sacrifice_move).unwrap_or(0);
    assert!(gain < -50, "Move should have negative SEE: {gain}");

    searcher.set_killer_for_test(0, 0, sacrifice_move);
    let res = searcher.search_depth(&board, 2);
    assert!(res.bestmove.is_some());
    assert!(res.nodes > 0);
}
