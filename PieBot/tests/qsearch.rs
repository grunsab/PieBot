use cozy_chess::Board;

#[test]
fn qsearch_improves_tactical_position() {
    use piebot::search::alphabeta::Searcher;
    let fen = "4k3/8/8/8/5Q2/8/8/2b4K b - - 0 1"; // hanging queen vs bishop
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let stand = piebot::search::eval::eval_cp(&b);
    let qs = s.qsearch_eval_cp(&b);
    // For black to move, eval is from side-to-move perspective; capturing queen should improve score over stand pat.
    assert!(
        qs > stand,
        "qsearch should improve eval: qs {qs} vs stand {stand}"
    );
}

#[test]
fn qsearch_equals_standpat_without_captures() {
    use piebot::search::alphabeta::Searcher;
    let fen = "k7/8/8/8/8/8/8/7K w - - 0 1"; // bare kings, no captures
    let b = Board::from_fen(fen, false).unwrap();
    let mut s = Searcher::default();
    let stand = piebot::search::eval::eval_cp(&b);
    let qs = s.qsearch_eval_cp(&b);
    assert_eq!(qs, stand, "qsearch should equal stand pat without captures");
}

#[test]
fn qsearch_temp_matches_baseline_tactical_evaluations() {
    let fens = [
        "4k3/8/8/8/5Q2/8/8/2b4K b - - 0 1",
        "r1bqk2r/pppp1ppp/2n5/4p3/1bB1n3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 0 6",
        "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "r2q1rk1/ppp2ppp/2n1bn2/3pp3/1bPP4/2N1PN2/PP1BBPPP/R2QK2R w KQ - 4 8",
        "r1b1k2r/ppppqppp/2n2n2/4p3/1bPP4/2N1PN2/PP2BPPP/R1BQK2R b KQkq - 3 6",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ];

    for fen in fens {
        let board = Board::from_fen(fen, false).unwrap();
        let mut baseline = piebot::search::alphabeta::Searcher::default();
        let mut temp = piebot::search::alphabeta_temp::Searcher::default();

        let base_eval = baseline.qsearch_eval_cp(&board);
        let temp_eval = temp.qsearch_eval_cp(&board);

        assert_eq!(
            base_eval, temp_eval,
            "qsearch eval mismatch on FEN {fen}: baseline {base_eval} vs temp {temp_eval}"
        );

        let base_res = baseline.search_depth(&board, 2);
        let temp_res = temp.search_depth(&board, 2);

        assert_eq!(
            base_res.score_cp, temp_res.score_cp,
            "score mismatch on FEN {fen}: baseline {} vs temp {}",
            base_res.score_cp, temp_res.score_cp
        );
        assert_eq!(
            base_res.nodes, temp_res.nodes,
            "node mismatch on FEN {fen}: baseline {} vs temp {}",
            base_res.nodes, temp_res.nodes
        );
    }
}
