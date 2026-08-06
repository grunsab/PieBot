use cozy_chess::Board;

const MIDGAME_FENS: [&str; 4] = [
    // Italian-ish middlegame
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
    // Queen's pawn structure with tension
    "rnbq1rk1/ppp1ppbp/5np1/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 1 6",
    // Open center, tactical potential
    "r2qkb1r/ppp2ppp/2n1bn2/3pp3/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 w kq - 4 6",
    // Late middlegame
    "2rq1rk1/pb2bppp/1pn1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 11",
];

macro_rules! configure_params {
    ($p:expr, $depth:expr) => {{
        let mut p = $p;
        p.depth = $depth;
        p.use_tt = true;
        p.order_captures = true;
        p.use_history = true;
        p.threads = 1;
        p.use_aspiration = false;
        p.use_lmr = true;
        p.use_killers = true;
        p.use_nullmove = true;
        p.deterministic = true;
        p
    }};
}

fn baseline_search(fen: &str, depth: u32) -> (i32, u64) {
    let board = Board::from_fen(fen, false).expect("valid fen");
    let mut searcher = piebot::search::alphabeta::Searcher::default();
    searcher.set_tt_capacity_mb(64);
    let params = configure_params!(
        piebot::search::alphabeta::SearchParams::default(),
        depth
    );
    let res = searcher.search_with_params(&board, params);
    (res.score_cp, res.nodes)
}

fn experimental_search(fen: &str, depth: u32) -> (i32, u64) {
    let board = Board::from_fen(fen, false).expect("valid fen");
    let mut searcher = piebot::search::alphabeta_temp::Searcher::default();
    searcher.set_tt_capacity_mb(64);
    let params = configure_params!(
        piebot::search::alphabeta_temp::SearchParams::default(),
        depth
    );
    let res = searcher.search_with_params(&board, params);
    (res.score_cp, res.nodes)
}

#[test]
fn interior_pvs_keeps_scores_and_reduces_total_nodes() {
    let depth = 6;
    let mut base_nodes_total = 0u64;
    let mut exp_nodes_total = 0u64;
    for fen in MIDGAME_FENS {
        let (base_score, base_nodes) = baseline_search(fen, depth);
        let (exp_score, exp_nodes) = experimental_search(fen, depth);
        assert_eq!(
            base_score, exp_score,
            "PVS must be score-exact on {fen} at depth {depth}"
        );
        base_nodes_total += base_nodes;
        exp_nodes_total += exp_nodes;
    }
    assert!(
        exp_nodes_total < base_nodes_total,
        "interior PVS should search fewer total nodes: experimental {exp_nodes_total} vs baseline {base_nodes_total}"
    );
}
