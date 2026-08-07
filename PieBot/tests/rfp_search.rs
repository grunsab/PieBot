use cozy_chess::Board;

const MIDGAME_FENS: [&str; 4] = [
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
    "rnbq1rk1/ppp1ppbp/5np1/3p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 1 6",
    "r2qkb1r/ppp2ppp/2n1bn2/3pp3/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 w kq - 4 6",
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

#[test]
fn reverse_futility_reduces_total_nodes_at_fixed_depth() {
    let depth = 8;
    let mut base_total = 0u64;
    let mut exp_total = 0u64;
    for fen in MIDGAME_FENS {
        let board = Board::from_fen(fen, false).expect("valid fen");

        let mut base = piebot::search::alphabeta::Searcher::default();
        base.set_tt_capacity_mb(64);
        let bp = configure_params!(piebot::search::alphabeta::SearchParams::default(), depth);
        let br = base.search_with_params(&board, bp);
        assert!(br.bestmove.is_some());
        base_total += br.nodes;

        let mut exp = piebot::search::alphabeta_temp::Searcher::default();
        exp.set_tt_capacity_mb(64);
        let ep = configure_params!(
            piebot::search::alphabeta_temp::SearchParams::default(),
            depth
        );
        let er = exp.search_with_params(&board, ep);
        assert!(er.bestmove.is_some());
        exp_total += er.nodes;
    }
    // Static-eval beta cutoffs at shallow interior nodes should prune
    // substantially; tactical safety is gated by matein3 + A/B games.
    assert!(
        (exp_total as f64) < (base_total as f64) * 0.90,
        "RFP should cut >=10% of nodes: experimental {exp_total} vs baseline {base_total}"
    );
}
