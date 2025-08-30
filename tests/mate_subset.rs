use cozy_chess::Board;

fn solve_exp(fen: &str, depth: u32) -> (Option<String>, i32, u64) {
    let board = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    s.set_tt_capacity_mb(64);
    let mut p = piebot::search::alphabeta_temp::SearchParams::default();
    p.depth = depth;
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.use_killers = true;
    p.use_lmr = true;
    p.use_nullmove = true;
    p.use_aspiration = true;
    p.threads = 1; // deterministic
    p.deterministic = true;
    let r = s.search_with_params(&board, p);
    (r.bestmove, r.score_cp, r.nodes)
}

#[test]
#[ignore]
fn mate_subset_idx_15_depth7() {
    let fen = "r1b2rk1/p4ppp/2p5/6q1/6P1/3p1Q1P/PPP5/1K2RR2 w - - 0 18";
    let (bm, sc, _n) = solve_exp(fen, 7);
    if bm.as_deref() != Some("f3f7") {
        assert!(sc >= 25_000, "expected f3f7 or any mating score, got bm={:?} sc={}", bm, sc);
    }
}

#[test]
#[ignore]
fn mate_subset_idx_28_depth7() {
    let fen = "2r3k1/pb3ppp/1p2pq2/1P6/2Q1PP2/6P1/P5BP/2R3K1 w - - 1 25";
    let (bm, sc, _n) = solve_exp(fen, 7);
    if bm.as_deref() != Some("c4c8") {
        assert!(sc >= 25_000, "expected c4c8 or any mating score, got bm={:?} sc={}", bm, sc);
    }
}

#[test]
#[ignore]
fn mate_subset_idx_32_depth7() {
    let fen = "2kr4/2pR4/2P1K1P1/8/8/4n3/p7/8 w - - 0 50";
    let (bm, sc, _n) = solve_exp(fen, 7);
    if bm.as_deref() != Some("d7d8") {
        assert!(sc >= 25_000, "expected d7d8 or any mating score, got bm={:?} sc={}", bm, sc);
    }
}

#[test]
#[ignore]
fn mate_subset_idx_43_depth7() {
    let fen = "8/6QR/pr5p/6p1/5p1k/q6P/2P2PPK/8 b - - 7 39";
    let (bm, sc, _n) = solve_exp(fen, 7);
    if bm.as_deref() != Some("a3g3") {
        assert!(sc >= 25_000, "expected a3g3 or any mating score, got bm={:?} sc={}", bm, sc);
    }
}
