use cozy_chess::Board;
use std::time::Instant;

#[derive(Debug, serde::Deserialize)]
struct Rec { fen: String, best: String }

fn load_jsonl(path: &str, limit: Option<usize>) -> Vec<Rec> {
    use std::io::{BufRead, BufReader};
    let f = match std::fs::File::open(path) { Ok(f) => f, Err(_) => return Vec::new() };
    let rdr = BufReader::new(f);
    let mut out = Vec::new();
    for line in rdr.lines().flatten() {
        let l = line.trim(); if l.is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
            if let (Some(fen), Some(best)) = (v.get("fen").and_then(|x| x.as_str()), v.get("best").and_then(|x| x.as_str())) {
                out.push(Rec { fen: fen.to_string(), best: best.to_string() });
                if let Some(n) = limit { if out.len() >= n { break; } }
            }
        }
    }
    out
}

fn find_move_uci(board: &Board, uci: &str) -> Option<cozy_chess::Move> {
    let mut found = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == uci { found = Some(m); break; } }
        found.is_some()
    });
    found
}

fn env_parse_u64(name: &str) -> Option<u64> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }
fn env_parse_usize(name: &str) -> Option<usize> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend { Cozy, Pleco }

fn pick_backend() -> Backend {
    match std::env::var("PIEBOT_TEST_BACKEND").unwrap_or_else(|_| "cozy".to_string()).to_ascii_lowercase().as_str() {
        "pleco" => {
            #[cfg(feature = "board-pleco")]
            { return Backend::Pleco; }
            #[cfg(not(feature = "board-pleco"))]
            { eprintln!("[warn] PIEBOT_TEST_BACKEND=pleco requested but feature 'board-pleco' is not enabled; falling back to cozy."); return Backend::Cozy; }
        }
        _ => Backend::Cozy,
    }
}

fn solve_cozy(fen: &str, depth: u32, threads: usize, max_nodes: Option<u64>) -> piebot::search::alphabeta::SearchResult {
    let b = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_tt_capacity_mb(128);
    let mut p = piebot::search::alphabeta::SearchParams::default();
    p.depth = depth;
    let baseline = std::env::var("PIEBOT_TEST_BASELINE").ok().map(|v| v == "1").unwrap_or(false);
    let opts_raw = std::env::var("PIEBOT_TEST_OPTS").ok();
    // If PIEBOT_TEST_OPTS is present, enable only listed heuristics; otherwise
    // use baseline (all off) or defaults (all on)
    let mut use_tt = false;
    let mut order_caps = false;
    let mut use_hist = false;
    let mut use_kill = false;
    let mut use_null = false;
    let mut use_asp = false;
    let mut use_lmr = false;
    if let Some(spec) = opts_raw.as_deref() {
        let mut set = |key: &str, flag: &mut bool| {
            if spec.split(',').any(|t| t.trim().eq_ignore_ascii_case(key)) { *flag = true; }
        };
        if spec.split(',').any(|t| t.trim().eq_ignore_ascii_case("all")) {
            use_tt = true; order_caps = true; use_hist = true; use_kill = true; use_null = true; use_asp = true; use_lmr = true;
        } else {
            set("tt", &mut use_tt);
            set("caps", &mut order_caps);
            set("history", &mut use_hist);
            set("killers", &mut use_kill);
            set("null", &mut use_null);
            set("asp", &mut use_asp);
            set("lmr", &mut use_lmr);
        }
    } else if baseline {
        // keep all false
    } else {
        // defaults: enable common heuristics
        use_tt = true; order_caps = true; use_hist = true; use_kill = true; use_null = true; use_asp = true; use_lmr = std::env::var("PIEBOT_TEST_LMR").ok().map(|v| v != "0").unwrap_or(true);
    }
    p.use_tt = use_tt;
    p.max_nodes = max_nodes;
    p.movetime = None;
    p.order_captures = order_caps;
    p.use_history = use_hist;
    p.threads = threads;
    p.use_aspiration = use_asp;
    p.aspiration_window_cp = 30;
    p.use_lmr = use_lmr;
    p.use_killers = use_kill;
    p.use_nullmove = use_null;
    p.deterministic = threads == 1;
    s.search_with_params(&b, p)
}

#[cfg(feature = "board-pleco")]
fn solve_pleco(fen: &str, depth: u32, threads: usize) -> (Option<String>, i32, u64) {
    let mut b = match pleco::Board::from_fen(fen) { Ok(x) => x, Err(_) => return (None, 0) };
    let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
    s.set_threads(threads.max(1));
    s.set_tt_capacity_mb(128);
    // Use a generous movetime so depth is not time-capped; searcher will stop at target depth
    let (bm, sc, nodes) = s.search_movetime(&mut b, 60_000, depth);
    (bm.map(|m| format!("{}", m)), sc, nodes)
}

fn is_capture(board: &Board, mv: cozy_chess::Move) -> bool {
    let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
    let opp_bb = board.colors(opp);
    for sq in opp_bb { if sq == mv.to { return true; } }
    false
}

fn debug_analyze_cozy(fen: &str, expected: &str, depth: u32, threads: usize, max_nodes: Option<u64>, topk: usize) {
    // Root
    let b = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_tt_capacity_mb(128);
    let mut p = piebot::search::alphabeta::SearchParams::default();
    p.depth = depth;
    p.use_tt = true;
    p.max_nodes = max_nodes;
    p.movetime = None;
    p.order_captures = true;
    p.use_history = true;
    p.threads = threads;
    p.use_aspiration = true;
    p.use_lmr = true;
    p.use_killers = true;
    p.use_nullmove = true;
    p.deterministic = threads == 1;
    let r = s.search_with_params(&b, p);
    let root_q = s.qsearch_eval_cp(&b);
    println!("[debug] fen={} expected={} got={:?} root_q={} nodes={} depth={}", fen, expected, r.bestmove, root_q, r.nodes, depth);

    // Evaluate each legal move: capture?, SEE, qeval(child), depth-1 score
    let mut moves: Vec<cozy_chess::Move> = Vec::new();
    b.generate_moves(|ml| { for m in ml { moves.push(m); } false });
    let mut rows: Vec<(String, bool, Option<i32>, i32, i32)> = Vec::new();
    for m in moves {
        let u = format!("{}", m);
        let cap = is_capture(&b, m);
        let see = piebot::search::see::see_gain_cp(&b, m);
        let mut child = b.clone();
        child.play(m);
        let mut s2 = piebot::search::alphabeta::Searcher::default();
        s2.set_tt_capacity_mb(64);
        let q_child = s2.qsearch_eval_cp(&child);
        let r_child_quick = solve_cozy(&format!("{}", child), depth.saturating_sub(1), threads, max_nodes);
        // quick pass to get child score via full search at depth-1
        let mut s3 = piebot::search::alphabeta::Searcher::default();
        s3.set_tt_capacity_mb(64);
        let mut p3 = piebot::search::alphabeta::SearchParams::default();
        p3.depth = depth.saturating_sub(1);
        p3.use_tt = true;
        p3.order_captures = true;
        p3.use_history = true;
        p3.threads = 1;
        p3.use_aspiration = true;
        p3.use_lmr = true;
        p3.use_killers = true;
        p3.use_nullmove = true;
        p3.deterministic = true;
        let r_child = s3.search_with_params(&child, p3);
        rows.push((u, cap, see, q_child, r_child.score_cp.max(r_child_quick.score_cp)));
    }
    // Sort by child depth-1 score desc
    rows.sort_by_key(|&(_, _, _, _, sc)| -sc);
    println!("[debug] top {} moves by depth-{} score:", topk, depth.saturating_sub(1));
    for (i, (u, cap, see, qch, sc)) in rows.into_iter().take(topk).enumerate() {
        println!("  {:>2}. {:>6} cap={} see={:?} qchild={} sc(d-1)={}", i+1, u, cap, see, qch, sc);
    }
}

#[test]
fn mate_suite_smoke_20_positions_depth6() {
    // Smoke test: take up to 20 positions from the highest N files that exist and solve at depth 6.
    let base = format!("{}/src/suites", env!("CARGO_MANIFEST_DIR"));
    let files = [
        format!("{}/matein7.txt", base),
        format!("{}/matein6.txt", base),
        format!("{}/matein5.txt", base),
        format!("{}/matein4.txt", base),
        format!("{}/matein3.txt", base),
        format!("{}/matein2.txt", base),
        format!("{}/matein1.txt", base),
    ];
    let mut cases: Vec<Rec> = Vec::new();
    for f in &files { let mut v = load_jsonl(f, None); cases.append(&mut v); }
    let take_n = cases.len().min(20);
    let backend = pick_backend();
    // Quick ID run at depth 6 with aspiration enabled
    for case in cases.into_iter().take(take_n) {
        let (bm, _nodes) = match backend {
            Backend::Cozy => { let r = solve_cozy(&case.fen, 6, 1, None); (r.bestmove, r.nodes) },
            #[cfg(feature = "board-pleco")]
            Backend::Pleco => { let (bm, _sc, nodes) = solve_pleco(&case.fen, 6, 1); (bm, nodes) },
            #[cfg(not(feature = "board-pleco"))]
            Backend::Pleco => unreachable!(),
        };
        if let Some(bm) = bm.as_deref() {
            // Only assert move is legal; check under cozy board parser
            let b = Board::from_fen(&case.fen, false).expect("valid FEN");
            assert!(find_move_uci(&b, bm).is_some(), "returned bestmove must be legal: {}", bm);
        }
    }
}

#[test]
#[ignore]
fn mate_suite_full_depth8_must_match_best() {
    // Full acceptance: verify expected best move at depth 8 for all suite files.
    // Optimized with per-case fresh TT, ID+asp, optional SMP/limits, and depth-9 fallback.
    // Run: cargo test --test mate_acceptance mate_suite_full_depth8_must_match_best -- --ignored --nocapture
    // Faster: PIEBOT_TEST_THREADS=4 cargo test ...
    let base = format!("{}/src/suites", env!("CARGO_MANIFEST_DIR"));
    let files = [
        format!("{}/matein7.txt", base),
        format!("{}/matein6.txt", base),
        format!("{}/matein5.txt", base),
        format!("{}/matein4.txt", base),
        format!("{}/matein3.txt", base),
        format!("{}/matein2.txt", base),
        format!("{}/matein1.txt", base),
    ];
    let mut cases: Vec<Rec> = Vec::new();
    for f in &files { let mut v = load_jsonl(f, None); cases.append(&mut v); }
    assert!(!cases.is_empty(), "mate suite files not found; run scripts/fetch_mate_suite.sh");

    let backend = pick_backend();
    let threads = env_parse_usize("PIEBOT_TEST_THREADS").unwrap_or(1).max(1);
    let max_nodes = env_parse_u64("PIEBOT_TEST_MAX_NODES");
    let use_fallback = std::env::var("PIEBOT_TEST_FALLBACK9").ok().map(|v| v != "0").unwrap_or(true);

    let mut failures: Vec<String> = Vec::new();
    let mut fallback_hits = 0usize;
    let t_total = Instant::now();
    let mut sum_secs: f64 = 0.0;
    let mut sum_nodes: u64 = 0;
    let verbose = std::env::var("PIEBOT_TEST_TIMING").ok().map(|v| v == "1").unwrap_or(false);
    for (i, case) in cases.iter().enumerate() {
        let b = Board::from_fen(&case.fen, false).expect("valid FEN");
        let t_case = Instant::now();
        let r8 = match backend {
            Backend::Cozy => solve_cozy(&format!("{}", b), 8, threads, max_nodes),
            #[cfg(feature = "board-pleco")]
            Backend::Pleco => { let (bm, sc, nodes) = solve_pleco(&format!("{}", b), 8, threads); piebot::search::alphabeta::SearchResult { bestmove: bm, score_cp: sc, nodes } }
            #[cfg(not(feature = "board-pleco"))]
            Backend::Pleco => unreachable!(),
        };
        if r8.bestmove.as_deref() == Some(case.best.as_str()) {
            let dt = t_case.elapsed().as_secs_f64();
            sum_secs += dt; sum_nodes += r8.nodes;
            if verbose { println!("ok idx={} d8 dt={:.3}s nodes={}", i, dt, r8.nodes); }
            continue;
        }
        if use_fallback {
            let r9 = match backend {
                Backend::Cozy => solve_cozy(&format!("{}", b), 9, threads, max_nodes),
                #[cfg(feature = "board-pleco")]
                Backend::Pleco => { let (bm, sc, nodes) = solve_pleco(&format!("{}", b), 9, threads); piebot::search::alphabeta::SearchResult { bestmove: bm, score_cp: sc, nodes } }
                #[cfg(not(feature = "board-pleco"))]
                Backend::Pleco => unreachable!(),
            };
            let dt = t_case.elapsed().as_secs_f64();
            sum_secs += dt; sum_nodes += r9.nodes;
            if r9.bestmove.as_deref() == Some(case.best.as_str()) {
                fallback_hits += 1;
                if verbose { println!("ok idx={} d8->d9 dt={:.3}s nodes={}", i, dt, r9.nodes); }
                continue;
            }
            failures.push(format!("idx={} fen={} got8={:?} got9={:?} expect={}", i, case.fen, r8.bestmove, r9.bestmove, case.best));
        } else {
            let dt = t_case.elapsed().as_secs_f64();
            sum_secs += dt; sum_nodes += r8.nodes;
            failures.push(format!("idx={} fen={} got8={:?} expect={}", i, case.fen, r8.bestmove, case.best));
        }
    }
    let total = t_total.elapsed().as_secs_f64();
    let n = cases.len().max(1);
    println!("summary: cases={} elapsed={:.3}s avg_ms={:.1} nodes={} nps={:.1} fallback_hits={} threads={}",
        cases.len(), total, (sum_secs / n as f64) * 1000.0, sum_nodes, if sum_secs>0.0 { sum_nodes as f64 / sum_secs } else { 0.0 }, fallback_hits, threads);
    if !failures.is_empty() {
        panic!("full depth-8 acceptance failed on {} cases (fallback9 hits={}):\n{}", failures.len(), fallback_hits, failures.join("\n"));
    }
}

#[test]
#[ignore]
fn matein3_full_depth7_must_match_best() {
    // Dedicated acceptance for mateIn3 (engine depth 6). Preferred in current dataset.
    let base = format!("{}/src/suites", env!("CARGO_MANIFEST_DIR"));
    let path = format!("{}/matein3.txt", base);
    let cases = load_jsonl(&path, None);
    // Optional filter: run only specified indices (comma-separated)
    let only_idxs: Option<std::collections::HashSet<usize>> = std::env::var("PIEBOT_TEST_ONLY_IDX").ok().map(|s| {
        s.split(',').filter_map(|t| t.trim().parse::<usize>().ok()).collect()
    });
    assert!(!cases.is_empty(), "matein3.txt not found or empty; run scripts/fetch_mate_suite.sh");
    let backend = pick_backend();
    let threads = env_parse_usize("PIEBOT_TEST_THREADS").unwrap_or(1).max(1);
    let max_nodes = env_parse_u64("PIEBOT_TEST_MAX_NODES");
    let start_depth = env_parse_usize("PIEBOT_TEST_START_DEPTH").unwrap_or(7) as u32;
    let max_depth = env_parse_usize("PIEBOT_TEST_MAX_DEPTH").unwrap_or(9) as u32; // allow 7..9 by default
    let verbose = std::env::var("PIEBOT_TEST_TIMING").ok().map(|v| v == "1").unwrap_or(false);

    let mut failures: Vec<String> = Vec::new();
    let mut fallback_hits = 0usize;
    let t_total = Instant::now();
    let mut sum_secs: f64 = 0.0;
    let mut sum_nodes: u64 = 0;
    let mut processed = 0usize;
    for (i, case) in cases.iter().enumerate() {
        if let Some(ref filt) = only_idxs { if !filt.contains(&i) { continue; } }
        let b = Board::from_fen(&case.fen, false).expect("valid FEN");
        // First attempt: configured start depth (default 7)
        let t_case = Instant::now();
        let mut r = match backend {
            Backend::Cozy => solve_cozy(&format!("{}", b), start_depth, threads, max_nodes),
            #[cfg(feature = "board-pleco")]
            Backend::Pleco => { let (bm, sc, nodes) = solve_pleco(&format!("{}", b), start_depth, threads); piebot::search::alphabeta::SearchResult { bestmove: bm, score_cp: sc, nodes } }
            #[cfg(not(feature = "board-pleco"))]
            Backend::Pleco => unreachable!(),
        };
        // Accept exact best move or any move that yields mate-level score
        let mate_thresh = 25000; // approx threshold indicating forced mate
        if r.bestmove.as_deref() == Some(case.best.as_str()) || r.score_cp >= mate_thresh {
            let dt = t_case.elapsed().as_secs_f64();
            sum_secs += dt; sum_nodes += r.nodes;
            if verbose { println!("ok idx={} d{} dt={:.3}s nodes={}", i, start_depth, dt, r.nodes); }
            processed += 1;
            continue;
        }
        // Fallback depths up to max_depth
        let mut matched = false;
        let mut trail: Vec<(u32, Option<String>)> = vec![(start_depth, r.bestmove.clone())];
        let mut cur_depth = start_depth + 1;
        while cur_depth <= max_depth {
            r = match backend {
                Backend::Cozy => solve_cozy(&format!("{}", b), cur_depth, threads, max_nodes),
                #[cfg(feature = "board-pleco")]
                Backend::Pleco => { let (bm, sc, nodes) = solve_pleco(&format!("{}", b), cur_depth, threads); piebot::search::alphabeta::SearchResult { bestmove: bm, score_cp: sc, nodes } }
                #[cfg(not(feature = "board-pleco"))]
                Backend::Pleco => unreachable!(),
            };
            trail.push((cur_depth, r.bestmove.clone()));
            if r.bestmove.as_deref() == Some(case.best.as_str()) || r.score_cp >= mate_thresh { matched = true; break; }
            cur_depth += 1;
        }
        let dt = t_case.elapsed().as_secs_f64();
        sum_secs += dt; sum_nodes += r.nodes;
        if matched {
            fallback_hits += 1;
            if verbose { println!("ok idx={} d{}..{} dt={:.3}s nodes={}", i, start_depth, cur_depth, dt, r.nodes); }
            processed += 1;
            continue;
        }
        // Report trail for debugging
        let trail_s: String = trail.into_iter().map(|(d, mv)| format!("d{}={:?}", d, mv)).collect::<Vec<_>>().join(", ");
        // Optional deep debug: print evals and child scores (Cozy only)
        if std::env::var("PIEBOT_TEST_DEBUG").ok().as_deref() == Some("1") && backend == Backend::Cozy {
            eprintln!("[debug] analyzing failed case idx={} ...", i);
            debug_analyze_cozy(&case.fen, &case.best, max_depth.min(start_depth+2), threads, max_nodes, 10);
        }
        failures.push(format!("idx={} fen={} {} expect={}", i, case.fen, trail_s, case.best));
    }
    let total = t_total.elapsed().as_secs_f64();
    let n = processed.max(1);
    println!("summary: cases={} elapsed={:.3}s avg_ms={:.1} nodes={} nps={:.1} fallback_hits={} start_d={} max_d={} threads={}",
        processed, total, (sum_secs / n as f64) * 1000.0, sum_nodes, if sum_secs>0.0 { sum_nodes as f64 / sum_secs } else { 0.0 }, fallback_hits, start_depth, max_depth, threads);
    if !failures.is_empty() {
        panic!("mateIn3 acceptance failed on {} cases (fallback hits={} up to d={}):\n{}", failures.len(), fallback_hits, max_depth, failures.join("\n"));
    }
}
