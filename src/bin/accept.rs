use cozy_chess::Board;
use std::time::Instant;
use std::collections::HashSet;

#[derive(Debug, serde::Deserialize)]
struct Rec { fen: String, best: String }

fn load_jsonl(path: &str) -> Vec<Rec> {
    use std::io::{BufRead, BufReader};
    let f = match std::fs::File::open(path) { Ok(f) => f, Err(_) => return Vec::new() };
    let rdr = BufReader::new(f);
    let mut out = Vec::new();
    for line in rdr.lines().flatten() {
        let l = line.trim(); if l.is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
            if let (Some(fen), Some(best)) = (v.get("fen").and_then(|x| x.as_str()), v.get("best").and_then(|x| x.as_str())) {
                out.push(Rec { fen: fen.to_string(), best: best.to_string() });
            }
        }
    }
    out
}

fn env_parse_usize(name: &str) -> Option<usize> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }
fn env_parse_u64(name: &str) -> Option<u64> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }

fn solve_cozy(fen: &str, depth: u32, threads: usize, max_nodes: Option<u64>) -> piebot::search::alphabeta::SearchResult {
    let b = Board::from_fen(fen, false).expect("valid FEN");
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_tt_capacity_mb(128);
    let mut p = piebot::search::alphabeta::SearchParams::default();
    p.depth = depth;
    let baseline = std::env::var("PIEBOT_TEST_BASELINE").ok().map(|v| v == "1").unwrap_or(false);
    let opts_raw = std::env::var("PIEBOT_TEST_OPTS").ok();
    let mut use_tt = false; let mut order_caps = false; let mut use_hist = false; let mut use_kill = false; let mut use_null = false; let mut use_asp = false; let mut use_lmr = false;
    if let Some(spec) = opts_raw.as_deref() {
        let mut set = |key: &str, flag: &mut bool| {
            if spec.split(',').any(|t| t.trim().eq_ignore_ascii_case(key)) { *flag = true; }
        };
        if spec.split(',').any(|t| t.trim().eq_ignore_ascii_case("all")) { use_tt = true; order_caps = true; use_hist = true; use_kill = true; use_null = true; use_asp = true; use_lmr = true; }
        else { set("tt", &mut use_tt); set("caps", &mut order_caps); set("history", &mut use_hist); set("killers", &mut use_kill); set("null", &mut use_null); set("asp", &mut use_asp); set("lmr", &mut use_lmr); }
    } else if baseline { /* keep all false */ } else { use_tt = true; order_caps = true; use_hist = true; use_kill = true; use_null = true; use_asp = true; use_lmr = true; }
    p.use_tt = use_tt; p.order_captures = order_caps; p.use_history = use_hist; p.threads = threads; p.use_aspiration = use_asp; p.use_lmr = use_lmr; p.use_killers = use_kill; p.use_nullmove = use_null;
    p.max_nodes = max_nodes; p.movetime = None; p.deterministic = threads == 1;
    s.search_with_params(&b, p)
}

fn main() {
    // Inputs via env (for simplicity)
    let base = format!("{}/src/suites", env!("CARGO_MANIFEST_DIR"));
    let path = std::env::var("PIEBOT_SUITE_FILE").ok().unwrap_or_else(|| format!("{}/matein3.txt", base));
    let cases = load_jsonl(&path);
    let threads = env_parse_usize("PIEBOT_TEST_THREADS").unwrap_or(1).max(1);
    let max_nodes = env_parse_u64("PIEBOT_TEST_MAX_NODES");
    let start_depth = env_parse_usize("PIEBOT_TEST_START_DEPTH").unwrap_or(7) as u32;
    let max_depth = match env_parse_usize("PIEBOT_TEST_MAX_DEPTH") {
        Some(v) => v as u32,
        None => start_depth,
    };
    let verbose = std::env::var("PIEBOT_TEST_TIMING").ok().map(|v| v == "1").unwrap_or(false);
    let mate_thresh = 25_000; // accept any mating move
    let only: Option<HashSet<usize>> = std::env::var("PIEBOT_TEST_ONLY_IDX").ok().map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());

    let t_total = Instant::now();
    let mut failures: Vec<String> = Vec::new();
    let mut fallback_hits = 0usize;
    let mut sum_secs = 0.0f64; let mut sum_nodes: u64 = 0; let mut processed = 0usize;

    for (i, case) in cases.iter().enumerate() {
        if let Some(ref filt) = only { if !filt.contains(&i) { continue; } }
        let t_case = Instant::now();
        let mut r = solve_cozy(&case.fen, start_depth, threads, max_nodes);
        if r.bestmove.as_deref() == Some(case.best.as_str()) || r.score_cp >= mate_thresh {
            let dt = t_case.elapsed().as_secs_f64(); sum_secs += dt; sum_nodes += r.nodes; processed += 1;
            if verbose { println!("ok idx={} d{} dt={:.3}s nodes={}", i, start_depth, dt, r.nodes); }
            continue;
        }
        // fallback ladder
        let mut matched = false; let mut cur_depth = start_depth + 1; let mut trail = vec![(start_depth, r.bestmove.clone())];
        while cur_depth <= max_depth {
            r = solve_cozy(&case.fen, cur_depth, threads, max_nodes);
            trail.push((cur_depth, r.bestmove.clone()));
            if r.bestmove.as_deref() == Some(case.best.as_str()) || r.score_cp >= mate_thresh { matched = true; break; }
            cur_depth += 1;
        }
        let dt = t_case.elapsed().as_secs_f64(); sum_secs += dt; sum_nodes += r.nodes;
        if matched { fallback_hits += 1; processed += 1; if verbose { println!("ok idx={} d..{} dt={:.3}s nodes={}", i, cur_depth, dt, r.nodes); } continue; }
        let trail_s: String = trail.into_iter().map(|(d, mv)| format!("d{}={:?}", d, mv)).collect::<Vec<_>>().join(", ");
        failures.push(format!("idx={} fen={} {} expect={}", i, case.fen, trail_s, case.best));
    }

    let total = t_total.elapsed().as_secs_f64();
    let n = processed.max(1);
    println!("summary: cases={} elapsed={:.3}s avg_ms={:.1} nodes={} nps={:.1} fallback_hits={} start_d={} max_d={} threads={} opts={}",
        processed, total, (sum_secs / n as f64) * 1000.0, sum_nodes, if sum_secs>0.0 { sum_nodes as f64 / sum_secs } else { 0.0 }, fallback_hits, start_depth, max_depth, threads, std::env::var("PIEBOT_TEST_OPTS").unwrap_or_else(|_| "(default)".into()));
    if !failures.is_empty() {
        eprintln!("failures ({}):\n{}", failures.len(), failures.join("\n"));
        std::process::exit(1);
    }
}
