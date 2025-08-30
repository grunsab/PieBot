use clap::Parser;
use cozy_chess::Board;
use piebot::search::alphabeta::{Searcher, SearchParams, EvalMode};
use std::time::Instant;
use rand::{SeedableRng, Rng};
use rand::rngs::SmallRng;

#[derive(Parser, Debug)]
#[command(name = "piebot-bench", version, about = "Benchmark PieBot search NPS with PST/NNUE")]
struct Args {
    /// FEN string or 'startpos'
    #[arg(long, default_value = "startpos")]
    fen: String,

    /// Threads
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Movetime in milliseconds (ignored if depth is set)
    #[arg(long, default_value_t = 1000)]
    movetime: u64,

    /// Fixed search depth (overrides movetime when > 0)
    #[arg(long, default_value_t = 0)]
    depth: u32,

    /// Use NNUE (requires NNUE file)
    #[arg(long, default_value_t = false)]
    use_nnue: bool,

    /// Dense NNUE file (PIENNUE1)
    #[arg(long)]
    nnue_file: Option<String>,

    /// Quantized NNUE file (PIENNQ01)
    #[arg(long)]
    nnue_quant_file: Option<String>,

    /// Eval blend percent (0..100). 0=PST, 100=NNUE
    #[arg(long, default_value_t = 100)]
    blend: u8,

    /// Transposition table size in MB (approximate)
    #[arg(long, default_value_t = 64)]
    hash_mb: usize,

    /// Eval mode: material | pst | nnue
    #[arg(long)]
    eval: Option<String>,

    /// Convenience: force Cozy to material-only (disables PST/NNUE)
    #[arg(long, default_value_t = false)]
    material_only: bool,

    /// Compare Cozy vs Pleco (requires --features board-pleco)
    #[arg(long, default_value_t = false)]
    compare: bool,

    /// Run a 20-position suite and report per-position + averages
    #[arg(long, default_value_t = false)]
    suite: bool,

    /// Deterministic seed for randomizing additional positions
    #[arg(long, default_value_t = 1u64)]
    seed: u64,

    /// Suite file path (name|fen per line, or JSONL {"name":...,"fen":...})
    #[arg(long)]
    suite_file: Option<String>,
}

#[cfg(feature = "board-pleco")]
fn run_pleco(fen: &str, threads: usize, depth: u32, movetime: Option<u64>, eval: &str) -> (Option<String>, i32, u64, f64, u32) {
    use piebot::search::alphabeta_pleco::{PlecoSearcher, SmpMode, PlecoEvalMode};
    let mut board = if fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(fen).expect("valid FEN") };
    let mut s = PlecoSearcher::default();
    s.set_threads(threads.max(1));
    s.set_smp_mode(SmpMode::InTree);
    // Use 'spend' policy for compare runs to utilize most of the movetime
    s.set_time_manager(false, 1.9);
    match eval.to_ascii_lowercase().as_str() {
        "material" => s.set_eval_mode(PlecoEvalMode::Material),
        "pst" => s.set_eval_mode(PlecoEvalMode::Pst),
        _ => s.set_eval_mode(PlecoEvalMode::Material),
    }
    let t0 = Instant::now();
    let (bm, sc, nodes) = {
        let ms = movetime.unwrap_or(10_000);
        s.search_movetime(&mut board, ms, depth)
    };
    let dt = t0.elapsed().as_secs_f64();
    let depth_reached = s.last_depth();
    (bm.map(|m| format!("{}", m)), sc, nodes, dt, depth_reached)
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let board = if args.fen == "startpos" { Board::default() } else { Board::from_fen(&args.fen, false).expect("valid FEN") };

    let mut s = Searcher::default();
    s.set_tt_capacity_mb(args.hash_mb);
    // Configure core toggles similarly to previous bench defaults
    s.set_threads(args.threads.max(1));
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_null_min_depth(8);
    s.set_hist_min_depth(10);
    s.set_root_see_top_k(6);
    s.set_use_aspiration(true);
    // Enable shallow pruning for benches
    s.set_use_futility(true);
    s.set_use_lmp(true);

    // Eval mode selection for Cozy
    let eval_mode = if args.material_only { EvalMode::Material } else if let Some(mode) = args.eval.as_deref() {
        match mode.to_ascii_lowercase().as_str() {
            "material" => EvalMode::Material,
            "pst" => EvalMode::Pst,
            "nnue" => EvalMode::Nnue,
            _ => EvalMode::Pst,
        }
    } else if args.use_nnue { EvalMode::Nnue } else { EvalMode::Pst };
    s.set_eval_mode(eval_mode);
    if matches!(eval_mode, EvalMode::Nnue) {
        s.set_use_nnue(true);
        s.set_eval_blend_percent(args.blend);
        if let Some(q) = args.nnue_quant_file.as_deref() {
            match piebot::eval::nnue::loader::QuantNnue::load_quantized(q) {
                Ok(model) => s.set_nnue_quant_model(model),
                Err(e) => eprintln!("failed to load quant NNUE: {}", e),
            }
        } else if let Some(d) = args.nnue_file.as_deref() {
            match piebot::eval::nnue::Nnue::load(d) {
                Ok(nn) => s.set_nnue_network(Some(nn)),
                Err(e) => eprintln!("failed to load dense NNUE: {}", e),
            }
        }
    }

    // Compare mode: run Cozy and Pleco and report NPS winner
    if args.compare && !args.suite {
        #[cfg(feature = "board-pleco")]
        {
            let eval_str = if let Some(e) = args.eval.as_deref() { e } else if args.use_nnue { "nnue" } else { "pst" };
            // Cozy run via search_movetime (uses Lazy-SMP internally when threads > 1)
            let t0 = Instant::now();
            let (bm_c, sc_c, nodes_c) = if args.threads > 1 {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
                pool.install(|| {
                    let (bm, sc, n) = s.search_movetime(&board, args.movetime, args.depth);
                    (bm.unwrap_or_else(|| "(none)".to_string()), sc, n)
                })
            } else {
                let (bm, sc, n) = s.search_movetime(&board, args.movetime, args.depth);
                (bm.unwrap_or_else(|| "(none)".to_string()), sc, n)
            };
            let dt_cozy = t0.elapsed().as_secs_f64();
            let depth_cozy = s.last_depth();
            let nps_cozy = if dt_cozy > 0.0 { nodes_c as f64 / dt_cozy } else { 0.0 };
            // Pleco run
            // Pass depth as-is (0 means unlimited via ID), so Pleco honors movetime runs
            let (bm_p, sc_p, nodes_p, dt_p, depth_pleco) = run_pleco(&args.fen, args.threads, args.depth, if args.depth > 0 { None } else { Some(args.movetime) }, eval_str);
            let nps_p = if dt_p > 0.0 { nodes_p as f64 / dt_p } else { 0.0 };
            println!("cozy:  bestmove={} score_cp={} nodes={} depth={} elapsed={:.3}s nps={:.1}", bm_c, sc_c, nodes_c, depth_cozy, dt_cozy, nps_cozy);
            println!("pleco: bestmove={} score_cp={} nodes={} depth={} elapsed={:.3}s nps={:.1}", bm_p.unwrap_or_else(|| "(none)".to_string()), sc_p, nodes_p, depth_pleco, dt_p, nps_p);
            if nps_cozy > nps_p { let diff = nps_cozy - nps_p; let pct = if nps_p > 0.0 { diff / nps_p * 100.0 } else { 0.0 }; println!("winner: cozy by +{:.1} nps (+{:.1}%)", diff, pct);
            } else if nps_p > nps_cozy { let diff = nps_p - nps_cozy; let pct = if nps_cozy > 0.0 { diff / nps_cozy * 100.0 } else { 0.0 }; println!("winner: pleco by +{:.1} nps (+{:.1}%)", diff, pct); } else { println!("tie: equal nps"); }
            return;
        }
        #[cfg(not(feature = "board-pleco"))]
        {
            eprintln!("Error: --compare requires building with --features board-pleco");
            std::process::exit(1);
        }
    }

    // Suite mode: build 20 positions (famous + randomized), run both engines, and report per-case + averages
    if args.compare && args.suite {
        #[cfg(feature = "board-pleco")]
        {
            let eval_str = if let Some(e) = args.eval.as_deref() { e } else if args.use_nnue { "nnue" } else { "pst" };
            // Suite positions: prefer external file when provided
            let mut suite: Vec<(String, String)> = Vec::new();
            if let Some(path) = &args.suite_file {
                if let Ok(text) = std::fs::read_to_string(path) {
                    for (lineno, line) in text.lines().enumerate() {
                        let l = line.trim();
                        if l.is_empty() || l.starts_with('#') { continue; }
                        if l.starts_with('{') {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                                // Accept {name,fen} or {fen} (optionally with other fields like 'best')
                                let fen_opt = v.get("fen").and_then(|x| x.as_str());
                                if let Some(fen) = fen_opt {
                                    let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("");
                                    let nm = if name.is_empty() { format!("pos{}", lineno + 1) } else { name.to_string() };
                                    suite.push((nm, fen.to_string()));
                                }
                            }
                        } else if let Some((name, fen)) = l.split_once('|') {
                            suite.push((name.trim().to_string(), fen.trim().to_string()));
                        } else {
                            // Try to treat the whole line as a FEN (6 fields) or bare placement (append side+meta)
                            let maybe_fen = l;
                            let fen_ok = if maybe_fen == "startpos" { true } else { cozy_chess::Board::from_fen(maybe_fen, false).is_ok() };
                            if fen_ok {
                                suite.push((format!("pos{}", lineno + 1), maybe_fen.to_string()));
                            } else {
                                // Attempt as bare placement with white to move
                                let fen_w = format!("{} w - - 0 1", maybe_fen);
                                if cozy_chess::Board::from_fen(&fen_w, false).is_ok() {
                                    suite.push((format!("pos{}", lineno + 1), fen_w));
                                } else {
                                    // Attempt as bare placement with black to move
                                    let fen_b = format!("{} b - - 0 1", maybe_fen);
                                    if cozy_chess::Board::from_fen(&fen_b, false).is_ok() {
                                        suite.push((format!("pos{}", lineno + 1), fen_b));
                                    } else {
                                        eprintln!("warn: {}:{}: unrecognized line; expected 'name|fen', JSONL, FEN, or placement-only", path, lineno+1);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("warn: failed to read suite file {}; falling back to defaults", path);
                }
            }
            if suite.is_empty() {
                // Fall back to two known positions and 18 randomized
                suite.push(("Startpos".to_string(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()));
                suite.push(("Kiwipete".to_string(), "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/P1N2N2/1P1BQPBP/R3K2R w KQkq - 0 1".to_string()));
            }
            // Generate additional randomized positions deterministically
            let mut rng = SmallRng::seed_from_u64(args.seed);
            while suite.len() < 20 {
                let mut b = Board::default();
                // random plies between 10..40
                let plies = 10 + (rng.gen::<u32>() % 31) as usize;
                for _ in 0..plies {
                    let mut moves: Vec<cozy_chess::Move> = Vec::new();
                    b.generate_moves(|ml| { for m in ml { moves.push(m); } false });
                    if moves.is_empty() { break; }
                    let idx = rng.gen_range(0..moves.len());
                    b.play(moves[idx]);
                }
                let fen = format!("{}", b);
                suite.push((format!("Rand{}", suite.len()+1), fen));
            }

            let mut rows = Vec::new();
            let mut sum_nps_cozy = 0.0;
            let mut sum_nps_pleco = 0.0;
            let mut sum_depth_cozy: u64 = 0;
            let mut sum_depth_pleco: u64 = 0;
            for (i, (name, fen)) in suite.iter().enumerate() {
                // Cozy per-position
                let board = if fen == "startpos" { Board::default() } else { Board::from_fen(fen, false).expect("valid FEN") };
                let mut sc = Searcher::default();
                sc.set_tt_capacity_mb(args.hash_mb);
                sc.set_threads(args.threads.max(1));
                sc.set_order_captures(true);
                sc.set_use_history(true);
                sc.set_use_killers(true);
                sc.set_use_lmr(true);
                sc.set_use_nullmove(true);
                sc.set_null_min_depth(8);
                sc.set_hist_min_depth(10);
                sc.set_root_see_top_k(6);
                sc.set_use_futility(true);
                sc.set_use_lmp(true);
                sc.set_use_aspiration(true);
                sc.set_eval_mode(eval_mode);
                if matches!(eval_mode, EvalMode::Nnue) {
                    sc.set_use_nnue(true);
                    if let Some(q) = args.nnue_quant_file.as_deref() { if let Ok(model) = piebot::eval::nnue::loader::QuantNnue::load_quantized(q) { sc.set_nnue_quant_model(model); } }
                    if let Some(d) = args.nnue_file.as_deref() { if let Ok(nn) = piebot::eval::nnue::Nnue::load(d) { sc.set_nnue_network(Some(nn)); } }
                }
                let t0 = Instant::now();
                let (bm_c, sc_c, nodes_c, depth_c) = if args.threads > 1 {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
                    pool.install(|| sc.search_movetime_lazy_smp(&board, args.movetime, args.depth))
                } else { let (bm, sc_score, n) = sc.search_movetime(&board, args.movetime, args.depth); (bm, sc_score, n, sc.last_depth()) };
                let dt_c = t0.elapsed().as_secs_f64();
                let nps_c = if dt_c > 0.0 { nodes_c as f64 / dt_c } else { 0.0 };

                // Pleco per-position
                let (bm_p, sc_p, nodes_p, dt_p, depth_p) = run_pleco(fen, args.threads, args.depth, if args.depth > 0 { None } else { Some(args.movetime) }, eval_str);
                let nps_p = if dt_p > 0.0 { nodes_p as f64 / dt_p } else { 0.0 };

                sum_nps_cozy += nps_c; sum_nps_pleco += nps_p;
                sum_depth_cozy += depth_c as u64; sum_depth_pleco += depth_p as u64;
                rows.push((i, name.clone(), depth_c, nps_c, depth_p, nps_p));
            }

            // Print per-position results
            for (i, name, d_c, nps_c, d_p, nps_p) in &rows {
                println!("case={} name={} cozy_depth={} cozy_nps={:.1} pleco_depth={} pleco_nps={:.1}", i, name, d_c, nps_c, d_p, nps_p);
            }
            let n = rows.len().max(1) as f64;
            let avg_depth_c = sum_depth_cozy as f64 / n;
            let avg_depth_p = sum_depth_pleco as f64 / n;
            let avg_nps_c = sum_nps_cozy / n;
            let avg_nps_p = sum_nps_pleco / n;
            println!("summary: positions={} avg_cozy_depth={:.2} avg_pleco_depth={:.2} avg_cozy_nps={:.1} avg_pleco_nps={:.1}", rows.len(), avg_depth_c, avg_depth_p, avg_nps_c, avg_nps_p);
            if avg_nps_c > avg_nps_p { println!("winner_nps: cozy by +{:.1} nps (+{:.1}%)", avg_nps_c - avg_nps_p, if avg_nps_p>0.0 {(avg_nps_c-avg_nps_p)/avg_nps_p*100.0} else {0.0}); }
            else if avg_nps_p > avg_nps_c { println!("winner_nps: pleco by +{:.1} nps (+{:.1}%)", avg_nps_p - avg_nps_c, if avg_nps_c>0.0 {(avg_nps_p-avg_nps_c)/avg_nps_c*100.0} else {0.0}); }
            if avg_depth_c > avg_depth_p { println!("winner_depth: cozy by +{:.2}", avg_depth_c - avg_depth_p); } else if avg_depth_p > avg_depth_c { println!("winner_depth: pleco by +{:.2}", avg_depth_p - avg_depth_c); }
            return;
        }
        #[cfg(not(feature = "board-pleco"))]
        {
            eprintln!("Error: --compare --suite requires --features board-pleco");
            std::process::exit(1);
        }
    }

    let t0 = Instant::now();
    let (bm, sc, nodes, dt) = if args.threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();
        pool.install(|| { let (bm, sc, n) = s.search_movetime(&board, args.movetime, args.depth); (bm, sc, n, t0.elapsed()) })
    } else { let (bm, sc, n) = s.search_movetime(&board, args.movetime, args.depth); (bm, sc, n, t0.elapsed()) };
    let nps = if dt.as_secs_f64() > 0.0 { nodes as f64 / dt.as_secs_f64() } else { 0.0 };
    println!("bestmove={} score_cp={} nodes={} elapsed={:.3}s nps={:.1}", bm.unwrap_or_else(|| "(none)".to_string()), sc, nodes, dt.as_secs_f64(), nps);
}
