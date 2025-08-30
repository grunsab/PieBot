#[cfg(feature = "board-pleco")]
use clap::Parser;
#[cfg(feature = "board-pleco")]
use std::time::Instant;
#[cfg(feature = "board-pleco")]
use rand::{SeedableRng, Rng};
#[cfg(feature = "board-pleco")]
use rand::rngs::SmallRng;

#[cfg(feature = "board-pleco")]
#[derive(Parser, Debug)]
#[command(name = "piebot-bench-pleco", version, about = "Benchmark with Pleco board (make/unmake)")]
struct Args {
    #[arg(long, default_value = "startpos")] fen: String,
    #[arg(long, default_value_t = 4)] threads: usize,
    #[arg(long, default_value_t = 2000)] movetime: u64,
    #[arg(long, default_value_t = 6)] depth: u32,
    /// SMP mode: off | in-tree | lazy
    #[arg(long, default_value = "in-tree")]
    smp: String,
    /// SEE ordering on/off
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    see_ordering: bool,
    /// SEE pruning on/off
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    see_prune: bool,
    /// SEE ordering top-K captures (0=all)
    #[arg(long, default_value_t = 6usize)]
    see_topk: usize,
    /// Singular extensions on/off
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    singular: bool,
    /// Stronger IID on/off
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    iid_strong: bool,
    /// Evaluation mode: material | pst
    #[arg(long, default_value = "material")]
    eval_mode: String,
    /// Deterministic seed to randomize starting positions
    #[arg(long, default_value_t = 1u64)]
    seed: u64,
    /// Number of positions to benchmark (when >1, runs a suite)
    #[arg(long, default_value_t = 1usize)]
    positions: usize,
    /// Minimum random plies from base FEN
    #[arg(long, default_value_t = 8usize)]
    min_plies: usize,
    /// Maximum random plies from base FEN
    #[arg(long, default_value_t = 40usize)]
    max_plies: usize,
    /// Optional suite file with FEN per line or JSONL {"fen":...}
    #[arg(long)]
    suite: Option<String>,
    /// Emit JSON lines for results
    #[arg(long, default_value_t = false)]
    json: bool,
    /// Time manager policy: finish | spend
    #[arg(long, default_value = "finish")]
    tm_policy: String,
    /// Time manager factor (predict next iter time multiplier)
    #[arg(long, default_value_t = 1.9f32)]
    tm_factor: f32,
    /// Generate positions via engine rollout at this fixed depth (0=disabled)
    #[arg(long, default_value_t = 0u32)]
    rollout_depth: u32,
    /// Number of rollout best moves to generate positions from (when >0)
    #[arg(long, default_value_t = 0usize)]
    rollout_moves: usize,
    /// Number of initial tempered plies (choose among best moves via temperature)
    #[arg(long, default_value_t = 2usize)]
    rollout_tempered_plies: usize,
    /// Temperature in centipawns for tempered selection (softmax over cp/T)
    #[arg(long, default_value_t = 200.0f32)]
    rollout_temp_cp: f32,
    /// Limit to top-K moves for tempered sampling (0=all)
    #[arg(long, default_value_t = 3usize)]
    rollout_topk: usize,
    /// Compare eval modes (material vs pst) for NPS on the same cases
    #[arg(long, default_value_t = false)]
    compare_eval: bool,
}

#[cfg(feature = "board-pleco")]
fn main_inner() {
    let args = Args::parse();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads).build().unwrap();

    if args.compare_eval {
        return run_compare_eval(&pool, &args);
    }

    if args.positions <= 1 && args.suite.is_none() {
        let mut board = if args.fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&args.fen).expect("valid fen") };
        randomize_board(&mut board, args.seed, args.min_plies, args.max_plies);
        let t0 = Instant::now();
        let (bm, sc, nodes, depth_reached, seldepth) = pool.install(|| run_one(&mut board.clone(), &args));
        let dt = t0.elapsed();
        if args.json {
            println!("{{\"nodes\":{},\"depth\":{},\"seldepth\":{},\"nps\":{:.1},\"score_cp\":{},\"bestmove\":\"{:?}\"}}",
                nodes, depth_reached, seldepth, nodes as f64 / dt.as_secs_f64(), sc, bm);
        } else {
            println!("bestmove={:?} score_cp={} nodes={} depth={} seldepth={} elapsed={:.3}s nps={:.1}", bm, sc, nodes, depth_reached, seldepth, dt.as_secs_f64(), nodes as f64 / dt.as_secs_f64());
        }
        return;
    }

    // Multi-position run
    let mut fens: Vec<String> = Vec::new();
    if let Some(path) = &args.suite {
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                let l = line.trim(); if l.is_empty() { continue; }
                if l.starts_with('{') {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                        if let Some(f) = v.get("fen").and_then(|x| x.as_str()) { fens.push(f.to_string()); }
                    }
                } else { fens.push(l.to_string()); }
            }
        }
    } else { fens.push(args.fen.clone()); }

    // Deterministic shuffle by seed
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut idxs: Vec<usize> = (0..fens.len()).collect();
    for i in (1..idxs.len()).rev() { let j = rng.gen_range(0..=i); idxs.swap(i, j); }
    let mut picked = Vec::new();
    while picked.len() < args.positions && !idxs.is_empty() {
        for &i in &idxs { if picked.len() >= args.positions { break; } picked.push(fens[i].clone()); }
    }
    // Rollout mode: create positions by following engine best moves from a randomized start
    if args.rollout_depth > 0 && args.rollout_moves > 0 && args.suite.is_none() {
        let mut start = if args.fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&args.fen).expect("valid fen") };
        randomize_board(&mut start, args.seed, args.min_plies, args.max_plies);
        let cases = generate_rollout_positions(&start, args.rollout_moves, args.rollout_depth, args.rollout_tempered_plies, args.rollout_temp_cp, args.rollout_topk, args.seed, &args.eval_mode);
        let mut depths = Vec::with_capacity(cases.len());
        let mut seldepths = Vec::with_capacity(cases.len());
        let mut nodes_total: u64 = 0;
        let t0_all = Instant::now();
        for (i, mut board) in cases.into_iter().enumerate().take(args.positions) {
            let t0 = Instant::now();
            let (bm, sc, nodes, depth_reached, seldepth) = pool.install(|| run_one(&mut board, &args));
            let dt = t0.elapsed();
            depths.push(depth_reached); seldepths.push(seldepth); nodes_total += nodes;
            if args.json { println!("{{\"idx\":{},\"nodes\":{},\"depth\":{},\"seldepth\":{},\"nps\":{:.1},\"score_cp\":{},\"bestmove\":\"{:?}\"}}", i, nodes, depth_reached, seldepth, nodes as f64 / dt.as_secs_f64(), sc, bm); }
            else { println!("case={} depth={} seldepth={} nodes={} elapsed={:.3}s nps={:.1}", i, depth_reached, seldepth, nodes, dt.as_secs_f64(), nodes as f64 / dt.as_secs_f64()); }
        }
        let dt_all = t0_all.elapsed();
        let avg_depth = if depths.is_empty() { 0.0 } else { depths.iter().copied().sum::<u32>() as f64 / depths.len() as f64 };
        let avg_seldepth = if seldepths.is_empty() { 0.0 } else { seldepths.iter().copied().sum::<u32>() as f64 / seldepths.len() as f64 };
        let min_depth = depths.iter().copied().min().unwrap_or(0);
        let max_depth = depths.iter().copied().max().unwrap_or(0);
        if args.json { println!("{{\"summary\":true,\"positions\":{},\"avg_depth\":{:.2},\"avg_seldepth\":{:.2},\"min_depth\":{},\"max_depth\":{},\"total_nodes\":{},\"elapsed\":{:.3}}}", depths.len(), avg_depth, avg_seldepth, min_depth, max_depth, nodes_total, dt_all.as_secs_f64()); }
        else { println!("summary: positions={} avg_depth={:.2} avg_seldepth={:.2} min={} max={} total_nodes={} elapsed={:.3}s", depths.len(), avg_depth, avg_seldepth, min_depth, max_depth, nodes_total, dt_all.as_secs_f64()); }
        return;
    }

    let mut depths = Vec::with_capacity(picked.len());
    let mut seldepths = Vec::with_capacity(picked.len());
    let mut nodes_total: u64 = 0;
    let t0_all = Instant::now();
    for (i, fen) in picked.iter().enumerate() {
        let mut board = if fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&fen).expect("valid fen") };
        randomize_board(&mut board, args.seed.wrapping_add((i as u64).wrapping_mul(101_390_4223)), args.min_plies, args.max_plies);
        let t0 = Instant::now();
        let (bm, sc, nodes, depth_reached, seldepth) = pool.install(|| run_one(&mut board.clone(), &args));
        let dt = t0.elapsed();
        depths.push(depth_reached); seldepths.push(seldepth); nodes_total += nodes;
        if args.json {
            println!("{{\"idx\":{},\"nodes\":{},\"depth\":{},\"seldepth\":{},\"nps\":{:.1},\"score_cp\":{},\"bestmove\":\"{:?}\"}}",
                i, nodes, depth_reached, seldepth, nodes as f64 / dt.as_secs_f64(), sc, bm);
        } else {
            println!("case={} depth={} seldepth={} nodes={} elapsed={:.3}s nps={:.1}", i, depth_reached, seldepth, nodes, dt.as_secs_f64(), nodes as f64 / dt.as_secs_f64());
        }
    }
    let dt_all = t0_all.elapsed();
    let avg_depth = if depths.is_empty() { 0.0 } else { depths.iter().copied().sum::<u32>() as f64 / depths.len() as f64 };
    let avg_seldepth = if seldepths.is_empty() { 0.0 } else { seldepths.iter().copied().sum::<u32>() as f64 / seldepths.len() as f64 };
    let min_depth = depths.iter().copied().min().unwrap_or(0);
    let max_depth = depths.iter().copied().max().unwrap_or(0);
    if args.json { println!("{{\\\"summary\\\":true,\\\"positions\\\":{},\\\"avg_depth\\\":{:.2},\\\"avg_seldepth\\\":{:.2},\\\"min_depth\\\":{},\\\"max_depth\\\":{},\\\"total_nodes\\\":{},\\\"elapsed\\\":{:.3}}}", depths.len(), avg_depth, avg_seldepth, min_depth, max_depth, nodes_total, dt_all.as_secs_f64()); }
    else { println!("summary: positions={} avg_depth={:.2} avg_seldepth={:.2} min={} max={} total_nodes={} elapsed={:.3}s", depths.len(), avg_depth, avg_seldepth, min_depth, max_depth, nodes_total, dt_all.as_secs_f64()); }
}

#[cfg(not(feature = "board-pleco"))]
fn main() {
    eprintln!("bench_pleco requires --features board-pleco");
}

#[cfg(feature = "board-pleco")]
fn main() { main_inner(); }

#[cfg(feature = "board-pleco")]
fn randomize_board(board: &mut pleco::Board, seed: u64, min_plies: usize, max_plies: usize) {
    if max_plies == 0 || max_plies < min_plies { return; }
    let mut rng = SmallRng::seed_from_u64(seed);
    let span = (max_plies - min_plies + 1) as u32;
    let plies = min_plies + (if span == 0 { 0 } else { (rng.gen::<u32>() % span) as usize });
    for _ in 0..plies {
        let moves = board.generate_moves();
        let len = moves.len(); if len == 0 { break; }
        let idx = rng.gen_range(0..len);
        let mv = moves[idx];
        board.apply_move(mv);
        if board.checkmate() || board.stalemate() { break; }
    }
}

#[cfg(feature = "board-pleco")]
fn run_one(board: &mut pleco::Board, args: &Args) -> (Option<pleco::BitMove>, i32, u64, u32, u32) {
    use piebot::search::alphabeta_pleco::SmpMode;
    use piebot::search::alphabeta_pleco::PlecoEvalMode;
    let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
    s.set_threads(args.threads);
    s.set_see_ordering(args.see_ordering);
    s.set_see_prune(args.see_prune);
    s.set_see_ordering_topk(args.see_topk);
    s.set_singular_enable(args.singular);
    s.set_iid_strong(args.iid_strong);
    let eval_mode = match args.eval_mode.as_str() {
        "pst" => PlecoEvalMode::Pst,
        _ => PlecoEvalMode::Material,
    };
    s.set_eval_mode(eval_mode);
    let smp_mode = match args.smp.as_str() {
        "off" => SmpMode::Off,
        "in-tree" => SmpMode::InTree,
        "lazy-indep" => SmpMode::LazyIndep,
        "lazy-coop" => SmpMode::LazyCoop,
        "lazy-hybrid" => SmpMode::LazyHybrid,
        "lazy" => SmpMode::LazyCoop,
        _ => SmpMode::InTree,
    };
    s.set_smp_mode(smp_mode);
    let finish = match args.tm_policy.as_str() { "spend" => false, _ => true };
    s.set_time_manager(finish, args.tm_factor);
    let (bm, sc, nodes) = s.search_movetime(board, args.movetime, args.depth);
    (bm, sc, nodes, s.last_depth(), s.last_seldepth())
}

#[cfg(feature = "board-pleco")]
fn run_one_mode(board: &mut pleco::Board, args: &Args, eval_mode: &str) -> (Option<pleco::BitMove>, i32, u64, u32, u32) {
    use piebot::search::alphabeta_pleco::{SmpMode, PlecoEvalMode};
    let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
    s.set_threads(args.threads);
    s.set_see_ordering(args.see_ordering);
    s.set_see_prune(args.see_prune);
    s.set_see_ordering_topk(args.see_topk);
    s.set_singular_enable(args.singular);
    s.set_iid_strong(args.iid_strong);
    s.set_eval_mode(match eval_mode { "pst" => PlecoEvalMode::Pst, _ => PlecoEvalMode::Material });
    let smp_mode = match args.smp.as_str() {
        "off" => SmpMode::Off,
        "in-tree" => SmpMode::InTree,
        "lazy-indep" => SmpMode::LazyIndep,
        "lazy-coop" => SmpMode::LazyCoop,
        "lazy-hybrid" => SmpMode::LazyHybrid,
        "lazy" => SmpMode::LazyCoop,
        _ => SmpMode::InTree,
    };
    s.set_smp_mode(smp_mode);
    let finish = match args.tm_policy.as_str() { "spend" => false, _ => true };
    s.set_time_manager(finish, args.tm_factor);
    let (bm, sc, nodes) = s.search_movetime(board, args.movetime, args.depth);
    (bm, sc, nodes, s.last_depth(), s.last_seldepth())
}

#[cfg(feature = "board-pleco")]
fn run_compare_eval(pool: &rayon::ThreadPool, args: &Args) {
    // Build cases (suite or rollout or randomized from fen) once, then run both evals on same cases
    let mut cases: Vec<pleco::Board> = Vec::new();
    if args.rollout_depth > 0 && args.rollout_moves > 0 && args.suite.is_none() {
        let mut start = if args.fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&args.fen).expect("valid fen") };
        randomize_board(&mut start, args.seed, args.min_plies, args.max_plies);
        cases = generate_rollout_positions(&start, args.rollout_moves, args.rollout_depth, args.rollout_tempered_plies, args.rollout_temp_cp, args.rollout_topk, args.seed, &args.eval_mode);
    } else if let Some(path) = &args.suite {
        if let Ok(text) = std::fs::read_to_string(path) {
            for line in text.lines() {
                let l = line.trim(); if l.is_empty() { continue; }
                if l.starts_with('{') {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(l) {
                        if let Some(f) = v.get("fen").and_then(|x| x.as_str()) {
                            cases.push(pleco::Board::from_fen(f).expect("valid fen"));
                        }
                    }
                } else { cases.push(pleco::Board::from_fen(l).expect("valid fen")); }
            }
        }
    } else {
        // randomize from fen for positions count
        let mut base = if args.fen == "startpos" { pleco::Board::start_pos() } else { pleco::Board::from_fen(&args.fen).expect("valid fen") };
        randomize_board(&mut base, args.seed, args.min_plies, args.max_plies);
        cases.push(base);
    }
    // Ensure we don't exceed requested positions
    if cases.len() > args.positions { cases.truncate(args.positions); }

    // Run suite for a given eval mode, compute totals
    let mut run_suite = |mode: &str| -> (u64, f64, f64) {
        let t0_all = Instant::now();
        let mut nodes_total: u64 = 0;
        let mut depth_sum: u64 = 0;
        for mut b in cases.clone() {
            let t0 = Instant::now();
            let (_bm, _sc, nodes, depth_reached, _sd) = pool.install(|| run_one_mode(&mut b, args, mode));
            let _ = t0.elapsed();
            nodes_total += nodes;
            depth_sum += depth_reached as u64;
        }
        let dt = t0_all.elapsed().as_secs_f64();
        let avg_depth = if cases.is_empty() { 0.0 } else { depth_sum as f64 / cases.len() as f64 };
        (nodes_total, dt, avg_depth)
    };

    let (nodes_mat, dt_mat, avgd_mat) = run_suite("material");
    let (nodes_pst, dt_pst, avgd_pst) = run_suite("pst");
    let nps_mat = if dt_mat > 0.0 { nodes_mat as f64 / dt_mat } else { 0.0 };
    let nps_pst = if dt_pst > 0.0 { nodes_pst as f64 / dt_pst } else { 0.0 };
    println!("{{\"compare_eval\":true,\"positions\":{},\"material\":{{\"nodes\":{},\"elapsed\":{:.3},\"nps\":{:.1},\"avg_depth\":{:.2}}},\"pst\":{{\"nodes\":{},\"elapsed\":{:.3},\"nps\":{:.1},\"avg_depth\":{:.2}}}}}",
        cases.len(), nodes_mat, dt_mat, nps_mat, avgd_mat, nodes_pst, dt_pst, nps_pst, avgd_pst);
}

#[cfg(feature = "board-pleco")]
fn generate_rollout_positions(start: &pleco::Board, moves: usize, depth: u32, tempered_plies: usize, temp_cp: f32, topk: usize, seed: u64, eval_mode: &str) -> Vec<pleco::Board> {
    use piebot::search::alphabeta_pleco::SmpMode;
    use piebot::search::alphabeta_pleco::PlecoEvalMode;
    let mut boards = Vec::with_capacity(moves);
    let mut cur = start.clone();
    let mut rng = SmallRng::seed_from_u64(seed);
    for step in 0..moves {
        boards.push(cur.clone());
        // choose best move deterministically at fixed depth
        let mut w = piebot::search::alphabeta_pleco::PlecoSearcher::default();
        w.set_threads(1);
        w.set_smp_mode(SmpMode::Off);
        w.set_time_manager(true, 1.9);
        let em = match eval_mode { "pst" => PlecoEvalMode::Pst, _ => PlecoEvalMode::Material };
        w.set_eval_mode(em);
        // Compute root move scores at the specified rollout depth (parity with earlier selection)
        let mut board_clone = cur.clone();
        let scores = w.score_root_moves(&mut board_clone, depth);
        // Choose tempered for initial plies, then absolute best after
        let mv_opt = if step < tempered_plies {
            tempered_pick(&scores, temp_cp, topk, &mut rng)
        } else {
            scores.first().map(|(m, _)| *m)
        };
        if let Some(mv) = mv_opt { cur.apply_move(mv); } else {
            // no legal move; if we run out, break
            break;
        }
        if cur.checkmate() || cur.stalemate() { break; }
    }
    boards
}

#[cfg(feature = "board-pleco")]
fn tempered_pick(scores: &[(pleco::BitMove, i32)], temp_cp: f32, topk: usize, rng: &mut SmallRng) -> Option<pleco::BitMove> {
    if scores.is_empty() { return None; }
    let use_k = if topk == 0 { scores.len() } else { topk.min(scores.len()) };
    let slice = &scores[..use_k];
    if temp_cp <= 0.0 { return Some(slice[0].0); }
    let t = temp_cp;
    let max_sc = slice.iter().map(|&(_, sc)| sc).max().unwrap_or(0) as f32;
    let mut weights: Vec<f32> = slice.iter().map(|&(_, sc)| ((sc as f32 - max_sc)/t).exp()).collect();
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 { return Some(slice[0].0); }
    let mut r = rng.gen::<f32>() * sum;
    for (i, w) in weights.iter().enumerate() { r -= *w; if r <= 0.0 { return Some(slice[i].0); } }
    Some(slice[use_k-1].0)
}
