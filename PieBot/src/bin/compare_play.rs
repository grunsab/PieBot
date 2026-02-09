use clap::Parser;
use cozy_chess::{BitBoard, Color, Piece, Square};
use cozy_chess::{Board, Move};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "compare-play",
    about = "Play games: baseline (alphabeta) vs experimental (alphabeta_temp)"
)]
struct Args {
    /// Number of games to play
    #[arg(long, default_value_t = 40)]
    games: usize,

    /// Movetime per move in milliseconds
    #[arg(long, default_value_t = 200)]
    movetime: u64,

    /// Number of noisy plies at the start of each game (both sides)
    #[arg(long, default_value_t = 12)]
    noise_plies: usize,

    /// Top-K moves (by ordering) to sample from under noise
    #[arg(long, default_value_t = 5)]
    noise_topk: usize,

    /// Max plies before declaring a draw
    #[arg(long, default_value_t = 200)]
    max_plies: usize,

    /// Threads for each engine (1 recommended for reproducibility)
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Random seed
    #[arg(long, default_value_t = 1u64)]
    seed: u64,

    /// Force both sides to use baseline search implementation (model-only A/B).
    #[arg(long, default_value_t = false)]
    same_search: bool,

    /// Optional: write summary JSON to this path
    #[arg(long)]
    json_out: Option<String>,

    /// Optional: write summary CSV to this path
    #[arg(long)]
    csv_out: Option<String>,

    /// Optional: write all games to a single PGN file
    #[arg(long)]
    pgn_out: Option<String>,

    // Baseline config
    #[arg(long)]
    base_threads: Option<usize>,
    #[arg(long)]
    base_eval: Option<String>, // material|pst|nnue
    #[arg(long)]
    base_use_nnue: Option<bool>,
    #[arg(long)]
    base_blend: Option<u8>, // 0..100
    #[arg(long)]
    base_nnue_quant_file: Option<String>,
    #[arg(long)]
    base_nnue_file: Option<String>,
    #[arg(long)]
    base_hash_mb: Option<usize>,

    // Experimental config
    #[arg(long)]
    exp_threads: Option<usize>,
    #[arg(long)]
    exp_eval: Option<String>,
    #[arg(long)]
    exp_use_nnue: Option<bool>,
    #[arg(long)]
    exp_blend: Option<u8>,
    #[arg(long)]
    exp_nnue_quant_file: Option<String>,
    #[arg(long)]
    exp_nnue_file: Option<String>,
    #[arg(long)]
    exp_hash_mb: Option<usize>,
}

#[cfg(test)]
fn legal_moves(board: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    board.generate_moves(|ml| {
        for m in ml {
            v.push(m);
        }
        false
    });
    v
}

fn bb_contains(bb: BitBoard, target: Square) -> bool {
    for sq in bb {
        if sq == target {
            return true;
        }
    }
    false
}

fn piece_at(board: &Board, sq: Square) -> Option<(Color, Piece)> {
    for &color in &[Color::White, Color::Black] {
        let cb = board.colors(color);
        for &piece in &[
            Piece::Pawn,
            Piece::Knight,
            Piece::Bishop,
            Piece::Rook,
            Piece::Queen,
            Piece::King,
        ] {
            if bb_contains(cb & board.pieces(piece), sq) {
                return Some((color, piece));
            }
        }
    }
    None
}

fn is_capture_move(board: &Board, mv: Move) -> bool {
    piece_at(board, mv.to).is_some()
}

fn san_for_move(board: &Board, mv: Move) -> String {
    // Castling
    if let Some((_, Piece::King)) = piece_at(board, mv.from) {
        let from = format!("{}", mv.from);
        let to = format!("{}", mv.to);
        let ff = from.as_bytes()[0];
        let tf = to.as_bytes()[0];
        if (ff as i32 - tf as i32).abs() == 2 {
            return if tf > ff {
                "O-O".into()
            } else {
                "O-O-O".into()
            };
        }
    }
    let mut s = String::new();
    let (piece_char, is_pawn) = match piece_at(board, mv.from).map(|(_, p)| p) {
        Some(Piece::Knight) => ('N', false),
        Some(Piece::Bishop) => ('B', false),
        Some(Piece::Rook) => ('R', false),
        Some(Piece::Queen) => ('Q', false),
        Some(Piece::King) => ('K', false),
        _ => (' ', true),
    };
    let capture = is_capture_move(board, mv);
    // Disambiguation for non-pawn non-king moves
    let mut disamb_file = false;
    let mut disamb_rank = false;
    if !is_pawn && piece_char != 'K' {
        let mut candidates: Vec<Move> = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if m.to == mv.to && m != mv {
                    if let Some((_, p)) = piece_at(board, m.from) {
                        if !is_pawn && p.to_string().chars().next().unwrap_or(' ') == piece_char {
                            candidates.push(m);
                        }
                    }
                }
            }
            false
        });
        if !candidates.is_empty() {
            let from_file = format!("{}", mv.from).as_bytes()[0];
            let from_rank = format!("{}", mv.from).as_bytes()[1];
            let mut file_unique = true;
            let mut rank_unique = true;
            for m in &candidates {
                let f = format!("{}", m.from);
                if f.as_bytes()[0] == from_file {
                    file_unique = false;
                }
                if f.as_bytes()[1] == from_rank {
                    rank_unique = false;
                }
            }
            disamb_file = !file_unique && rank_unique;
            disamb_rank = file_unique && !rank_unique;
            if !disamb_file && !disamb_rank {
                disamb_file = true;
                disamb_rank = true;
            }
        }
    }
    if !is_pawn {
        s.push(piece_char);
    }
    if !is_pawn && (disamb_file || disamb_rank) {
        let from = format!("{}", mv.from);
        if disamb_file || (disamb_file && disamb_rank) {
            s.push(from.chars().next().unwrap());
        }
        if disamb_rank || (disamb_file && disamb_rank) {
            s.push(from.chars().nth(1).unwrap());
        }
    }
    if is_pawn && capture {
        // Pawn capture SAN starts with file of from
        let from = format!("{}", mv.from);
        s.push(from.chars().next().unwrap());
    }
    if capture {
        s.push('x');
    }
    s.push_str(&format!("{}", mv.to));
    // Promotion
    if let Some((_, Piece::Pawn)) = piece_at(board, mv.from) {
        if let Some(promo) = mv.promotion {
            let c = match promo {
                Piece::Knight => 'N',
                Piece::Bishop => 'B',
                Piece::Rook => 'R',
                Piece::Queen => 'Q',
                _ => 'Q',
            };
            s.push('=');
            s.push(c);
        }
    }
    // Check or checkmate
    let mut next = board.clone();
    next.play_unchecked(mv);
    let in_check = !(next.checkers()).is_empty();
    let mut has_legal = false;
    next.generate_moves(|_| {
        has_legal = true;
        true
    });
    if in_check {
        if !has_legal {
            s.push('#');
        } else {
            s.push('+');
        }
    }
    s
}

fn noisy_choice(order: &[Move], topk: usize, rng: &mut SmallRng) -> Option<Move> {
    if order.is_empty() {
        return None;
    }
    let k = topk.min(order.len()).max(1);
    let idx = rng.gen_range(0..k);
    Some(order[idx])
}

fn choose_move_noisy_baseline(
    board: &Board,
    engine: &BaselineEngine,
    topk: usize,
    rng: &mut SmallRng,
) -> Option<Move> {
    let order = engine.searcher.debug_order_root(board);
    noisy_choice(&order, topk, rng)
}

fn choose_move_noisy_experimental(
    board: &Board,
    engine: &ExperimentalEngine,
    topk: usize,
    rng: &mut SmallRng,
) -> Option<Move> {
    let order = match &engine.inner {
        ExperimentalEngineKind::Temp(s) => s.debug_order_root(board),
        ExperimentalEngineKind::Base(s) => s.debug_order_root(board),
    };
    noisy_choice(&order, topk, rng)
}

struct BaselineEngine {
    searcher: piebot::search::alphabeta::Searcher,
}

enum ExperimentalEngineKind {
    Temp(piebot::search::alphabeta_temp::Searcher),
    Base(piebot::search::alphabeta::Searcher),
}

struct ExperimentalEngine {
    inner: ExperimentalEngineKind,
}

fn parse_eval_mode_base(raw: Option<&str>) -> piebot::search::alphabeta::EvalMode {
    match raw.unwrap_or("pst").to_ascii_lowercase().as_str() {
        "material" => piebot::search::alphabeta::EvalMode::Material,
        "nnue" => piebot::search::alphabeta::EvalMode::Nnue,
        _ => piebot::search::alphabeta::EvalMode::Pst,
    }
}

fn parse_eval_mode_exp(raw: Option<&str>) -> piebot::search::alphabeta_temp::EvalMode {
    match raw.unwrap_or("pst").to_ascii_lowercase().as_str() {
        "material" => piebot::search::alphabeta_temp::EvalMode::Material,
        "nnue" => piebot::search::alphabeta_temp::EvalMode::Nnue,
        _ => piebot::search::alphabeta_temp::EvalMode::Pst,
    }
}

fn build_baseline_engine(args: &Args) -> BaselineEngine {
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_tt_capacity_mb(args.base_hash_mb.unwrap_or(64));
    s.set_threads(args.base_threads.unwrap_or(args.threads).max(1));
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_null_min_depth(8);
    s.set_use_aspiration(true);

    let mut mode = parse_eval_mode_base(args.base_eval.as_deref());
    if args.base_nnue_quant_file.is_some() || args.base_nnue_file.is_some() || args.base_use_nnue == Some(true) {
        mode = piebot::search::alphabeta::EvalMode::Nnue;
    }
    s.set_eval_mode(mode);
    if matches!(mode, piebot::search::alphabeta::EvalMode::Nnue) || args.base_use_nnue == Some(true) {
        s.set_use_nnue(true);
        s.set_eval_blend_percent(args.base_blend.unwrap_or(100));
        if let Some(path) = args.base_nnue_quant_file.as_deref() {
            let model = piebot::eval::nnue::loader::QuantNnue::load_quantized(path)
                .unwrap_or_else(|e| panic!("failed to load baseline quant NNUE {}: {}", path, e));
            s.set_nnue_quant_model(model);
        } else if let Some(path) = args.base_nnue_file.as_deref() {
            let nn = piebot::eval::nnue::Nnue::load(path)
                .unwrap_or_else(|e| panic!("failed to load baseline dense NNUE {}: {}", path, e));
            s.set_nnue_network(Some(nn));
        }
    }
    BaselineEngine { searcher: s }
}

fn build_experimental_engine(args: &Args) -> ExperimentalEngine {
    if args.same_search {
        let mut s = piebot::search::alphabeta::Searcher::default();
        s.set_tt_capacity_mb(args.exp_hash_mb.unwrap_or(64));
        s.set_threads(args.exp_threads.unwrap_or(args.threads).max(1));
        s.set_order_captures(true);
        s.set_use_history(true);
        s.set_use_killers(true);
        s.set_use_lmr(true);
        s.set_use_nullmove(true);
        s.set_null_min_depth(8);
        s.set_use_aspiration(true);

        let mut mode = parse_eval_mode_base(args.exp_eval.as_deref());
        if args.exp_nnue_quant_file.is_some()
            || args.exp_nnue_file.is_some()
            || args.exp_use_nnue == Some(true)
        {
            mode = piebot::search::alphabeta::EvalMode::Nnue;
        }
        s.set_eval_mode(mode);
        if matches!(mode, piebot::search::alphabeta::EvalMode::Nnue) || args.exp_use_nnue == Some(true)
        {
            s.set_use_nnue(true);
            s.set_eval_blend_percent(args.exp_blend.unwrap_or(100));
            if let Some(path) = args.exp_nnue_quant_file.as_deref() {
                let model = piebot::eval::nnue::loader::QuantNnue::load_quantized(path)
                    .unwrap_or_else(|e| panic!("failed to load experimental quant NNUE {}: {}", path, e));
                s.set_nnue_quant_model(model);
            } else if let Some(path) = args.exp_nnue_file.as_deref() {
                let nn = piebot::eval::nnue::Nnue::load(path)
                    .unwrap_or_else(|e| panic!("failed to load experimental dense NNUE {}: {}", path, e));
                s.set_nnue_network(Some(nn));
            }
        }
        return ExperimentalEngine {
            inner: ExperimentalEngineKind::Base(s),
        };
    }

    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    s.set_tt_capacity_mb(args.exp_hash_mb.unwrap_or(64));
    s.set_threads(args.exp_threads.unwrap_or(args.threads).max(1));
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_null_min_depth(8);
    s.set_use_aspiration(true);

    let mut mode = parse_eval_mode_exp(args.exp_eval.as_deref());
    if args.exp_nnue_quant_file.is_some() || args.exp_nnue_file.is_some() || args.exp_use_nnue == Some(true) {
        mode = piebot::search::alphabeta_temp::EvalMode::Nnue;
    }
    s.set_eval_mode(mode);
    if matches!(mode, piebot::search::alphabeta_temp::EvalMode::Nnue) || args.exp_use_nnue == Some(true) {
        s.set_use_nnue(true);
        s.set_eval_blend_percent(args.exp_blend.unwrap_or(100));
        if let Some(path) = args.exp_nnue_quant_file.as_deref() {
            let model = piebot::eval::nnue::loader::QuantNnue::load_quantized(path).unwrap_or_else(
                |e| panic!("failed to load experimental quant NNUE {}: {}", path, e),
            );
            s.set_nnue_quant_model(model);
        } else if let Some(path) = args.exp_nnue_file.as_deref() {
            let nn = piebot::eval::nnue::Nnue::load(path)
                .unwrap_or_else(|e| panic!("failed to load experimental dense NNUE {}: {}", path, e));
            s.set_nnue_network(Some(nn));
        }
    }
    ExperimentalEngine {
        inner: ExperimentalEngineKind::Temp(s),
    }
}

fn decide_move_baseline(
    board: &Board,
    movetime: u64,
    engine: &mut BaselineEngine,
) -> (Option<Move>, u32, u64, f64) {
    let t0 = Instant::now();
    let (bm, _sc, nodes) = engine.searcher.search_movetime(board, movetime, 0);
    let dt = t0.elapsed().as_secs_f64();
    let depth = engine.searcher.last_depth();
    (bm.and_then(|u| find_move_uci(board, u.as_str())), depth, nodes, dt)
}

fn decide_move_experimental(
    board: &Board,
    movetime: u64,
    engine: &mut ExperimentalEngine,
) -> (Option<Move>, u32, u64, f64) {
    match &mut engine.inner {
        ExperimentalEngineKind::Temp(s) => {
            let t0 = Instant::now();
            let (bm, _sc, nodes) = s.search_movetime(board, movetime, 0);
            let dt = t0.elapsed().as_secs_f64();
            let depth = s.last_depth();
            (bm.and_then(|u| find_move_uci(board, u.as_str())), depth, nodes, dt)
        }
        ExperimentalEngineKind::Base(s) => {
            let t0 = Instant::now();
            let (bm, _sc, nodes) = s.search_movetime(board, movetime, 0);
            let dt = t0.elapsed().as_secs_f64();
            let depth = s.last_depth();
            (bm.and_then(|u| find_move_uci(board, u.as_str())), depth, nodes, dt)
        }
    }
}

fn find_move_uci(board: &Board, uci: &str) -> Option<Move> {
    let mut found = None;
    board.generate_moves(|ml| {
        for m in ml {
            if format!("{}", m) == uci {
                found = Some(m);
                break;
            }
        }
        found.is_some()
    });
    found
}

fn is_game_over(board: &Board) -> Option<i32> {
    // Return Some(1) if side-to-move is checkmated (previous side wins)
    // Some(0) draw, Some(-1) if stalemate counts as draw too; use 0 for draw
    let mut has_legal = false;
    board.generate_moves(|_| {
        has_legal = true;
        true
    });
    if !has_legal {
        if !(board.checkers()).is_empty() {
            Some(1)
        } else {
            Some(0)
        }
    } else {
        None
    }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let mut rng = SmallRng::seed_from_u64(args.seed);

    // Detect if experimental search is identical to baseline (alphabeta_temp reexports alphabeta)
    let tn_base = std::any::type_name::<piebot::search::alphabeta::Searcher>();
    let tn_exp = std::any::type_name::<piebot::search::alphabeta_temp::Searcher>();
    let self_compare = args.same_search || tn_base == tn_exp;
    if args.same_search {
        eprintln!("[INFO] same-search mode enabled: both sides use baseline search implementation.");
    } else if tn_base == tn_exp {
        eprintln!("[WARN] Experimental search equals baseline (alphabeta_temp reexports alphabeta). Comparing baseline against itself.");
    }
    let mut base_engine = build_baseline_engine(&args);
    let mut exp_engine = build_experimental_engine(&args);

    let mut baseline_points = 0.0f64;
    let mut experimental_points = 0.0f64;
    let mut draws = 0usize;
    // Stats
    let mut sum_nodes_base: u64 = 0;
    let mut sum_time_base: f64 = 0.0;
    let mut sum_depth_base: u64 = 0;
    let mut cnt_base: u64 = 0;
    let mut sum_nodes_exp: u64 = 0;
    let mut sum_time_exp: f64 = 0.0;
    let mut sum_depth_exp: u64 = 0;
    let mut cnt_exp: u64 = 0;

    let mut pgn_buf = String::new();

    for g in 0..args.games {
        let mut board = Board::default();
        let baseline_is_white = g % 2 == 0;
        let mut plies = 0usize;
        let mut san_moves: Vec<String> = Vec::new();

        let result = loop {
            if let Some(res) = is_game_over(&board) {
                break match res {
                    1 => {
                        // side to move has no moves and is in check => previous mover won
                        let prev_was_baseline =
                            (plies > 0) && ((plies - 1) % 2 == 0) == baseline_is_white;
                        if prev_was_baseline {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    _ => 0.0,
                };
            }
            if plies >= args.max_plies {
                break 0.0;
            }

            let baseline_to_move = (plies % 2 == 0) == baseline_is_white;
            let mv = if plies < args.noise_plies {
                // Noisy selection from ordered top-K
                if baseline_to_move {
                    choose_move_noisy_baseline(&board, &base_engine, args.noise_topk, &mut rng)
                } else {
                    choose_move_noisy_experimental(&board, &exp_engine, args.noise_topk, &mut rng)
                }
            } else {
                if baseline_to_move {
                    let (m, d, n, dt) = decide_move_baseline(&board, args.movetime, &mut base_engine);
                    if let Some(_) = m {
                        sum_nodes_base += n;
                        sum_time_base += dt;
                        sum_depth_base += d as u64;
                        cnt_base += 1;
                    }
                    m
                } else {
                    let (m, d, n, dt) = decide_move_experimental(
                        &board,
                        args.movetime,
                        &mut exp_engine,
                    );
                    if let Some(_) = m {
                        sum_nodes_exp += n;
                        sum_time_exp += dt;
                        sum_depth_exp += d as u64;
                        cnt_exp += 1;
                    }
                    m
                }
            };

            let mv = match mv {
                Some(m) => m,
                None => {
                    break 0.0;
                }
            };
            // Record SAN before updating board
            let san = san_for_move(&board, mv);
            let mut next = board.clone();
            next.play_unchecked(mv);
            board = next;
            san_moves.push(san);
            plies += 1;
        };

        match result.partial_cmp(&0.0).unwrap() {
            std::cmp::Ordering::Greater => baseline_points += 1.0,
            std::cmp::Ordering::Less => experimental_points += 1.0,
            std::cmp::Ordering::Equal => draws += 1,
        }

        println!(
            "game={} result={} (baseline_white={}) plies={}",
            g + 1,
            result,
            baseline_is_white,
            plies
        );

        // Append PGN if requested
        if args.pgn_out.is_some() {
            let res = match result.partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Greater => {
                    if baseline_is_white {
                        "1-0"
                    } else {
                        "0-1"
                    }
                }
                std::cmp::Ordering::Less => {
                    if baseline_is_white {
                        "0-1"
                    } else {
                        "1-0"
                    }
                }
                std::cmp::Ordering::Equal => "1/2-1/2",
            };
            let white = if baseline_is_white {
                "Baseline"
            } else {
                "Experimental"
            };
            let black = if baseline_is_white {
                "Experimental"
            } else {
                "Baseline"
            };
            pgn_buf.push_str(&format!("[Event \"Cozy A/B\"]\n[Site \"Local\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n[TimeControl \"{}\"]\n\n",
                                     g + 1, white, black, res, args.movetime));
            // Moves with numbers
            let mut move_num = 1;
            for i in (0..san_moves.len()).step_by(2) {
                if i + 1 < san_moves.len() {
                    pgn_buf.push_str(&format!(
                        "{}. {} {} ",
                        move_num,
                        san_moves[i],
                        san_moves[i + 1]
                    ));
                } else {
                    pgn_buf.push_str(&format!("{}. {} ", move_num, san_moves[i]));
                }
                move_num += 1;
            }
            pgn_buf.push_str(&format!("{}\n\n", res));
        }
    }

    let avg_nps_base = if sum_time_base > 0.0 {
        sum_nodes_base as f64 / sum_time_base
    } else {
        0.0
    };
    let avg_nps_exp = if sum_time_exp > 0.0 {
        sum_nodes_exp as f64 / sum_time_exp
    } else {
        0.0
    };
    let avg_depth_base = if cnt_base > 0 {
        sum_depth_base as f64 / cnt_base as f64
    } else {
        0.0
    };
    let avg_depth_exp = if cnt_exp > 0 {
        sum_depth_exp as f64 / cnt_exp as f64
    } else {
        0.0
    };

    println!(
        "summary: games={} baseline_pts={} experimental_pts={} draws={}",
        args.games, baseline_points, experimental_points, draws
    );
    println!(
        "baseline: avg_nps={:.1} avg_depth={:.2} moves={} nodes={} time={:.3}s",
        avg_nps_base, avg_depth_base, cnt_base, sum_nodes_base, sum_time_base
    );
    println!(
        "experimental: avg_nps={:.1} avg_depth={:.2} moves={} nodes={} time={:.3}s",
        avg_nps_exp, avg_depth_exp, cnt_exp, sum_nodes_exp, sum_time_exp
    );

    // Optional machine-readable outputs
    if let Some(path) = args.json_out.as_deref() {
        let payload = serde_json::json!({
            "games": args.games,
            "movetime_ms": args.movetime,
            "noise_plies": args.noise_plies,
            "noise_topk": args.noise_topk,
            "threads": args.threads,
            "seed": args.seed,
            "self_compare": self_compare,
            "engines": {"baseline": tn_base, "experimental": tn_exp},
            "points": {"baseline": baseline_points, "experimental": experimental_points, "draws": draws},
            "baseline": {
                "moves": cnt_base, "nodes": sum_nodes_base, "time_s": sum_time_base,
                "avg_nps": avg_nps_base, "avg_depth": avg_depth_base
            },
            "experimental": {
                "moves": cnt_exp, "nodes": sum_nodes_exp, "time_s": sum_time_exp,
                "avg_nps": avg_nps_exp, "avg_depth": avg_depth_exp
            }
        });
        if let Err(e) = std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap()) {
            eprintln!("warn: failed to write json_out: {}", e);
        }
    }

    if let Some(path) = args.csv_out.as_deref() {
        // Single-row CSV summary with header
        let header = "games,movetime_ms,noise_plies,noise_topk,threads,seed,self_compare,base_type,exp_type,baseline_pts,experimental_pts,draws,base_moves,base_nodes,base_time_s,base_avg_nps,base_avg_depth,exp_moves,exp_nodes,exp_time_s,exp_avg_nps,exp_avg_depth\n";
        let row = format!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{},{},{},{:.6},{:.1},{:.2},{},{},{:.6},{:.1},{:.2}\n",
            args.games, args.movetime, args.noise_plies, args.noise_topk, args.threads, args.seed, self_compare, tn_base, tn_exp,
            baseline_points, experimental_points, draws,
            cnt_base, sum_nodes_base, sum_time_base, avg_nps_base, avg_depth_base,
            cnt_exp, sum_nodes_exp, sum_time_exp, avg_nps_exp, avg_depth_exp
        );
        let mut buf = String::new();
        buf.push_str(header);
        buf.push_str(&row);
        if let Err(e) = std::fs::write(path, buf) {
            eprintln!("warn: failed to write csv_out: {}", e);
        }
    }

    if let Some(path) = args.pgn_out.as_deref() {
        if let Err(e) = std::fs::write(path, pgn_buf) {
            eprintln!("warn: failed to write pgn_out: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_order_difference_position() -> Board {
        let candidates = [
            // Positions from mate-in-1 suite (guaranteed legal).
            "2kr1b1r/p1p2pp1/2pqN3/7p/6n1/2NPB3/PPP2PPP/R2Q1RK1 b - - 0 13",
            "6k1/p1p3pp/4N3/1p6/2q1r1n1/2B5/PP4PP/3R1R1K w - - 0 29",
            "8/3B2pp/p5k1/6P1/1ppp1K2/8/1P6/8 w - - 0 39",
            "r4rk1/pp3ppp/3b4/2p1pPB1/7N/2PP3n/PP4PP/R2Q2RK b - - 0 18",
        ];
        for fen in candidates {
            let board = Board::from_fen(fen, false).expect("valid FEN");
            let legal = legal_moves(&board);
            if legal.is_empty() {
                continue;
            }
            let mut searcher = piebot::search::alphabeta_temp::Searcher::default();
            searcher.set_order_captures(true);
            searcher.set_use_history(true);
            searcher.set_use_killers(true);
            let ordered = searcher.debug_order_root(&board);
            if !ordered.is_empty() && ordered[0] != legal[0] {
                return board;
            }
        }
        panic!("no candidate position showed a legal-order vs engine-order difference");
    }

    #[test]
    fn noisy_choice_topk_one_picks_first() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false).expect("valid FEN");
        let order = legal_moves(&board);
        let mut rng = SmallRng::seed_from_u64(1);
        let pick = noisy_choice(&order, 1, &mut rng).expect("pick");
        assert_eq!(pick, order[0]);
    }

    #[test]
    fn noisy_experimental_uses_engine_order_for_top1() {
        let board = first_order_difference_position();
        let mut searcher = piebot::search::alphabeta_temp::Searcher::default();
        searcher.set_order_captures(true);
        searcher.set_use_history(true);
        searcher.set_use_killers(true);
        let ordered = searcher.debug_order_root(&board);
        let expected = ordered[0];
        let engine = ExperimentalEngine {
            inner: ExperimentalEngineKind::Temp(searcher),
        };
        let mut rng = SmallRng::seed_from_u64(1);
        let picked = choose_move_noisy_experimental(&board, &engine, 1, &mut rng).expect("pick");
        assert_eq!(picked, expected);
    }

    #[test]
    fn noisy_baseline_uses_engine_order_for_top1() {
        let board = first_order_difference_position();
        let mut searcher = piebot::search::alphabeta::Searcher::default();
        searcher.set_order_captures(true);
        searcher.set_use_history(true);
        searcher.set_use_killers(true);
        let expected = searcher.debug_order_root(&board)[0];
        let engine = BaselineEngine { searcher };
        let mut rng = SmallRng::seed_from_u64(1);
        let picked = choose_move_noisy_baseline(&board, &engine, 1, &mut rng).expect("pick");
        assert_eq!(picked, expected);
    }
}
