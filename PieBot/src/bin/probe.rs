//! Position probe: one fresh fixed-depth search per depth, so each line is
//! exactly what `go depth N` returns, followed by a principal variation read
//! back by re-searching along the best line.
//!
//! Diagnostic tool for studying single positions with either search
//! implementation; it measures nothing about playing strength.
use clap::Parser;
use cozy_chess::Board;
use piebot::eval::nnue::loader::QuantNnue;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Probe one position depth by depth with baseline or experimental search")]
struct Args {
    /// Root position.
    #[arg(long, default_value = "")]
    fen: String,
    /// Batch mode: one FEN per line, each searched once at --max-depth with a
    /// fresh searcher and no history; prints `score best nodes` per line.
    #[arg(long)]
    fen_file: Option<String>,
    /// Quantized NNUE file.
    #[arg(long)]
    net: String,
    /// `base` (alphabeta) or `temp` (alphabeta_temp).
    #[arg(long, default_value = "base")]
    engine: String,
    /// File with one FEN per line: the game positions before the root.
    #[arg(long)]
    history: Option<String>,
    #[arg(long, default_value_t = 1)]
    min_depth: u32,
    #[arg(long, default_value_t = 16)]
    max_depth: u32,
    /// Stop deepening once one search has used this many nodes.
    #[arg(long)]
    node_cap: Option<u64>,
    /// Plies of principal variation to read back at the last depth.
    #[arg(long, default_value_t = 10)]
    pv_len: usize,
    #[arg(long, default_value_t = 256)]
    hash_mb: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long)]
    no_lmr: bool,
    /// Also disables the pruning rules that share the null-move switch.
    #[arg(long)]
    no_null: bool,
    #[arg(long)]
    no_aspiration: bool,
}

/// The legal move whose engine spelling (cozy-chess notation) is `name`.
fn legal_move_named(board: &Board, name: &str) -> Option<cozy_chess::Move> {
    let mut found = None;
    board.generate_moves(|moves| {
        for mv in moves {
            if format!("{mv}") == name {
                found = Some(mv);
                return true;
            }
        }
        false
    });
    found
}

macro_rules! probe_impl {
    ($name:ident, $module:path) => {
        fn $name(args: &Args, model: &QuantNnue, root: &Board, history: &[Board]) {
            use $module as engine;
            if let Some(path) = &args.fen_file {
                let text = std::fs::read_to_string(path).expect("read fen file");
                for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    let Ok(board) = Board::from_fen(line, false) else {
                        println!("invalid");
                        continue;
                    };
                    let mut s = engine::Searcher::default();
                    s.set_tt_capacity_mb(args.hash_mb);
                    s.set_use_nnue(true);
                    s.set_eval_mode(engine::EvalMode::Nnue);
                    s.set_eval_blend_percent(100);
                    s.set_nnue_quant_model(model.clone());
                    let mut p = engine::SearchParams::default();
                    p.depth = args.max_depth;
                    p.use_tt = true;
                    p.order_captures = true;
                    p.use_history = true;
                    p.threads = 1;
                    p.use_aspiration = !args.no_aspiration;
                    p.aspiration_window_cp = 35;
                    p.use_lmr = !args.no_lmr;
                    p.use_killers = true;
                    p.use_nullmove = !args.no_null;
                    p.deterministic = true;
                    let r = s.search_with_params(&board, p);
                    println!(
                        "{} {} {}",
                        r.score_cp,
                        r.bestmove.unwrap_or_default(),
                        r.nodes
                    );
                }
                return;
            }
            let params = |depth: u32| {
                let mut p = engine::SearchParams::default();
                p.depth = depth;
                p.use_tt = true;
                p.order_captures = true;
                p.use_history = true;
                p.threads = args.threads.max(1);
                p.use_aspiration = !args.no_aspiration;
                p.aspiration_window_cp = 35;
                p.use_lmr = !args.no_lmr;
                p.use_killers = true;
                p.use_nullmove = !args.no_null;
                p.deterministic = args.threads <= 1;
                p
            };
            let fresh = || {
                let mut s = engine::Searcher::default();
                s.set_tt_capacity_mb(args.hash_mb);
                s.set_use_nnue(true);
                s.set_eval_mode(engine::EvalMode::Nnue);
                s.set_eval_blend_percent(100);
                s.set_nnue_quant_model(model.clone());
                s.set_threads(args.threads.max(1));
                s
            };
            let mut last = None;
            for depth in args.min_depth..=args.max_depth {
                let mut s = fresh();
                s.set_position_history(history);
                let t0 = Instant::now();
                let r = s.search_with_params(root, params(depth));
                let secs = t0.elapsed().as_secs_f64();
                println!(
                    "depth {:>2} seldepth {:>2} score {:>6} best {:<6} nodes {:>11} time {:>7.2}s",
                    depth,
                    s.last_seldepth(),
                    r.score_cp,
                    r.bestmove.clone().unwrap_or_default(),
                    r.nodes,
                    secs
                );
                let over = args.node_cap.map_or(false, |cap| r.nodes >= cap);
                last = Some((s, depth));
                if over {
                    break;
                }
            }
            let Some((mut s, depth)) = last else { return };
            let mut board = root.clone();
            let mut line = Vec::new();
            let mut hist: Vec<Board> = history.to_vec();
            for step in 0..args.pv_len {
                let remaining = depth.saturating_sub(step as u32);
                if remaining == 0 {
                    break;
                }
                s.set_position_history(&hist);
                let r = s.search_with_params(&board, params(remaining));
                let Some(best) = r.bestmove else { break };
                let Some(mv) = legal_move_named(&board, &best) else {
                    break;
                };
                line.push(format!("{best}({})", r.score_cp));
                hist.push(board.clone());
                board.play_unchecked(mv);
            }
            println!("pv {}", line.join(" "));
            println!("pv_end_fen {board}");
        }
    };
}

probe_impl!(probe_base, piebot::search::alphabeta);
probe_impl!(probe_temp, piebot::search::alphabeta_temp);

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model = QuantNnue::load_quantized(&args.net)?;
    let root = if args.fen.is_empty() {
        Board::default()
    } else {
        Board::from_fen(&args.fen, false).map_err(|e| anyhow::anyhow!("bad fen: {e:?}"))?
    };
    let mut history = Vec::new();
    if let Some(path) = &args.history {
        for line in std::fs::read_to_string(path)?.lines() {
            let line = line.trim();
            if !line.is_empty() {
                history.push(
                    Board::from_fen(line, false).map_err(|e| anyhow::anyhow!("bad fen: {e:?}"))?,
                );
            }
        }
    }
    match args.engine.as_str() {
        "base" => probe_base(&args, &model, &root, &history),
        "temp" => probe_temp(&args, &model, &root, &history),
        other => anyhow::bail!("unknown engine {other}"),
    }
    Ok(())
}
