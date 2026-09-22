use crate::eval::nnue::loader::QuantNnue;
use crate::search::alphabeta::{EvalMode, SearchParams, Searcher};
use crate::search::zobrist;
use cozy_chess::{Board, Color, GameStatus, Move};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct SelfPlayParams {
    pub games: usize,
    pub max_plies: usize,
    pub threads: usize,        // engine search threads per game
    pub parallel_games: usize, // 0 => auto from available cores
    pub use_engine: bool,
    pub depth: u32,
    pub movetime_ms: Option<u64>,
    pub seed: u64,
    pub temperature_tau: f32,           // softmax temperature; 0 => greedy
    pub temp_cp_scale: f32,             // scale cp to logits
    pub dirichlet_alpha: f32,           // alpha for Dirichlet
    pub dirichlet_epsilon: f32,         // mixing coefficient
    pub dirichlet_plies: usize,         // apply Dirichlet noise for first N plies
    pub temperature_moves: usize,       // apply temperature for first N plies
    pub openings_path: Option<PathBuf>, // optional path to FEN list (one per line)
    pub temperature_tau_final: f32,     // anneal temperature to this by temperature_moves
    pub nnue_quant_model: Option<QuantNnue>,
    pub nnue_blend_percent: u8, // 0..100, only used when nnue_quant_model is Some
    pub resign_cp: f32,         // white-perspective cp threshold; 0 disables resignation
    pub resign_plies: usize,    // consecutive plies past the threshold before resigning
    pub no_resign_fraction: f32, // fraction of games with resignation disabled (0..=1)
    pub draw_adj_cp: f32,       // |cp| threshold for draw adjudication; 0 disables
    pub draw_adj_plies: usize,  // consecutive quiet plies before adjudicating a draw
    pub draw_adj_min_ply: usize, // earliest ply index a draw adjudication may fire
    pub actor_tt_mb: usize,      // 0 = legacy 4096-entry table; >0 = real TT in MB
    pub policy_node_cap: u64,    // per-move policy-scoring node budget
    pub bestmove_node_cap: u64,  // best-move search node budget
}

/// Search budget for per-move policy scoring; extracted so the actor's
/// node caps are configuration, not constants buried in the game loop.
pub fn policy_search_params(params: &SelfPlayParams, depth: u32) -> SearchParams {
    let mut p = base_actor_search_params(params, depth);
    p.max_nodes = Some(params.policy_node_cap.max(1));
    p
}

/// Search budget for the clean root label, also used to play outside noise.
pub fn bestmove_search_params(params: &SelfPlayParams, depth: u32) -> SearchParams {
    let mut p = base_actor_search_params(params, depth);
    p.max_nodes = Some(params.bestmove_node_cap.max(1));
    p
}

fn base_actor_search_params(params: &SelfPlayParams, depth: u32) -> SearchParams {
    let mut p = SearchParams::default();
    p.depth = depth;
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = params.threads;
    p.use_aspiration = true;
    p.aspiration_window_cp = 50;
    p.use_lmr = true;
    p.use_killers = true;
    p.use_nullmove = true;
    p.movetime = params
        .movetime_ms
        .map(std::time::Duration::from_millis);
    p
}

pub struct GameRecord {
    pub run_id: String,  // stable identifier shared by all games for one seeded run
    pub game_id: String, // stable identifier derived from run_id + game index
    pub start_fen: String,
    pub moves: Vec<String>,                       // played moves
    pub move_target_best: Vec<Option<String>>,    // teacher best move for each ply
    pub move_value_cp: Vec<Option<f32>>,          // white-perspective teacher value for each ply
    pub move_teacher_depth: Vec<Option<u32>>,     // deepest completed root iteration
    pub move_policy_top: Vec<Vec<(String, f32)>>, // optional root policy samples
    pub result: i8,                               // 1 white win, 0 draw, -1 black win
    pub outcome_valid: bool, // false when generation stopped without a chess result
    pub termination: GameTermination,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameTermination {
    Checkmate,
    Stalemate,
    FiftyMove,
    InsufficientMaterial,
    ThreefoldRepetition,
    MaxPlies,
    NoMove,
    Resigned,
    AdjudicatedDraw,
}

impl GameTermination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Checkmate => "checkmate",
            Self::Stalemate => "stalemate",
            Self::FiftyMove => "fifty_move",
            Self::InsufficientMaterial => "insufficient_material",
            Self::ThreefoldRepetition => "threefold_repetition",
            Self::MaxPlies => "max_plies",
            Self::NoMove => "no_move",
            Self::Resigned => "resigned",
            Self::AdjudicatedDraw => "adjudicated_draw",
        }
    }
}

/// Verdict produced by the ply-by-ply adjudication state machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjudicationVerdict {
    ResignWhiteWins,
    ResignBlackWins,
    AdjudicatedDraw,
}

/// Pure resign/draw adjudication state machine fed white-perspective evals.
///
/// A cp threshold of 0 disables the corresponding rule.
pub struct Adjudicator {
    resign_cp: f32,
    resign_plies: usize,
    draw_adj_cp: f32,
    draw_adj_plies: usize,
    draw_adj_min_ply: usize,
    white_resign_streak: usize,
    black_resign_streak: usize,
    draw_streak: usize,
}

impl Adjudicator {
    pub fn new(
        resign_cp: f32,
        resign_plies: usize,
        draw_adj_cp: f32,
        draw_adj_plies: usize,
        draw_adj_min_ply: usize,
    ) -> Self {
        Self {
            resign_cp,
            resign_plies,
            draw_adj_cp,
            draw_adj_plies,
            draw_adj_min_ply,
            white_resign_streak: 0,
            black_resign_streak: 0,
            draw_streak: 0,
        }
    }

    pub fn observe(&mut self, ply: usize, white_cp: Option<f32>) -> Option<AdjudicationVerdict> {
        let cp = match white_cp {
            Some(cp) => cp,
            None => {
                self.white_resign_streak = 0;
                self.black_resign_streak = 0;
                self.draw_streak = 0;
                return None;
            }
        };
        if self.resign_cp > 0.0 && self.resign_plies > 0 {
            if cp >= self.resign_cp {
                self.white_resign_streak += 1;
                self.black_resign_streak = 0;
            } else if cp <= -self.resign_cp {
                self.black_resign_streak += 1;
                self.white_resign_streak = 0;
            } else {
                self.white_resign_streak = 0;
                self.black_resign_streak = 0;
            }
            if self.white_resign_streak >= self.resign_plies {
                return Some(AdjudicationVerdict::ResignWhiteWins);
            }
            if self.black_resign_streak >= self.resign_plies {
                return Some(AdjudicationVerdict::ResignBlackWins);
            }
        }
        if self.draw_adj_cp > 0.0 && self.draw_adj_plies > 0 {
            if cp.abs() <= self.draw_adj_cp {
                self.draw_streak += 1;
            } else {
                self.draw_streak = 0;
            }
            if self.draw_streak >= self.draw_adj_plies && ply >= self.draw_adj_min_ply {
                return Some(AdjudicationVerdict::AdjudicatedDraw);
            }
        }
        None
    }
}

struct MoveChoice {
    played_mv: Move,
    target_best_mv: Option<Move>,
    value_cp: Option<f32>,
    teacher_depth: Option<u32>,
    policy_top: Vec<(String, f32)>,
}

pub fn generate_games(params: &SelfPlayParams) -> Result<Vec<GameRecord>, String> {
    let openings = load_openings(params)?;
    if params.games == 0 {
        return Ok(Vec::new());
    }

    let parallel_games = effective_parallel_games(params);
    if parallel_games <= 1 || params.games <= 1 {
        return Ok((0..params.games)
            .map(|game_idx| generate_single_game(params, &openings, game_idx))
            .collect());
    }

    let games = match rayon::ThreadPoolBuilder::new()
        .num_threads(parallel_games)
        .build()
    {
        Ok(pool) => pool.install(|| {
            (0..params.games)
                .into_par_iter()
                .map(|game_idx| generate_single_game(params, &openings, game_idx))
                .collect()
        }),
        Err(_) => (0..params.games)
            .map(|game_idx| generate_single_game(params, &openings, game_idx))
            .collect(),
    };
    Ok(games)
}

pub fn effective_parallel_games(params: &SelfPlayParams) -> usize {
    if params.games == 0 {
        return 1;
    }
    if params.parallel_games > 0 {
        return params.parallel_games.max(1).min(params.games);
    }
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let per_game_threads = params.threads.max(1);
    let auto_games = (available / per_game_threads).max(1);
    auto_games.min(params.games)
}

fn generate_single_game(
    params: &SelfPlayParams,
    openings: &[Board],
    game_idx: usize,
) -> GameRecord {
    let game_seed = game_seed(params.seed, game_idx);
    let run_id = run_id(params.seed);
    let mut rng = SmallRng::seed_from_u64(game_seed);
    let mut searcher = if params.use_engine {
        Some(build_selfplay_searcher(params))
    } else {
        None
    };
    let mut board = if !openings.is_empty() {
        let idx = (mix_u64(game_seed ^ 0xA5A5_A5A5_A5A5_A5A5) as usize) % openings.len();
        openings[idx].clone()
    } else {
        Board::default()
    };
    let mut record = GameRecord {
        game_id: game_id(&run_id, game_idx),
        run_id,
        start_fen: format!("{}", board),
        moves: Vec::new(),
        move_target_best: Vec::new(),
        move_value_cp: Vec::new(),
        move_teacher_depth: Vec::new(),
        move_policy_top: Vec::new(),
        result: 0,
        outcome_valid: false,
        termination: GameTermination::MaxPlies,
    };
    let resign_allowed = !resign_disabled_for_game(params.no_resign_fraction, game_seed);
    let mut adjudicator = Adjudicator::new(
        if resign_allowed { params.resign_cp } else { 0.0 },
        params.resign_plies,
        params.draw_adj_cp,
        params.draw_adj_plies,
        params.draw_adj_min_ply,
    );
    let mut position_history = vec![board.clone()];
    let mut plies = 0usize;
    loop {
        if let Some((result, termination)) = adjudicate_position(&board, &position_history) {
            record.result = result;
            record.outcome_valid = true;
            record.termination = termination;
            break;
        }
        if plies >= params.max_plies {
            break;
        }
        // choose move
        let mv = if params.use_engine {
            searcher
                .as_mut()
                .expect("engine searcher available")
                .set_position_history(&position_history);
            select_engine_move(
                &board,
                params,
                plies,
                game_seed,
                searcher.as_mut().expect("engine searcher available"),
            )
        } else {
            select_random_move(&board, &mut rng)
        };
        if let Some(m) = mv {
            let value_cp = m.value_cp;
            let mstr = format!("{}", m.played_mv);
            record.moves.push(mstr);
            record
                .move_target_best
                .push(m.target_best_mv.map(|x| format!("{}", x)));
            record.move_value_cp.push(m.value_cp);
            record.move_teacher_depth.push(m.teacher_depth);
            record.move_policy_top.push(m.policy_top);
            board.play_unchecked(m.played_mv);
            position_history.push(board.clone());
            plies += 1;
            if let Some(verdict) = adjudicator.observe(plies - 1, value_cp) {
                let (result, termination) = match verdict {
                    AdjudicationVerdict::ResignWhiteWins => (1, GameTermination::Resigned),
                    AdjudicationVerdict::ResignBlackWins => (-1, GameTermination::Resigned),
                    AdjudicationVerdict::AdjudicatedDraw => (0, GameTermination::AdjudicatedDraw),
                };
                record.result = result;
                record.outcome_valid = true;
                record.termination = termination;
                break;
            }
        } else {
            record.termination = GameTermination::NoMove;
            break;
        }
    }
    record
}

pub fn adjudicate_position(
    board: &Board,
    position_history: &[Board],
) -> Option<(i8, GameTermination)> {
    match board.status() {
        GameStatus::Won => {
            let result = if board.side_to_move() == Color::White {
                -1
            } else {
                1
            };
            return Some((result, GameTermination::Checkmate));
        }
        GameStatus::Drawn => {
            let termination = if board.halfmove_clock() >= 100 {
                GameTermination::FiftyMove
            } else {
                GameTermination::Stalemate
            };
            return Some((0, termination));
        }
        GameStatus::Ongoing => {}
    }

    if has_insufficient_material(board) {
        return Some((0, GameTermination::InsufficientMaterial));
    }
    if position_history
        .iter()
        .filter(|previous| previous.same_position(board))
        .take(3)
        .count()
        >= 3
    {
        return Some((0, GameTermination::ThreefoldRepetition));
    }
    None
}

fn has_insufficient_material(board: &Board) -> bool {
    crate::search::draw::is_insufficient_material(board)
}

fn game_seed(base_seed: u64, game_idx: usize) -> u64 {
    mix_u64(base_seed ^ ((game_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

fn resign_disabled_for_game(no_resign_fraction: f32, game_seed: u64) -> bool {
    if no_resign_fraction <= 0.0 {
        return false;
    }
    if no_resign_fraction >= 1.0 {
        return true;
    }
    const NO_RESIGN_DOMAIN: u64 = 0x4E4F_5245_5349_474E; // "NORESIGN"
    let bucket = mix_u64(game_seed ^ NO_RESIGN_DOMAIN) % 10_000;
    (bucket as f32) < no_resign_fraction * 10_000.0
}

fn run_id(base_seed: u64) -> String {
    const RUN_ID_DOMAIN: u64 = 0x5049_4542_4f54_5255;
    format!("run-{:016x}", mix_u64(base_seed ^ RUN_ID_DOMAIN))
}

fn game_id(run_id: &str, game_idx: usize) -> String {
    format!("{}-game-{:016x}", run_id, game_idx as u64)
}

fn mix_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn build_selfplay_searcher(params: &SelfPlayParams) -> Searcher {
    let mut s = Searcher::default();
    if params.actor_tt_mb > 0 {
        s.set_tt_capacity_mb(params.actor_tt_mb);
    }
    if let Some(model) = params.nnue_quant_model.as_ref() {
        s.set_use_nnue(true);
        s.set_eval_mode(EvalMode::Nnue);
        s.set_eval_blend_percent(params.nnue_blend_percent);
        s.set_nnue_quant_model(model.clone());
    }
    s
}

fn select_random_move(board: &Board, rng: &mut SmallRng) -> Option<MoveChoice> {
    let mut moves: Vec<Move> = Vec::new();
    board.generate_moves(|ml| {
        for m in ml {
            moves.push(m);
        }
        false
    });
    if moves.is_empty() {
        return None;
    }
    let mv = moves[rng.gen_range(0..moves.len())];
    Some(MoveChoice {
        played_mv: mv,
        target_best_mv: None,
        value_cp: None,
        teacher_depth: None,
        policy_top: Vec::new(),
    })
}

fn select_engine_move(
    board: &Board,
    params: &SelfPlayParams,
    ply_idx: usize,
    game_seed: u64,
    searcher: &mut Searcher,
) -> Option<MoveChoice> {
    // Produce one clean root label for every actor position. Outside the
    // exploration window this result is also the played move. During noisy
    // plies the policy searches below choose the played move, while this root
    // result remains the value target. That lets a depth-7/144k actor replace
    // a separate full-corpus depth-7 relabel pass without turning its sampled
    // opening move into the teacher target.
    let teacher_params = bestmove_search_params(params, params.depth);
    let teacher_result = searcher.search_with_params(board, teacher_params);
    let teacher_move = teacher_result.bestmove.as_ref().and_then(|best| {
        let mut found = None;
        board.generate_moves(|ml| {
            for mv in ml {
                if format!("{}", mv) == *best {
                    found = Some(mv);
                    break;
                }
            }
            found.is_some()
        });
        found
    })?;
    let teacher_score_white = if board.side_to_move() == Color::White {
        teacher_result.score_cp as f32
    } else {
        -(teacher_result.score_cp as f32)
    };

    // If temperature or Dirichlet requested, compute root policy and sample
    let use_temp = params.temperature_tau > 0.0 && ply_idx < params.temperature_moves;
    let use_dir = params.dirichlet_epsilon > 0.0 && ply_idx < params.dirichlet_plies;
    let use_policy = use_temp || use_dir;
    if use_policy {
        let mut moves: Vec<Move> = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return None;
        }
        // Score each child with a slightly reduced depth
        let pol_depth = if params.depth > 1 {
            params.depth - 1
        } else {
            1
        };
        let mut scores: Vec<f32> = Vec::with_capacity(moves.len());
        for &m in &moves {
            let mut child = board.clone();
            child.play_unchecked(m);
            let p = policy_search_params(params, pol_depth);
            let r = searcher.search_with_params(&child, p);
            let score_from_parent = -(r.score_cp as f32);
            scores.push(score_from_parent);
        }
        // Softmax with temperature
        // Anneal temperature linearly over first temperature_moves plies
        let tau = if use_temp && params.temperature_moves > 1 {
            let t0 = params.temperature_tau.max(0.0001);
            let t1 = params.temperature_tau_final.max(0.0001);
            let f = (ply_idx as f32) / (params.temperature_moves as f32 - 1.0);
            (1.0 - f) * t0 + f * t1
        } else if params.temperature_tau > 0.0 {
            params.temperature_tau
        } else {
            1.0
        };
        let scale = if params.temp_cp_scale > 0.0 {
            params.temp_cp_scale
        } else {
            200.0
        };
        let logits: Vec<f32> = scores.iter().map(|s| s / (scale * tau)).collect();
        let max_log = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut clean_probs: Vec<f32> = logits.iter().map(|l| (l - max_log).exp()).collect();
        let sum_p: f32 = clean_probs.iter().sum();
        if sum_p > 0.0 {
            for p in &mut clean_probs {
                *p /= sum_p;
            }
        } else {
            let n = clean_probs.len() as f32;
            for p in &mut clean_probs {
                *p = 1.0 / n;
            }
        }
        let mut sample_probs = clean_probs.clone();
        // Dirichlet noise
        if use_dir && params.dirichlet_alpha > 0.0 {
            let alpha = params.dirichlet_alpha;
            let gamma = Gamma::new(alpha, 1.0).unwrap();
            let mut rng = SmallRng::seed_from_u64(game_seed ^ zobrist::compute(board));
            let mut noise: Vec<f32> = (0..sample_probs.len())
                .map(|_| gamma.sample(&mut rng) as f32)
                .collect();
            let sum_n: f32 = noise.iter().sum();
            if sum_n > 0.0 {
                for n in &mut noise {
                    *n /= sum_n;
                }
            }
            let eps = params.dirichlet_epsilon;
            for i in 0..sample_probs.len() {
                sample_probs[i] = (1.0 - eps) * sample_probs[i] + eps * noise[i];
            }
        }
        // Sample according to probs
        let mut rng =
            SmallRng::seed_from_u64(game_seed ^ (zobrist::compute(board).rotate_left(13)));
        let r: f32 = rng.gen();
        let mut cdf = 0.0f32;
        let mut picked_idx = moves.len() - 1;
        for (i, &p) in sample_probs.iter().enumerate() {
            cdf += p.max(0.0);
            if r <= cdf {
                picked_idx = i;
                break;
            }
        }
        let mut order: Vec<usize> = (0..clean_probs.len()).collect();
        order.sort_by(|&a, &b| {
            clean_probs[b]
                .partial_cmp(&clean_probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let keep = order.len().min(8);
        let mut policy_top = Vec::with_capacity(keep);
        for &idx in &order[..keep] {
            policy_top.push((format!("{}", moves[idx]), clean_probs[idx]));
        }
        return Some(MoveChoice {
            played_mv: moves[picked_idx],
            target_best_mv: Some(teacher_move),
            value_cp: Some(teacher_score_white),
            teacher_depth: Some(teacher_result.depth),
            policy_top,
        });
    }
    Some(MoveChoice {
        played_mv: teacher_move,
        target_best_mv: Some(teacher_move),
        value_cp: Some(teacher_score_white),
        teacher_depth: Some(teacher_result.depth),
        policy_top: Vec::new(),
    })
}

fn load_openings(params: &SelfPlayParams) -> Result<Vec<Board>, String> {
    match params.openings_path {
        Some(ref p) => crate::io::openings::load_fen_suite(p),
        None => Ok(Vec::new()),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RecordBin {
    pub key: u64,
    pub result: i8, // from white perspective
    pub stm: u8,    // 0 white, 1 black
    pub _pad: u16,  // reserved
}

pub const SHARD_MAGIC: &[u8; 8] = b"PIESP001"; // Pie Self-Play v1
pub const RECORD_SIZE: usize = 8 + 1 + 1 + 2;

pub fn flatten_game_to_records(game: &GameRecord) -> Vec<RecordBin> {
    // PIESP001 has no field for outcome validity. Emitting an unfinished game
    // would turn its placeholder result=0 into a false training draw.
    if !game.outcome_valid {
        return Vec::new();
    }
    let mut recs = Vec::new();
    let mut board = Board::from_fen(&game.start_fen, false).unwrap_or_default();
    for mv_str in &game.moves {
        let key = zobrist::compute(&board);
        let stm = if board.side_to_move() == Color::White {
            0u8
        } else {
            1u8
        };
        recs.push(RecordBin {
            key,
            result: game.result,
            stm,
            _pad: 0,
        });
        // apply move
        let mut chosen = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == *mv_str {
                    chosen = Some(m);
                    break;
                }
            }
            chosen.is_some()
        });
        if let Some(m) = chosen {
            board.play_unchecked(m);
        } else {
            break;
        }
    }
    recs
}

#[derive(serde::Serialize)]
struct JsonlSelfPlayRecord<'a> {
    run_id: &'a str,
    game_id: &'a str,
    fen: String,
    ply: usize,
    result: i8,
    result_q: f32,
    outcome_valid: bool,
    termination: GameTermination,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_cp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_depth: Option<u32>,
    played_move: &'a str,
    target_best_move: &'a str,
    best_move: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    policy_top: Vec<JsonPolicyTopEntry<'a>>,
}

#[derive(serde::Serialize)]
struct JsonPolicyTopEntry<'a> {
    #[serde(rename = "move")]
    mv: &'a str,
    p: f32,
}

pub fn write_jsonl_shards<P: AsRef<Path>>(
    games: &[GameRecord],
    out_dir: P,
    max_records_per_shard: usize,
) -> std::io::Result<Vec<PathBuf>> {
    create_dir_all(&out_dir)?;
    let mut shard_index = 0usize;
    let mut rec_in_shard = 0usize;
    let mut out_paths = Vec::new();
    let mut writer: Option<BufWriter<File>> = None;

    let mut start_new_shard = |idx: usize| -> std::io::Result<BufWriter<File>> {
        let path = out_dir.as_ref().join(format!("shard_{:06}.jsonl", idx));
        let f = BufWriter::new(File::create(&path)?);
        out_paths.push(path);
        Ok(f)
    };

    for g in games {
        let mut board = Board::from_fen(&g.start_fen, false).unwrap_or_default();
        for (ply, mv_str) in g.moves.iter().enumerate() {
            if writer.is_none() || rec_in_shard >= max_records_per_shard {
                writer = Some(start_new_shard(shard_index)?);
                shard_index += 1;
                rec_in_shard = 0;
            }
            let w = writer.as_mut().unwrap();
            let value_cp = g.move_value_cp.get(ply).copied().flatten();
            let teacher_depth = if value_cp.is_some() {
                g.move_teacher_depth.get(ply).copied().flatten()
            } else {
                None
            };
            let target_best = g
                .move_target_best
                .get(ply)
                .and_then(|s| s.as_deref())
                .unwrap_or(mv_str.as_str());
            let mut policy_top = Vec::new();
            if let Some(items) = g.move_policy_top.get(ply) {
                policy_top.reserve(items.len());
                for (mv, p) in items {
                    policy_top.push(JsonPolicyTopEntry {
                        mv: mv.as_str(),
                        p: *p,
                    });
                }
            }
            let rec = JsonlSelfPlayRecord {
                run_id: g.run_id.as_str(),
                game_id: g.game_id.as_str(),
                fen: format!("{}", board),
                ply,
                result: g.result,
                result_q: g.result as f32,
                outcome_valid: g.outcome_valid,
                termination: g.termination,
                value_cp,
                teacher_depth,
                played_move: mv_str.as_str(),
                target_best_move: target_best,
                best_move: target_best,
                policy_top,
            };
            serde_json::to_writer(&mut *w, &rec)?;
            w.write_all(b"\n")?;
            rec_in_shard += 1;

            let mut chosen = None;
            board.generate_moves(|ml| {
                for m in ml {
                    if format!("{}", m) == *mv_str {
                        chosen = Some(m);
                        break;
                    }
                }
                chosen.is_some()
            });
            if let Some(m) = chosen {
                board.play_unchecked(m);
            } else {
                break;
            }
        }
    }
    if let Some(mut w) = writer {
        w.flush()?;
    }
    Ok(out_paths)
}

pub fn write_shards<P: AsRef<Path>>(
    games: &[GameRecord],
    out_dir: P,
    max_records_per_shard: usize,
) -> std::io::Result<Vec<PathBuf>> {
    create_dir_all(&out_dir)?;
    let mut shard_index = 0usize;
    let mut rec_in_shard = 0usize;
    let mut out_paths = Vec::new();
    let mut writer: Option<BufWriter<File>> = None;

    let mut start_new_shard = |idx: usize| -> std::io::Result<BufWriter<File>> {
        let path = out_dir.as_ref().join(format!("shard_{:06}.bin", idx));
        let mut f = BufWriter::new(File::create(&path)?);
        f.write_all(SHARD_MAGIC)?;
        out_paths.push(path);
        Ok(f)
    };

    for g in games {
        let recs = flatten_game_to_records(g);
        for r in recs {
            if writer.is_none() || rec_in_shard >= max_records_per_shard {
                writer = Some(start_new_shard(shard_index)?);
                shard_index += 1;
                rec_in_shard = 0;
            }
            let w = writer.as_mut().unwrap();
            let mut buf = [0u8; RECORD_SIZE];
            buf[0..8].copy_from_slice(&r.key.to_le_bytes());
            buf[8] = r.result as u8;
            buf[9] = r.stm;
            // pad zeros for 10..=11
            w.write_all(&buf)?;
            rec_in_shard += 1;
        }
    }
    // flush last shard
    if let Some(mut w) = writer {
        w.flush()?;
    }
    Ok(out_paths)
}

pub fn read_shard<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<RecordBin>> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != SHARD_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "bad magic",
        ));
    }
    let mut recs = Vec::new();
    let mut buf = [0u8; RECORD_SIZE];
    loop {
        match f.read_exact(&mut buf) {
            Ok(()) => {
                let mut key_bytes = [0u8; 8];
                key_bytes.copy_from_slice(&buf[0..8]);
                let key = u64::from_le_bytes(key_bytes);
                let result = buf[8] as i8;
                let stm = buf[9];
                recs.push(RecordBin {
                    key,
                    result,
                    stm,
                    _pad: 0,
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(recs)
}

#[cfg(test)]
mod tests {
    use super::{adjudicate_position, has_insufficient_material, GameTermination};
    use cozy_chess::Board;

    #[test]
    fn adjudicates_fifty_move_rule() {
        let board = Board::from_fen("4k3/8/8/8/8/8/R7/4K3 w - - 100 51", false).expect("valid FEN");
        assert_eq!(
            adjudicate_position(&board, &[board.clone()]),
            Some((0, GameTermination::FiftyMove))
        );
    }

    #[test]
    fn adjudicates_threefold_repetition() {
        let board = Board::default();
        let history = vec![board.clone(), board.clone(), board.clone()];
        assert_eq!(
            adjudicate_position(&board, &history),
            Some((0, GameTermination::ThreefoldRepetition))
        );
    }

    #[test]
    fn detects_only_strict_insufficient_material_cases() {
        let bare_kings =
            Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", false).expect("valid FEN");
        let bishop = Board::from_fen("4k3/8/8/8/8/8/8/2B1K3 w - - 0 1", false).expect("valid FEN");
        let mating_material =
            Board::from_fen("4k3/8/8/8/8/8/8/2BNK3 w - - 0 1", false).expect("valid FEN");

        assert!(has_insufficient_material(&bare_kings));
        assert!(has_insufficient_material(&bishop));
        assert!(!has_insufficient_material(&mating_material));
    }
}
