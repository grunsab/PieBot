use crate::search::alphabeta::{SearchParams, Searcher};
use crate::search::zobrist;
use cozy_chess::{Board, Color, Move};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct SelfPlayParams {
    pub games: usize,
    pub max_plies: usize,
    pub threads: usize,
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
}

pub struct GameRecord {
    pub start_fen: String,
    pub moves: Vec<String>, // played moves
    pub move_target_best: Vec<Option<String>>,     // teacher best move for each ply
    pub move_value_cp: Vec<Option<f32>>,           // white-perspective teacher value for each ply
    pub move_policy_top: Vec<Vec<(String, f32)>>,  // optional root policy samples
    pub result: i8, // 1 white win, 0 draw, -1 black win
}

struct MoveChoice {
    played_mv: Move,
    target_best_mv: Option<Move>,
    value_cp: Option<f32>,
    policy_top: Vec<(String, f32)>,
}

pub fn generate_games(params: &SelfPlayParams) -> Vec<GameRecord> {
    let mut rng = SmallRng::seed_from_u64(params.seed);
    let openings = load_openings(params);
    let mut games = Vec::with_capacity(params.games);
    for gi in 0..params.games {
        let mut board = if !openings.is_empty() {
            let idx = (rng.gen::<u64>() ^ (gi as u64)) as usize % openings.len();
            openings[idx].clone()
        } else {
            Board::default()
        };
        let mut record = GameRecord {
            start_fen: format!("{}", board),
            moves: Vec::new(),
            move_target_best: Vec::new(),
            move_value_cp: Vec::new(),
            move_policy_top: Vec::new(),
            result: 0,
        };
        let mut plies = 0usize;
        loop {
            if plies >= params.max_plies {
                break;
            }
            // Determine end conditions
            let mut has_move = false;
            board.generate_moves(|ml| {
                if !ml.is_empty() {
                    has_move = true;
                }
                false
            });
            if !has_move {
                if (board.checkers()).is_empty() {
                    record.result = 0;
                } else {
                    record.result = if board.side_to_move() == Color::White {
                        -1
                    } else {
                        1
                    };
                }
                break;
            }
            {
                // choose move
                let mv = if params.use_engine {
                    select_engine_move(&board, params, plies)
                } else {
                    select_random_move(&board, &mut rng)
                };
                if let Some(m) = mv {
                    let mstr = format!("{}", m.played_mv);
                    record.moves.push(mstr);
                    record
                        .move_target_best
                        .push(m.target_best_mv.map(|x| format!("{}", x)));
                    record.move_value_cp.push(m.value_cp);
                    record.move_policy_top.push(m.policy_top);
                    board.play_unchecked(m.played_mv);
                    plies += 1;
                } else {
                    break;
                }
            }
        }
        games.push(record);
    }
    games
}

fn select_random_move(board: &Board, rng: &mut SmallRng) -> Option<MoveChoice> {
    let mut moves: Vec<Move> = Vec::new();
    board.generate_moves(|ml| {
        for m in ml {
            moves.push(m);
        }
        false
    });
    if moves.is_empty() { return None; }
    let mv = moves[rng.gen_range(0..moves.len())];
    Some(MoveChoice {
        played_mv: mv,
        target_best_mv: None,
        value_cp: None,
        policy_top: Vec::new(),
    })
}

fn select_engine_move(board: &Board, params: &SelfPlayParams, ply_idx: usize) -> Option<MoveChoice> {
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
            let mut s = Searcher::default();
            let mut p = SearchParams::default();
            p.depth = pol_depth;
            p.use_tt = true;
            p.order_captures = true;
            p.use_history = true;
            p.threads = params.threads;
            p.use_aspiration = true;
            p.aspiration_window_cp = 50;
            p.use_lmr = true;
            p.use_killers = true;
            p.use_nullmove = true;
            p.max_nodes = Some(10_000);
            p.movetime = params
                .movetime_ms
                .map(|t| std::time::Duration::from_millis(t));
            let r = s.search_with_params(&child, p);
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
        let mut best_idx = 0usize;
        for i in 1..clean_probs.len() {
            if clean_probs[i] > clean_probs[best_idx] {
                best_idx = i;
            }
        }
        let mut sample_probs = clean_probs.clone();
        // Dirichlet noise
        if use_dir && params.dirichlet_alpha > 0.0 {
            let alpha = params.dirichlet_alpha;
            let gamma = Gamma::new(alpha, 1.0).unwrap();
            let mut rng = SmallRng::seed_from_u64(params.seed ^ zobrist::compute(board));
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
            SmallRng::seed_from_u64(params.seed ^ (zobrist::compute(board).rotate_left(13)));
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
        let best_cp_parent = scores[best_idx];
        let best_cp_white = if board.side_to_move() == Color::White {
            best_cp_parent
        } else {
            -best_cp_parent
        };
        return Some(MoveChoice {
            played_mv: moves[picked_idx],
            target_best_mv: Some(moves[best_idx]),
            value_cp: Some(best_cp_white),
            policy_top,
        });
    }
    // Greedy best move
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.depth = params.depth;
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = params.threads;
    p.use_aspiration = true;
    p.aspiration_window_cp = 50;
    p.use_lmr = true;
    p.use_killers = true;
    p.use_nullmove = true;
    p.max_nodes = Some(20_000);
    p.movetime = params
        .movetime_ms
        .map(|t| std::time::Duration::from_millis(t));
    let res = s.search_with_params(board, p);
    let score_white = if board.side_to_move() == Color::White {
        res.score_cp as f32
    } else {
        -(res.score_cp as f32)
    };
    res.bestmove.and_then(|s| {
        let mut choice = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == s {
                    choice = Some(m);
                    break;
                }
            }
            choice.is_some()
        });
        choice.map(|mv| MoveChoice {
            played_mv: mv,
            target_best_mv: Some(mv),
            value_cp: Some(score_white),
            policy_top: Vec::new(),
        })
    })
}

fn load_openings(params: &SelfPlayParams) -> Vec<Board> {
    let mut out = Vec::new();
    if let Some(ref p) = params.openings_path {
        if let Ok(mut f) = std::fs::File::open(p) {
            let mut s = String::new();
            if f.read_to_string(&mut s).is_ok() {
                for line in s.lines() {
                    let raw = line.trim();
                    if raw.is_empty() || raw.starts_with('#') {
                        continue;
                    }
                    // Support EPD (4 fields) by padding halfmove/fullmove
                    let parts: Vec<&str> = raw.split_whitespace().collect();
                    let fen = if parts.len() >= 6 {
                        parts[0..6].join(" ")
                    } else if parts.len() >= 4 {
                        let mut v = parts[0..4].to_vec();
                        v.push("0");
                        v.push("1");
                        v.join(" ")
                    } else {
                        raw.to_string()
                    };
                    if let Ok(b) = Board::from_fen(&fen, false) {
                        out.push(b);
                    }
                }
            }
        }
    }
    out
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
    fen: String,
    ply: usize,
    result: i8,
    result_q: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    value_cp: Option<f32>,
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
                fen: format!("{}", board),
                ply,
                result: g.result,
                result_q: g.result as f32,
                value_cp,
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
