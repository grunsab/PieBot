use cozy_chess::{Board, Move, Square};
use crate::search::eval::{eval_cp, material_eval_cp, MATE_SCORE, DRAW_SCORE};
use std::time::{Duration, Instant};
use crate::search::zobrist;
use crate::search::tt::{Tt, Entry, Bound};
use std::sync::Arc;
use rayon::prelude::*;
use std::sync::atomic::{AtomicI32, Ordering};
use crate::eval::nnue::network::QuantNetwork;
use crate::eval::nnue::loader::QuantNnue;
const HIST_PROMO_KINDS: usize = 5; // None, N, B, R, Q
const HIST_SIZE: usize = 64 * 64 * HIST_PROMO_KINDS;

#[inline]
fn promo_index(p: Option<cozy_chess::Piece>) -> usize {
    match p {
        Some(cozy_chess::Piece::Knight) => 1,
        Some(cozy_chess::Piece::Bishop) => 2,
        Some(cozy_chess::Piece::Rook) => 3,
        Some(cozy_chess::Piece::Queen) => 4,
        _ => 0,
    }
}

#[inline]
fn move_index(m: Move) -> usize {
    let from = m.from as usize;
    let to = m.to as usize;
    let pi = promo_index(m.promotion);
    (from * 64 + to) * HIST_PROMO_KINDS + pi
}

#[inline]
fn piece_value_cp(p: cozy_chess::Piece) -> i32 {
    match p {
        cozy_chess::Piece::Pawn => 100,
        cozy_chess::Piece::Knight => 320,
        cozy_chess::Piece::Bishop => 330,
        cozy_chess::Piece::Rook => 500,
        cozy_chess::Piece::Queen => 900,
        cozy_chess::Piece::King => 20000,
    }
}

#[inline]
fn piece_at(board: &Board, sq: Square) -> Option<(cozy_chess::Color, cozy_chess::Piece)> {
    for &color in &[cozy_chess::Color::White, cozy_chess::Color::Black] {
        let cb = board.colors(color);
        for &piece in &[cozy_chess::Piece::Pawn, cozy_chess::Piece::Knight, cozy_chess::Piece::Bishop, cozy_chess::Piece::Rook, cozy_chess::Piece::Queen, cozy_chess::Piece::King] {
            let bb = cb & board.pieces(piece);
            for s in bb { if s == sq { return Some((color, piece)); } }
        }
    }
    None
}

#[inline]
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let to = m.to; let from = m.from;
    let victim = piece_at(board, to).map(|(_, p)| piece_value_cp(p)).unwrap_or(0);
    let attacker = piece_at(board, from).map(|(_, p)| piece_value_cp(p)).unwrap_or(0);
    victim * 10 - attacker
}

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchParams {
    pub depth: u32,
    pub use_tt: bool,
    pub max_nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub order_captures: bool,
    pub use_history: bool,
    pub threads: usize,
    pub use_aspiration: bool,
    pub aspiration_window_cp: i32,
    pub use_lmr: bool,
    pub use_killers: bool,
    pub use_nullmove: bool,
    pub deterministic: bool,
}

#[derive(Default, Debug, Clone)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub score_cp: i32,
    pub nodes: u64,
}

pub struct Searcher {
    tt: Arc<Tt>,
    pub(crate) nodes: u64,
    node_limit: u64,
    deadline: Option<Instant>,
    order_captures: bool,
    use_history: bool,
    threads: usize,
    abort: Option<Arc<std::sync::atomic::AtomicBool>>,
    killers: Vec<[Option<Move>; 2]>,
    use_aspiration: bool,
    use_lmr: bool,
    use_killers: bool,
    use_nullmove: bool,
    // Optional NNUE evaluator (scalar path for now)
    use_nnue: bool,
    nnue: Option<crate::eval::nnue::Nnue>,
    nnue_quant: Option<QuantNetwork>,
    eval_blend_percent: u8, // 0..100, 0=PST only, 100=NNUE only
    // New: array-based history and counter-move tables
    history_table: Vec<i32>,
    counter_move: Vec<usize>,
    deterministic: bool,
    // Eval mode: material-only, PST, or NNUE
    eval_mode: EvalMode,
    // Instrumentation
    last_depth: u32,
    max_seldepth: u32,
}

impl Default for Searcher {
    fn default() -> Self {
        let mut t = Tt::new();
        t.set_capacity_entries(4096);
        Self {
            tt: Arc::new(t),
            nodes: 0,
            node_limit: u64::MAX,
            deadline: None,
            order_captures: false,
            use_history: false,
            threads: 1,
            abort: None,
            killers: Vec::new(),
            use_aspiration: false,
            use_lmr: false,
            use_killers: false,
            use_nullmove: false,
            use_nnue: false,
            nnue: None,
            nnue_quant: None,
            eval_blend_percent: 100,
            history_table: vec![0; HIST_SIZE],
            counter_move: vec![usize::MAX; HIST_SIZE],
            deterministic: false,
            eval_mode: EvalMode::Pst,
            last_depth: 0,
            max_seldepth: 0,
        }
    }
}

impl Searcher {
    // Choose evaluation mode
    pub fn set_eval_mode(&mut self, mode: EvalMode) { self.eval_mode = mode; }
    pub fn set_threads(&mut self, t: usize) { self.threads = t.max(1); }
    pub fn set_order_captures(&mut self, on: bool) { self.order_captures = on; }
    pub fn set_use_history(&mut self, on: bool) { self.use_history = on; }
    pub fn set_use_killers(&mut self, on: bool) { self.use_killers = on; }
    pub fn set_use_lmr(&mut self, on: bool) { self.use_lmr = on; }
    pub fn set_use_nullmove(&mut self, on: bool) { self.use_nullmove = on; }
    pub fn set_use_aspiration(&mut self, on: bool) { self.use_aspiration = on; }
    pub fn set_deterministic(&mut self, on: bool) { self.deterministic = on; }
    pub fn see_gain_cp(&mut self, board: &Board, uci: &str) -> Option<i32> {
        // Locate a matching legal move by UCI string
        let mut chosen: Option<Move> = None;
        board.generate_moves(|ml| {
            for m in ml { if format!("{}", m) == uci { chosen = Some(m); break; } }
            chosen.is_some()
        });
        chosen.and_then(|m| crate::search::see::see_gain_cp(board, m))
    }

    pub fn qsearch_eval_cp(&mut self, board: &Board) -> i32 {
        if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { qn.refresh(board); } }
        self.qsearch(board, -MATE_SCORE, MATE_SCORE)
    }

    // Time-managed iterative deepening up to a maximum depth
    pub fn search_movetime(&mut self, board: &Board, millis: u64, depth: u32) -> (Option<String>, i32, u64) {
        self.nodes = 0;
        self.node_limit = u64::MAX;
        self.deadline = Some(Instant::now() + Duration::from_millis(millis));
        if self.use_history {
            for h in &mut self.history_table { *h = 0; }
            for c in &mut self.counter_move { *c = usize::MAX; }
        }
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut best: Option<String> = None;
        let mut last_score = 0;
        for d in 1..=max_depth {
            self.tt.bump_generation();
            let res = self.search_depth(board, d);
            best = res.bestmove.clone();
            last_score = res.score_cp;
            self.last_depth = d;
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, last_score, self.nodes)
    }

    fn qsearch(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        // Stand pat
        let stand = self.eval_current(board);
        if stand >= beta { return beta; }
        if stand > alpha { alpha = stand; }

        // Captures only
        let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
        let opp_bb = board.colors(opp);
        let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
        let mut caps: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                if (occ_mask & bit) != 0 { caps.push(m); }
            }
            false
        });
        // Order captures quickly via MVV-LVA heuristic
        caps.sort_by_key(|&m| -mvv_lva_score(board, m));
        for m in caps {
            let mut child = board.clone(); child.play(m);
            let mut change = None;
            if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { change = Some(qn.apply_move(board, m, &child)); } }
            let score = -self.qsearch(&child, -beta, -alpha);
            if let Some(ch) = change { if let Some(qn) = self.nnue_quant.as_mut() { qn.revert(ch); } }
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        // Root-split parallel search if threads > 1 and depth > 1
        if self.threads > 1 && depth > 1 && !self.deterministic {
            return self.search_depth_parallel(board, depth);
        }

        if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { qn.refresh(board); } }
        let orig_alpha = alpha;
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }
        // TT-first
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }
        // Order with captures/history and a small bonus for checking moves
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            moves.sort_by_key(|&m| {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures { if (occ_mask & bit) != 0 { 1 } else { 0 } } else { 0 };
                let mvv = if is_cap == 1 { mvv_lva_score(board, m) } else { 0 };
                let see_b = if is_cap == 1 { crate::search::see::see_gain_cp(board, m).unwrap_or(0) / 8 } else { 0 };
                let gives_check_bonus = {
                    let mut c = board.clone(); c.play(m); if !(c.checkers()).is_empty() { 30 } else { 0 }
                };
                let mi = move_index(m);
                let hist = if self.use_history { self.history_table.get(mi).copied().unwrap_or(0) } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(0, m) } else { 0 };
                -(is_cap * 1000 + mvv + see_b + gives_check_bonus + kb + hist)
            });
        }
        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let mut change = None;
            if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { change = Some(qn.apply_move(board, m, &child)); } }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let score = -self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m));
            if let Some(ch) = change { if let Some(qn) = self.nnue_quant.as_mut() { qn.revert(ch); } }
            if score > best_score { best_score = score; bestmove = Some(m); }
            if score > alpha { alpha = score; }
        }

        // Store root in TT as exact when using full window
        let root_bound = if best_score <= orig_alpha { Bound::Upper } else if best_score >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best_score, bestmove, root_bound);

        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn search_depth_parallel(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }

        // Optional: TT move first (ordering only)
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }

        // Evaluate each root move independently with full window in parallel
        let deadline = self.deadline;
        let order_captures = self.order_captures;
        let use_history = self.use_history;
        let shared_tt = self.tt.clone();
        let quant_model = self.nnue_quant.as_ref().map(|qn| qn.model.clone());
        let use_nnue = self.use_nnue;
        let results: Vec<(Move, i32, u64)> = moves.par_iter().map(|&m| {
            let mut child = board.clone();
            child.play(m);
            let mut w = Searcher::default();
            w.node_limit = u64::MAX; // rely on shared deadline for stopping
            w.deadline = deadline;
            w.order_captures = order_captures;
            w.use_history = use_history;
            w.tt = shared_tt.clone();
            w.use_nnue = use_nnue;
            if let Some(model) = &quant_model { w.nnue_quant = Some(QuantNetwork::new(model.clone())); if w.use_nnue { if let Some(qn) = w.nnue_quant.as_mut() { qn.refresh(&child); } } }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let score = -w.alphabeta(&child, next_depth, -MATE_SCORE, MATE_SCORE, 1, move_index(m));
            (m, score, w.nodes)
        }).collect();

        // Reduce to best
        let mut best: Option<(Move, i32)> = None;
        let mut total_nodes = 0u64;
        for (m, s, n) in results {
            total_nodes += n;
            if best.map_or(true, |(_, bs)| s > bs) { best = Some((m, s)); }
        }
        self.nodes = total_nodes;
        if let Some((bm, sc)) = best {
            // Store TT root as exact
            self.tt_put(board, depth, sc, Some(bm), Bound::Exact);
            return SearchResult { bestmove: Some(format!("{}", bm)), score_cp: sc, nodes: self.nodes };
        }
        SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }
    }

    fn alphabeta(&mut self, board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32, parent_move_idx: usize) -> i32 {
        if let Some(ref flag) = self.abort { if flag.load(Ordering::Relaxed) { return self.eval_cp_internal(board); } }
        self.nodes += 1;
        if self.nodes >= self.node_limit { return self.eval_cp_internal(board); }
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval_cp_internal(board); } }
        if depth == 0 { return self.qsearch(board, alpha, beta); }
        // Null-move pruning (guarded)
        // Disable at depth <= 7 for mate puzzles (too tactical)
        if self.use_nullmove && depth >= 8 {
            // avoid in check
            if (board.checkers()).is_empty() {
                // null move
                let nb = board.clone();
                // cozy_chess null move; if unavailable, skip
                let null_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nb.null_move();
                })).is_ok();
                if null_ok {
                    let r = 2 + (depth / 4) as u32;
                    let score = -self.alphabeta(&nb, depth - 1 - r, -beta, -beta + 1, ply + 1, usize::MAX);
                    if score >= beta { return score; }
                }
            }
        }

        // TT probe (exact-only)
        if let Some(en) = self.tt_get(board) {
            if en.depth >= depth {
                match en.bound {
                    Bound::Exact => return en.score,
                    Bound::Lower => if en.score >= beta { return en.score; },
                    Bound::Upper => if en.score <= alpha { return en.score; },
                }
            }
        }

        // Build movelist and order
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return self.eval_terminal(board, ply); }
        // TT move first
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }
        // Captures-first, killers, and history ordering
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            moves.sort_by_key(|&m| {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures { if (occ_mask & bit) != 0 { 1 } else { 0 } } else { 0 };
                let mvv = if is_cap == 1 { mvv_lva_score(board, m) } else { 0 };
                let see_b = if is_cap == 1 { crate::search::see::see_gain_cp(board, m).unwrap_or(0) / 8 } else { 0 };
                let mi = move_index(m);
                let hist = if self.use_history { self.history_table.get(mi).copied().unwrap_or(0) } else { 0 };
                let cm = if self.use_history && parent_move_idx != usize::MAX {
                    if self.counter_move.get(parent_move_idx).copied().unwrap_or(usize::MAX) == mi { 40 } else { 0 }
                } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(ply, m) } else { 0 };
                -(is_cap * 1000 + mvv + see_b + kb + hist + cm)
            });
        }

        // In-tree split (jamboree-lite): PV seed + parallel tail with shared alpha
        if self.threads > 1 && depth >= 3 && moves.len() >= 12 {
            let shared_tt = self.tt.clone();
            let deadline = self.deadline;
            let order_captures = self.order_captures;
            let use_history = self.use_history;
            let quant_model = self.nnue_quant.as_ref().map(|qn| qn.model.clone());
            let use_nnue = self.use_nnue;

            // PV seed: evaluate first move serially to get a strong alpha
            let mut iter = moves.into_iter();
            let first = iter.next().unwrap();
            let mut child = board.clone();
            child.play(first);
            let mut seed = Searcher::default();
            seed.node_limit = u64::MAX;
            seed.deadline = deadline;
            seed.order_captures = order_captures;
            seed.use_history = use_history;
            seed.tt = shared_tt.clone();
            seed.use_nnue = use_nnue;
            if let Some(model) = &quant_model { seed.nnue_quant = Some(QuantNetwork::new(model.clone())); if seed.use_nnue { if let Some(qn) = seed.nnue_quant.as_mut() { qn.refresh(&child); } } }
            let mut best = -seed.alphabeta(&child, depth - 1, -MATE_SCORE, MATE_SCORE, ply + 1, move_index(first));
            let mut best_move_local: Option<Move> = Some(first);
            self.nodes += seed.nodes;
            let alpha_shared = AtomicI32::new(best);
            let abort_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

            // Parallel tail search with shared alpha
            let tails: Vec<Move> = iter.collect();
            let results: Vec<(Move, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone();
                c.play(m);
                let mut w = Searcher::default();
                w.node_limit = u64::MAX;
                w.deadline = deadline;
                w.order_captures = order_captures;
                w.use_history = use_history;
                w.tt = shared_tt.clone();
                w.use_nnue = use_nnue;
                if let Some(model) = &quant_model { w.nnue_quant = Some(QuantNetwork::new(model.clone())); if w.use_nnue { if let Some(qn) = w.nnue_quant.as_mut() { qn.refresh(&c); } } }
                w.abort = Some(abort_flag.clone());
                // Read current alpha
                let a = alpha_shared.load(Ordering::Relaxed);
                if abort_flag.load(Ordering::Relaxed) { return (m, -MATE_SCORE, 0); }
                let score = -w.alphabeta(&c, depth - 1, -MATE_SCORE, -a, ply + 1, move_index(m));
                // Update shared alpha if improved
                let mut cur = a;
                while score > cur {
                    match alpha_shared.compare_exchange(cur, score, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(observed) => { if observed >= score { break; } cur = observed; }
                    }
                }
                if score >= beta { abort_flag.store(true, Ordering::Relaxed); }
                (m, score, w.nodes)
            }).collect();

            for (m, s, n) in results {
                self.nodes += n;
                if s > best { best = s; best_move_local = Some(m); }
            }
            // Store as exact at this node
            self.tt_put(board, depth, best, best_move_local, Bound::Exact);
            if let Some(mv) = best_move_local {
                if self.use_history { let mi = move_index(mv); if let Some(h) = self.history_table.get_mut(mi) { *h += (depth as i32) * (depth as i32); } }
            }
            return best;
        }

        let mut best = -MATE_SCORE;
        let mut best_move_local: Option<Move> = None;
        let orig_alpha = alpha;
        for (idx, m) in moves.into_iter().enumerate() {
            let mut child = board.clone();
            child.play(m);
            let gives_check = !(child.checkers()).is_empty();
            let ext = if gives_check { 1 } else { 0 };
            let score;
            if self.use_lmr && depth >= 3 {
                // Simple LMR: reduce late quiet moves
                let is_cap = self.is_capture(&board, m);
                if !is_cap && !gives_check && idx >= 3 {
                    let r = 1; // basic reduction
                    let red = -self.alphabeta(&child, depth - 1 - r + ext, -alpha - 1, -alpha, ply + 1, move_index(m));
                    if red > alpha { score = -self.alphabeta(&child, depth - 1 + ext, -beta, -alpha, ply + 1, move_index(m)); } else { score = red; }
                } else {
                    score = -self.alphabeta(&child, depth - 1 + ext, -beta, -alpha, ply + 1, move_index(m));
                }
            } else {
                score = -self.alphabeta(&child, depth - 1 + ext, -beta, -alpha, ply + 1, move_index(m));
            }
            if score > best { best = score; best_move_local = Some(m); }
            if best > alpha { alpha = best; }
            if alpha >= beta { break; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
            // (removed) string-based continuation history
        }
        // Store exact score and best move
        let bound = if best <= orig_alpha { Bound::Upper } else if best >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best, best_move_local, bound);
        if let Some(mv) = best_move_local {
            let mi = move_index(mv);
            if self.use_history { let v = (depth as i32) * (depth as i32); if let Some(h) = self.history_table.get_mut(mi) { *h += v; } }
            if self.use_killers && bound == Bound::Lower { self.update_killers(ply, mv); }
            if self.use_history && bound == Bound::Lower && parent_move_idx != usize::MAX {
                if let Some(slot) = self.counter_move.get_mut(parent_move_idx) { *slot = mi; }
            }
        }
        best
    }

    fn eval_terminal(&self, board: &Board, ply: i32) -> i32 {
        if !(board.checkers()).is_empty() { return -MATE_SCORE + ply; }
        DRAW_SCORE
    }
}

impl Searcher {
    fn tt_key(board: &Board) -> u64 { zobrist::compute(board) }
    fn tt_get(&self, board: &Board) -> Option<Entry> { self.tt.get(Self::tt_key(board)) }
    fn tt_put(&mut self, board: &Board, depth: u32, score: i32, best: Option<Move>, bound: Bound) {
        let e = Entry { key: Self::tt_key(board), depth, score, best, bound, gen: 0 };
        self.tt.put(e);
    }

    pub fn search_with_params(&mut self, board: &Board, params: SearchParams) -> SearchResult {
        // Configure this search
        self.nodes = 0;
        self.last_depth = 0;
        self.node_limit = params.max_nodes.unwrap_or(u64::MAX);
        if !params.use_tt { self.tt = Arc::new(Tt::new()); }
        self.order_captures = params.order_captures;
        self.use_history = params.use_history;
        self.threads = params.threads.max(1);
        self.use_aspiration = params.use_aspiration;
        self.use_lmr = params.use_lmr;
        self.use_killers = params.use_killers;
        self.use_nullmove = params.use_nullmove;
        self.killers = vec![[None, None]; 256];
        self.deterministic = params.deterministic;
        if self.use_history {
            for h in &mut self.history_table { *h = 0; }
            for c in &mut self.counter_move { *c = usize::MAX; }
        }
        let mut best: Option<String> = None;
        let mut last_score = 0;
        self.deadline = params.movetime.map(|d| Instant::now() + d);
        let max_depth = if params.depth == 0 { 99 } else { params.depth };
        for d in 1..=max_depth {
            self.tt.bump_generation();
            let r = if self.use_aspiration && d > 1 {
                let window = params.aspiration_window_cp.max(10);
                let alpha = last_score - window;
                let beta = last_score + window;
                let mut res = self.search_depth_window(board, d, alpha, beta);
                if res.score_cp <= alpha || res.score_cp >= beta {
                    res = self.search_depth(board, d);
                }
                res
            } else {
                self.search_depth(board, d)
            };
            best = r.bestmove.clone();
            last_score = r.score_cp;
            self.last_depth = d;
            if self.nodes >= self.node_limit { break; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        SearchResult { bestmove: best, score_cp: last_score, nodes: self.nodes }
    }

    fn search_depth_window(&mut self, board: &Board, depth: u32, alpha0: i32, beta0: i32) -> SearchResult {
        let mut alpha = alpha0;
        let beta = beta0;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        if self.threads > 1 && depth > 1 { return self.search_depth(board, depth); }

        if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { qn.refresh(board); } }
        let orig_alpha = alpha;
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                }
            }
        }
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            moves.sort_by_key(|&m| {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures { if (occ_mask & bit) != 0 { 1 } else { 0 } } else { 0 };
                let gives_check_bonus = {
                    let mut c = board.clone(); c.play(m); if !(c.checkers()).is_empty() { 30 } else { 0 }
                };
                let mi = move_index(m);
                let hist = if self.use_history { self.history_table.get(mi).copied().unwrap_or(0) } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(0, m) } else { 0 };
                -(is_cap * 10 + gives_check_bonus + kb + hist)
            });
        }
        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let mut change = None;
            if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { change = Some(qn.apply_move(board, m, &child)); } }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let score = -self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m));
            if let Some(ch) = change { if let Some(qn) = self.nnue_quant.as_mut() { qn.revert(ch); } }
            if score > best_score { best_score = score; bestmove = Some(m); }
            if score > alpha { alpha = score; }
        }
        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn is_capture(&self, board: &Board, m: Move) -> bool {
        let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
        let opp_bb = board.colors(opp);
        let m_str = format!("{}", m);
        let to_str = &m_str[2..4];
        for sq in opp_bb { if format!("{}", sq) == to_str { return true; } }
        false
    }

    fn update_killers(&mut self, ply: i32, m: Move) {
        let p = ply as usize;
        if p >= self.killers.len() { return; }
        let slot = &mut self.killers[p];
        if slot[0] == Some(m) { return; }
        if slot[1] == Some(m) {
            slot[1] = slot[0];
            slot[0] = Some(m);
            return;
        }
        slot[1] = slot[0];
        slot[0] = Some(m);
    }

    fn killer_bonus(&self, ply: i32, m: Move) -> i32 {
        let p = ply as usize;
        if p >= self.killers.len() { return 0; }
        let slot = &self.killers[p];
        if slot[0] == Some(m) { 50 } else if slot[1] == Some(m) { 30 } else { 0 }
    }

    // removed string-based continuation parent key

    pub fn tt_probe(&self, board: &Board) -> Option<(u32, Bound)> {
        self.tt_get(board).map(|e| (e.depth, e.bound))
    }

    pub fn set_tt_capacity_mb(&mut self, mb: usize) {
        let mut tt = Tt::new();
        tt.set_capacity_mb(mb);
        self.tt = Arc::new(tt);
    }
    pub fn get_threads(&self) -> usize { self.threads }
    pub fn last_depth(&self) -> u32 { self.last_depth }
    pub fn last_seldepth(&self) -> u32 { self.max_seldepth }

    pub fn set_use_nnue(&mut self, on: bool) { self.use_nnue = on; }
    pub fn set_nnue_network(&mut self, nn: Option<crate::eval::nnue::Nnue>) { self.nnue = nn; }
    pub fn set_nnue_quant_model(&mut self, model: QuantNnue) { self.nnue_quant = Some(QuantNetwork::new(model)); }
    pub fn clear_nnue_quant(&mut self) { self.nnue_quant = None; }
    pub fn set_eval_blend_percent(&mut self, p: u8) { self.eval_blend_percent = p.min(100); }

    fn eval_cp_internal(&self, board: &Board) -> i32 { self.eval_current(board) }

    #[inline]
    fn eval_current(&self, board: &Board) -> i32 {
        match self.eval_mode {
            EvalMode::Material => material_eval_cp(board),
            EvalMode::Pst => eval_cp(board),
            EvalMode::Nnue => {
                // Fall back to PST if no NNUE configured
                if self.use_nnue {
                    if let Some(qn) = &self.nnue_quant {
                        let score = qn.eval_full(board);
                        return if board.side_to_move() == cozy_chess::Color::White { score } else { -score };
                    } else if let Some(nn) = &self.nnue {
                        let score = nn.evaluate(board);
                        return if board.side_to_move() == cozy_chess::Color::White { score } else { -score };
                    }
                }
                eval_cp(board)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EvalMode { Material, Pst, Nnue }
