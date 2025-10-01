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
// (no global switch) Null-move pruning is controlled per-searcher via a depth gate.

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
    // Continuation history (parent_move_idx, child_move_idx) -> bonus
    cont_hist: Vec<i32>,
    // Minimum remaining depth to allow null-move pruning
    null_min_depth: u32,
    // Minimum remaining depth for history to contribute in ordering
    hist_min_depth: u32,
    // Root-only SEE refinement: number of top moves to refine by SEE
    root_see_top_k: usize,
    // Pruning toggles
    use_futility: bool,
    use_lmp: bool,
    deterministic: bool,
    // Eval mode: material-only, PST, or NNUE
    eval_mode: EvalMode,
    // Instrumentation
    last_depth: u32,
    max_seldepth: u32,
    // SMP/Lazy-SMP diversification knobs
    lmr_aggr: i32,
    null_r_bonus: i32,
    tt_first: bool,
    order_offset: usize,
    // Helper/tail write policy: when true, helpers only write Exact bounds to TT
    helper_tt_exact_only: bool,
    // Root tail policy for parallel search at root/in-tree
    tail_policy: TailPolicy,
    // SEE ordering control
    see_ordering: bool,
    see_ordering_topk: usize,
    // Singular extensions + IID
    use_singular: bool,
    singular_margin_cp: i32,
    iid_strong: bool,
    // SMP control
    smp_diversify: bool,
    smp_safe: bool,
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
            cont_hist: vec![0; 1 << 18],
            null_min_depth: 10,
            hist_min_depth: 0,
            root_see_top_k: 0,
            use_futility: false,
            use_lmp: false,
            deterministic: false,
            eval_mode: EvalMode::Pst,
            last_depth: 0,
            max_seldepth: 0,
            lmr_aggr: 0,
            null_r_bonus: 0,
            tt_first: true,
            order_offset: 0,
            helper_tt_exact_only: false,
            tail_policy: TailPolicy::Pvs,
            see_ordering: false,
            see_ordering_topk: 0,
            use_singular: true,
            singular_margin_cp: 32,
            iid_strong: true,
            smp_diversify: true,
            smp_safe: false,
        }
    }
}

impl Searcher {
    #[inline]
    fn cont_index(parent_idx: usize, child_idx: usize) -> usize {
        let key = (parent_idx as u64).wrapping_mul(1_000_003) ^ (child_idx as u64).wrapping_mul(0x9E37_79B9);
        (key as usize) & ((1 << 18) - 1)
    }

    #[inline]
    fn order_moves_internal(&self, board: &Board, moves: &mut Vec<Move>, parent_move_idx: usize, ply: i32, remaining_depth: u32) {
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            // Precompute top-K capture set for SEE refinement
            let mut cap_candidates: Vec<(Move, i32)> = Vec::new();
            if self.see_ordering && self.see_ordering_topk > 0 {
                for &m in moves.iter() {
                    let to_sq: Square = m.to; let bit = 1u64 << (to_sq as usize);
                    if (occ_mask & bit) != 0 { cap_candidates.push((m, mvv_lva_score(board, m))); }
                }
                cap_candidates.sort_by_key(|&(_, sc)| -sc);
            }
            moves.sort_by_key(|&m| {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures { if (occ_mask & bit) != 0 { 1 } else { 0 } } else { 0 };
                let mvv = if is_cap == 1 { mvv_lva_score(board, m) } else { 0 };
                let mi = move_index(m);
                // Apply history only when deep enough to avoid shallow tactical regressions
                let hist = if self.use_history && remaining_depth >= self.hist_min_depth { self.history_table.get(mi).copied().unwrap_or(0) } else { 0 };
                let cm = if self.use_history && parent_move_idx != usize::MAX {
                    if self.counter_move.get(parent_move_idx).copied().unwrap_or(usize::MAX) == mi { 200 } else { 0 }
                } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(ply, m) } else { 0 };
                let cont = if parent_move_idx != usize::MAX { self.cont_hist[Self::cont_index(parent_move_idx, mi)] } else { 0 };
                let see_b = if is_cap == 1 && self.see_ordering {
                    if self.see_ordering_topk == 0 || cap_candidates.iter().take(self.see_ordering_topk.min(cap_candidates.len())).any(|&(mm, _)| mm == m) {
                        crate::search::see::see_gain_cp(board, m).unwrap_or(0) / 8
                    } else { 0 }
                } else { 0 };
                -(is_cap * 1000 + mvv + kb + hist + cm + cont + see_b)
            });
            // Diversification: rotate tail by offset
            if self.order_offset > 0 && moves.len() > 2 {
                let tail = &mut moves[1..];
                let k = self.order_offset % tail.len();
                tail.rotate_left(k);
            }
        }
    }
    // Choose evaluation mode
    pub fn set_eval_mode(&mut self, mode: EvalMode) { self.eval_mode = mode; }
    pub fn set_threads(&mut self, t: usize) { self.threads = t.max(1); }
    pub fn set_order_captures(&mut self, on: bool) { self.order_captures = on; }
    pub fn set_use_history(&mut self, on: bool) { self.use_history = on; }
    pub fn set_use_killers(&mut self, on: bool) { self.use_killers = on; }
    pub fn set_use_lmr(&mut self, on: bool) { self.use_lmr = on; }
    pub fn set_use_nullmove(&mut self, on: bool) { self.use_nullmove = on; }
    pub fn set_use_futility(&mut self, on: bool) { self.use_futility = on; }
    pub fn set_use_lmp(&mut self, on: bool) { self.use_lmp = on; }
    pub fn set_use_aspiration(&mut self, on: bool) { self.use_aspiration = on; }
    pub fn set_deterministic(&mut self, on: bool) { self.deterministic = on; }
    // Diversification and ordering knobs (for SMP helpers)
    pub fn set_lmr_aggr(&mut self, v: i32) { self.lmr_aggr = v; }
    pub fn set_null_r_bonus(&mut self, v: i32) { self.null_r_bonus = v; }
    pub fn set_tt_first(&mut self, on: bool) { self.tt_first = on; }
    pub fn set_order_offset(&mut self, off: usize) { self.order_offset = off; }
    pub fn set_helper_tt_exact_only(&mut self, on: bool) { self.helper_tt_exact_only = on; }
    pub fn set_tail_policy(&mut self, p: TailPolicy) { self.tail_policy = p; }
    pub fn set_see_ordering(&mut self, on: bool) { self.see_ordering = on; }
    pub fn set_see_ordering_topk(&mut self, k: usize) { self.see_ordering_topk = k; }
    pub fn set_use_singular(&mut self, on: bool) { self.use_singular = on; }
    pub fn set_singular_margin(&mut self, cp: i32) { self.singular_margin_cp = cp; }
    pub fn set_iid_strong(&mut self, on: bool) { self.iid_strong = on; }
    pub fn set_smp_diversify(&mut self, on: bool) { self.smp_diversify = on; }
    pub fn set_smp_safe(&mut self, on: bool) { self.smp_safe = on; }
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
        // Use Lazy-SMP when multiple threads are configured and determinism is not requested
        if self.threads > 1 && !self.deterministic {
            let (bm, sc, n, d) = self.search_movetime_lazy_smp(board, millis, depth);
            self.last_depth = d;
            return (bm, sc, n);
        }
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

    // Lazy-SMP style: run N independent workers with shared TT and minor heuristic diversification,
    // pick the deepest result (ties prefer worker 0).
    pub fn search_movetime_lazy_smp(&mut self, board: &Board, millis: u64, depth: u32) -> (Option<String>, i32, u64, u32) {
        let threads = self.threads.max(1);
        if threads == 1 { let (bm, sc, n) = self.search_movetime(board, millis, depth); return (bm, sc, n, self.last_depth); }
        let shared_tt = self.tt.clone();
        let use_nnue = self.use_nnue;
        let quant_model = self.nnue_quant.as_ref().map(|qn| qn.model.clone());
        let deadline = Some(Instant::now() + Duration::from_millis(millis));
        let maxd = if depth == 0 { 99 } else { depth };
        // Auto-enable SMP-safe profile at very short movetimes to reduce tactical instability
        let auto_safe = self.threads > 1 && millis <= 20;
        let smp_safe = self.smp_safe || auto_safe;
        let smp_diversify = self.smp_diversify && !smp_safe;
        let helper_tt_exact_only = self.helper_tt_exact_only || smp_safe;
        let tail_policy = if smp_safe { TailPolicy::Full } else { self.tail_policy };
        let results: Vec<(usize, Option<String>, i32, u64, u32)> = (0..threads).into_par_iter().map(|wid| {
            let mut w = Searcher::default();
            w.tt = shared_tt.clone();
            w.threads = 1;
            // keep shared TT (do not reset)
            w.order_captures = self.order_captures;
            w.use_history = self.use_history;
            w.use_killers = self.use_killers;
            w.use_lmr = self.use_lmr;
            w.use_nullmove = self.use_nullmove;
            w.set_null_min_depth(self.null_min_depth);
            w.use_aspiration = self.use_aspiration;
            w.set_deterministic(false);
            // Diversify per worker
            if wid > 0 && smp_diversify { w.set_lmr_aggr(1); w.set_null_r_bonus(1); w.set_tt_first(wid % 2 == 0); w.set_order_offset(wid as usize); }
            // Helper mode safeguards
            w.set_helper_tt_exact_only(helper_tt_exact_only);
            w.set_tail_policy(tail_policy);
            w.set_smp_safe(smp_safe);
            // Optional: shallow pruning on helpers (disable in safe mode)
            if smp_safe { w.set_use_futility(false); w.set_use_lmp(false); } else { w.set_use_futility(self.use_futility); w.set_use_lmp(self.use_lmp); }
            // NNUE
            w.use_nnue = use_nnue;
            if let Some(model) = &quant_model { w.nnue_quant = Some(QuantNetwork::new(model.clone())); }
            w.deadline = deadline;
            // Use search_with_params to benefit from iterative deepening
            let mut p = SearchParams::default();
            p.depth = maxd;
            p.use_tt = true;
            p.max_nodes = None;
            p.movetime = Some(Duration::from_millis(millis));
            p.order_captures = self.order_captures;
            p.use_history = self.use_history;
            p.threads = 1;
            // Aspiration only on worker 0 to reduce instability
            p.use_aspiration = wid == 0;
            p.aspiration_window_cp = 30;
            p.use_lmr = self.use_lmr;
            p.use_killers = self.use_killers;
            p.use_nullmove = self.use_nullmove;
            p.deterministic = false;
            let r = w.search_with_params(board, p);
            (wid, r.bestmove, r.score_cp, r.nodes, w.last_depth())
        }).collect();
        // Consensus selection: prefer deepest among majority bestmove; fallback to deepest (tie -> worker 0)
        use std::collections::HashMap;
        let mut freq: HashMap<String, usize> = HashMap::new();
        for (_, bm, _, _, _) in &results {
            if let Some(b) = bm { *freq.entry(b.clone()).or_insert(0) += 1; }
        }
        let majority = freq.iter().max_by_key(|(_, c)| **c).map(|(bm, _)| bm.clone());
        let mut best = results[0].clone();
        if let Some(maj) = majority {
            let mut cand: Option<(usize, Option<String>, i32, u64, u32)> = None;
            for r in &results {
                if r.1.as_deref() == Some(maj.as_str()) {
                    if cand.is_none() || r.4 > cand.as_ref().unwrap().4 { cand = Some(r.clone()); }
                }
            }
            if let Some(c) = cand { best = c; }
            else {
                for r in &results { if r.4 > best.4 || (r.4 == best.4 && r.0 == 0 && best.0 != 0) { best = r.clone(); } }
            }
        } else {
            for r in &results { if r.4 > best.4 || (r.4 == best.4 && r.0 == 0 && best.0 != 0) { best = r.clone(); } }
        }
        let nodes: u64 = results.iter().map(|r| r.3).sum();
        (best.1, best.2, nodes, best.4)
    }

    fn qsearch(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        // Terminal detection at horizon: stalemate or checkmate
        {
            let mut has_legal = false;
            board.generate_moves(|_| { has_legal = true; true });
            if !has_legal { return self.eval_terminal(board, 0); }
        }

        // If in check, do not stand pat; search evasions (full moves)
        let in_check = !(board.checkers()).is_empty();
        if !in_check {
            // Stand pat for quiet positions
            let stand = self.eval_current(board);
            if stand >= beta { return beta; }
            if stand > alpha { alpha = stand; }
        }

        // Captures only (first)
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
        // Limited checks in qsearch: explore a small number of checking non-captures
        let mut checks_tried = 0usize;
        let checks_cap = 6usize; // cap to avoid explosion
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        for m in moves {
            let to_sq: Square = m.to;
            let bit = 1u64 << (to_sq as usize);
            if (occ_mask & bit) != 0 { continue; } // skip captures (already done)
            let mut child = board.clone(); child.play(m);
            if !(child.checkers()).is_empty() {
                let score = -self.qsearch(&child, -beta, -alpha);
                if score >= beta { return beta; }
                if score > alpha { alpha = score; }
                checks_tried += 1; if checks_tried >= checks_cap { break; }
            }
        }
        alpha
    }

    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;
        let mut second_move: Option<Move> = None;
        let mut second_score = -MATE_SCORE;

        // Root-split parallel search if threads > 1 and depth > 1
        if self.threads > 1 && depth > 1 && !self.deterministic {
            return self.search_depth_parallel(board, depth);
        }

        if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { qn.refresh(board); } }
        let orig_alpha = alpha;
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        // Root blunder guard: push obviously hanging quiet moves and clearly losing captures to the end
        if depth >= 1 {
            let quiet_loss_thresh = 200; // roughly a minor piece
            let capture_loss_thresh = 300; // avoid large losing captures (e.g., queen sac blunders)
            let severe_capture_loss = 500; // discard extremely losing captures entirely when alternatives exist
            let mut safe: Vec<Move> = Vec::with_capacity(moves.len());
            let mut blunders: Vec<Move> = Vec::new();
            let mut severe: Vec<Move> = Vec::new();
            // Precompute opponent occupancy for capture detection
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            for m in moves.drain(..) {
                let to_sq: Square = m.to;
                let to_bit = 1u64 << (to_sq as usize);
                let is_cap_fast = (occ_mask & to_bit) != 0; // exclude EP; acceptable for root guard
                let mut tmp = board.clone(); tmp.play(m);
                let gives_check = !(tmp.checkers()).is_empty();
                if !is_cap_fast && !gives_check && crate::search::safety::is_hanging_after_move(board, m, quiet_loss_thresh) {
                    blunders.push(m);
                } else if is_cap_fast {
                    // Losing capture by SEE? Push to tail
                    let see = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    if see <= -severe_capture_loss { severe.push(m); }
                    else if see <= -capture_loss_thresh { blunders.push(m); }
                    else { safe.push(m); }
                } else { safe.push(m); }
            }
            if !safe.is_empty() {
                safe.extend(blunders.into_iter());
                // Discard severe losers if we have alternatives
                moves = safe;
            } else {
                // No safe alternatives; include everything
                safe.extend(blunders.into_iter());
                safe.extend(severe.into_iter());
                moves = safe;
            }
        }
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }
        // Apply the same root guard in the windowed variant to keep behavior consistent
        if depth >= 1 {
            let quiet_loss_thresh = 200;
            let capture_loss_thresh = 300;
            let severe_capture_loss = 500;
            let mut safe: Vec<Move> = Vec::with_capacity(moves.len());
            let mut blunders: Vec<Move> = Vec::new();
            let mut severe: Vec<Move> = Vec::new();
            let opp = if board.side_to_move() == cozy_chess::Color::White { cozy_chess::Color::Black } else { cozy_chess::Color::White };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0; for sq in opp_bb { occ_mask |= 1u64 << (sq as usize); }
            for m in moves.drain(..) {
                let to_sq: Square = m.to;
                let to_bit = 1u64 << (to_sq as usize);
                let is_cap_fast = (occ_mask & to_bit) != 0;
                let mut tmp = board.clone(); tmp.play(m);
                let gives_check = !(tmp.checkers()).is_empty();
                if !is_cap_fast && !gives_check && crate::search::safety::is_hanging_after_move(board, m, quiet_loss_thresh) {
                    blunders.push(m);
                } else if is_cap_fast {
                    let see = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    if see <= -severe_capture_loss { severe.push(m); }
                    else if see <= -capture_loss_thresh { blunders.push(m); }
                    else { safe.push(m); }
                } else { safe.push(m); }
            }
            if !safe.is_empty() {
                safe.extend(blunders.into_iter());
                moves = safe;
            } else {
                safe.extend(blunders.into_iter());
                safe.extend(severe.into_iter());
                moves = safe;
            }
        }
        // TT-first (Exact-only trust)
        if self.tt_first {
            if let Some(en) = self.tt_get(board) {
                if let Some(ttm) = en.best {
                    let trusted = matches!(en.bound, Bound::Exact);
                    if trusted {
                        if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                            let mv = moves.remove(pos);
                            moves.insert(0, mv);
                        }
                    }
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
            // Rotate tail for diversification if requested
            if self.order_offset > 0 && moves.len() > 2 {
                let k = self.order_offset % (moves.len() - 1);
                moves[1..].rotate_left(k);
            }
            // Optional root-only SEE refinement for the top-K moves
            if self.root_see_top_k > 0 && !moves.is_empty() {
                let k = self.root_see_top_k.min(moves.len());
                let mut prefix: Vec<Move> = moves[..k].to_vec();
                // Partition captures vs non-captures in prefix
                let mut caps: Vec<(Move, i32)> = Vec::new();
                let mut quiets: Vec<Move> = Vec::new();
                for &m in &prefix {
                    let to_sq: Square = m.to;
                    let bit = 1u64 << (to_sq as usize);
                    if (occ_mask & bit) != 0 {
                        let see = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                        caps.push((m, see));
                    } else {
                        quiets.push(m);
                    }
                }
                caps.sort_by_key(|&(_, see)| -see);
                let mut refined: Vec<Move> = caps.into_iter().map(|(m, _)| m).collect();
                refined.extend(quiets.into_iter());
                // Copy back refined prefix
                for i in 0..k { moves[i] = refined[i]; }
            }
        }
        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let mut change = None;
            if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { change = Some(qn.apply_move(board, m, &child)); } }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let score = -self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m));
            if let Some(ch) = change { if let Some(qn) = self.nnue_quant.as_mut() { qn.revert(ch); } }
            if score > best_score {
                second_score = best_score; second_move = bestmove;
                best_score = score; bestmove = Some(m);
            } else if score > second_score {
                second_score = score; second_move = Some(m);
            }
            if score > alpha { alpha = score; }
        }

        // Optional root confirm: quick re-search of second-best in SMP-safe profile
        if self.smp_safe {
            if let (Some(_bm), Some(sb)) = (bestmove, second_move) {
                if let Some(dl) = self.deadline { if Instant::now() < dl { 
                    // Short confirm at depth-1
                    let mut child = board.clone(); child.play(sb);
                    let confirm_depth = depth.saturating_sub(1);
                    let sc = -self.alphabeta(&child, confirm_depth, -beta, -alpha, 1, move_index(sb));
                    if sc > best_score + 30 { best_score = sc; bestmove = Some(sb); }
                }}
            }
        }

        // Store root in TT as exact when using full window
        let root_bound = if best_score <= orig_alpha { Bound::Upper } else if best_score >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best_score, bestmove, root_bound);

        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn search_depth_parallel(&mut self, board: &Board, depth: u32) -> SearchResult {
        self.search_depth_parallel_window(board, depth, -MATE_SCORE, MATE_SCORE)
    }

    fn search_depth_parallel_window(&mut self, board: &Board, depth: u32, alpha0: i32, beta0: i32) -> SearchResult {
        use std::sync::atomic::{AtomicI32, Ordering};
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }

        // TT-first (trusted only)
        if self.tt_first {
            if let Some(en) = self.tt_get(board) {
                if let Some(ttm) = en.best {
                    let trusted = matches!(en.bound, Bound::Exact) || en.depth >= depth.saturating_sub(1);
                    if trusted {
                        if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                            let mv = moves.remove(pos);
                            moves.insert(0, mv);
                        }
                    }
                }
            }
        }
        // Root ordering including captures/SEE/check/history/killers
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
                let gives_check_bonus = { let mut c = board.clone(); c.play(m); if !(c.checkers()).is_empty() { 30 } else { 0 } };
                let mi = move_index(m);
                let hist = if self.use_history { self.history_table.get(mi).copied().unwrap_or(0) } else { 0 };
                let kb = if self.use_killers { self.killer_bonus(0, m) } else { 0 };
                -(is_cap * 1000 + mvv + see_b + gives_check_bonus + kb + hist)
            });
        }

        // PV seed serially to establish alpha
        let mut iter = moves.into_iter();
        let first = iter.next().unwrap();
        let mut child = board.clone();
        child.play(first);
        let mut seed = Searcher::default();
        seed.node_limit = u64::MAX;
        seed.deadline = self.deadline;
        seed.order_captures = self.order_captures;
        seed.use_history = self.use_history;
        seed.use_killers = self.use_killers;
        seed.use_lmr = self.use_lmr;
        seed.use_nullmove = self.use_nullmove;
        seed.set_null_min_depth(self.null_min_depth);
        seed.tt = self.tt.clone();
        seed.use_nnue = self.use_nnue;
        if let Some(model) = &self.nnue_quant.as_ref().map(|qn| qn.model.clone()) { seed.nnue_quant = Some(QuantNetwork::new(model.clone())); }
        let mut best_score = -seed.alphabeta(&child, depth - 1, -beta0, -alpha0, 1, move_index(first));
        self.nodes += seed.nodes;
        let mut best_move_local: Option<Move> = Some(first);
        let alpha_shared = AtomicI32::new(best_score);

        // Parallel tail with configurable policy: PVS (narrow then re-search) or full-window
        let deadline = self.deadline;
        let order_captures = self.order_captures;
        let use_history = self.use_history;
        let use_killers = self.use_killers;
        let use_lmr = self.use_lmr;
        let use_null = self.use_nullmove;
        let null_min = self.null_min_depth;
        let use_nnue = self.use_nnue;
        let helper_tt_exact_only = self.helper_tt_exact_only || self.smp_safe;
        // In safe SMP profile, prefer full-window tails for stability
        let tail_policy = if self.smp_safe { TailPolicy::Full } else { self.tail_policy };
        let shared_tt = self.tt.clone();
        let quant_model = self.nnue_quant.as_ref().map(|qn| qn.model.clone());
        let tails: Vec<Move> = iter.collect();
        let results: Vec<(Move, i32, u64)> = tails.par_iter().map(|&m| {
            let mut c = board.clone(); c.play(m);
            let mut w = Searcher::default();
            w.node_limit = u64::MAX;
            w.deadline = deadline;
            w.order_captures = order_captures;
            w.use_history = use_history;
            w.use_killers = use_killers;
            w.use_lmr = use_lmr;
            w.use_nullmove = use_null;
            w.set_null_min_depth(null_min);
            w.tt = shared_tt.clone();
            w.use_nnue = use_nnue;
            w.set_smp_safe(self.smp_safe);
            w.set_helper_tt_exact_only(helper_tt_exact_only);
            w.set_tail_policy(tail_policy);
            if let Some(model) = &quant_model { w.nnue_quant = Some(QuantNetwork::new(model.clone())); }
            let a = alpha_shared.load(Ordering::Relaxed);
            let next_depth = depth - 1;
            let mut sc = match tail_policy {
                TailPolicy::Pvs => {
                    // PVS: try narrow window first; if fail-high, re-search with full window
                    let mut tsc = -w.alphabeta(&c, next_depth, -a - 1, -a, 1, move_index(m));
                    if tsc > a { tsc = -w.alphabeta(&c, next_depth, -beta0, -a, 1, move_index(m)); }
                    tsc
                },
                TailPolicy::Full => {
                    -w.alphabeta(&c, next_depth, -beta0, -a, 1, move_index(m))
                }
            };
            // Update shared alpha
            let mut cur = a;
            while sc > cur { match alpha_shared.compare_exchange(cur, sc, Ordering::Relaxed, Ordering::Relaxed) { Ok(_) => break, Err(obs) => { if obs >= sc { break; } cur = obs; } } }
            (m, sc, w.nodes)
        }).collect();

        for (m, s, n) in results {
            self.nodes += n;
            if s > best_score { best_score = s; best_move_local = Some(m); }
        }
        self.tt_put(board, depth, best_score, best_move_local, Bound::Exact);
        SearchResult { bestmove: best_move_local.map(|mv| format!("{}", mv)), score_cp: best_score, nodes: self.nodes }
    }

    fn alphabeta(&mut self, board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32, parent_move_idx: usize) -> i32 {
        if let Some(ref flag) = self.abort { if flag.load(Ordering::Relaxed) { return self.eval_cp_internal(board); } }
        self.nodes += 1;
        if self.nodes >= self.node_limit { return self.eval_cp_internal(board); }
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval_cp_internal(board); } }
        if depth == 0 { return self.qsearch(board, alpha, beta); }
        // Null-move pruning (guarded)
        // Null-move pruning with shallow-depth verification to avoid tactical misses
        if self.use_nullmove && depth >= self.null_min_depth {
            // avoid in check
            if (board.checkers()).is_empty() {
                // null move
                let nb = board.clone();
                // cozy_chess null move; if unavailable, skip
                let null_ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    nb.null_move();
                })).is_ok();
                if null_ok {
                    let mut r = 2 + (depth / 4) as u32;
                    if self.null_r_bonus > 0 { r = r.saturating_add(self.null_r_bonus as u32); }
                    else if self.null_r_bonus < 0 { r = r.saturating_sub((-self.null_r_bonus) as u32).max(1); }
                    if r >= depth { r = depth - 1; }
                    let score = -self.alphabeta(&nb, depth - 1 - r, -beta, -beta + 1, ply + 1, usize::MAX);
                    if score >= beta {
                        // Verified null-move in safe SMP: a second confirmation with slightly less reduction
                        if self.smp_safe && depth > self.null_min_depth {
                            let r2 = r.saturating_sub(1).max(1);
                            let score2 = -self.alphabeta(&nb, depth - 1 - r2, -beta, -beta + 1, ply + 1, usize::MAX);
                            if score2 >= beta { return score2; }
                        } else {
                            return score;
                        }
                    }
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
                let trusted = matches!(en.bound, Bound::Exact);
                if trusted {
                    if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                        let mv = moves.remove(pos);
                        moves.insert(0, mv);
                    }
                }
            }
        }
        // Ordering including history/killers/counter/cont
        self.order_moves_internal(board, &mut moves, parent_move_idx, ply, depth);

        // In-tree split (jamboree-lite): PV seed + parallel tail with shared alpha
        if self.threads > 1 && !self.smp_safe && depth >= 3 && moves.len() >= 12 {
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
            let helper_tt_exact_only = self.helper_tt_exact_only || self.smp_safe;
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
                w.set_smp_safe(self.smp_safe);
                w.set_helper_tt_exact_only(helper_tt_exact_only);
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
        // Futility pre-eval
        let in_check_now = !(board.checkers()).is_empty();
        let stand_eval = if self.use_futility && !self.smp_safe && depth <= 3 && !in_check_now { Some(self.eval_current(board)) } else { None };
        for (idx, m) in moves.into_iter().enumerate() {
            let is_cap = self.is_capture(board, m);
            let mut child = board.clone();
            child.play(m);
            let gives_check = !(child.checkers()).is_empty();

            // Futility pruning: shallow, non-capture, non-check moves when not currently in check
            if let Some(stand) = stand_eval {
                if !is_cap && !gives_check {
                    let margin = match depth { 1 => 125, 2 => 200, _ => 300 };
                    if stand + margin <= alpha { continue; }
                }
            }

            // Late Move Pruning (LMP): prune tail quiets at shallow depth
            if self.use_lmp && !self.smp_safe && depth <= 3 && !in_check_now && !is_cap && !gives_check {
                let threshold = 3 + (depth as usize) * 2;
                if idx >= threshold { continue; }
            }
            let score = {
                let lmr_depth_gate = if self.smp_safe { 6 } else { 3 };
                let lmr_idx_gate = if self.smp_safe { 5 } else { 3 };
                let mut do_lmr = self.use_lmr;
                if self.smp_safe && ply <= 1 { do_lmr = false; }
                if do_lmr && depth >= lmr_depth_gate {
                    // Conservative LMR: reduce late quiet moves (no captures or checking moves)
                    if !is_cap && !gives_check && idx >= lmr_idx_gate {
                        let mut r = 1u32 + self.lmr_aggr.max(0) as u32;
                        if !self.smp_safe {
                            // Match baseline reductions for 1T: slightly more aggressive on deeper/tail moves
                            if idx >= 6 && depth >= 5 { r += 1; }
                            if idx >= 10 && depth >= 7 { r += 1; }
                        } else {
                            // Conservative cap in SMP-safe
                            if depth < 8 { r = r.min(1); } else { r = r.min(2); }
                        }
                        if r > depth - 1 { r = (depth - 1).min(2); }
                        let red = -self.alphabeta(&child, depth - 1 - r, -alpha - 1, -alpha, ply + 1, move_index(m));
                        if red > alpha { -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1, move_index(m)) } else { red }
                    } else {
                        -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1, move_index(m))
                    }
                } else {
                    -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1, move_index(m))
                }
            };
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
                // bump continuation history on beta-cutoff
                let idx = Self::cont_index(parent_move_idx, mi);
                if let Some(ch) = self.cont_hist.get_mut(idx) { *ch = ch.saturating_add((depth as i32) * (depth as i32)); }
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
        // Only suppress shallow non-Exact writes in SMP-safe mode to improve helper stability.
        // Keep baseline-friendly behavior for 1T/normal runs.
        if self.smp_safe {
            if !self.helper_tt_exact_only {
                if depth <= 3 {
                    if !matches!(bound, Bound::Exact) { return; }
                }
            }
        }
        let b = if self.helper_tt_exact_only { Bound::Exact } else { bound };
        let e = Entry { key: Self::tt_key(board), depth, score, best, bound: b, gen: 0 };
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
        let mut second_move: Option<Move> = None;
        let mut second_score = -MATE_SCORE;

        if self.threads > 1 && depth > 1 { return self.search_depth_parallel_window(board, depth, alpha, beta); }

        if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { qn.refresh(board); } }
        let orig_alpha = alpha;
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                let trusted = matches!(en.bound, Bound::Exact) || en.depth >= depth.saturating_sub(1);
                if trusted {
                    if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                        let mv = moves.remove(pos);
                        moves.insert(0, mv);
                    }
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
            // Optional root-only SEE refinement for the top-K moves
            if self.root_see_top_k > 0 && !moves.is_empty() {
                let k = self.root_see_top_k.min(moves.len());
                let mut prefix: Vec<Move> = moves[..k].to_vec();
                let mut caps: Vec<(Move, i32)> = Vec::new();
                let mut quiets: Vec<Move> = Vec::new();
                for &m in &prefix {
                    let to_sq: Square = m.to;
                    let bit = 1u64 << (to_sq as usize);
                    if (occ_mask & bit) != 0 {
                        let see = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                        caps.push((m, see));
                    } else { quiets.push(m); }
                }
                caps.sort_by_key(|&(_, see)| -see);
                let mut refined: Vec<Move> = caps.into_iter().map(|(m, _)| m).collect();
                refined.extend(quiets.into_iter());
                for i in 0..k { moves[i] = refined[i]; }
            }
        }
        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let mut change = None;
            if self.use_nnue { if let Some(qn) = self.nnue_quant.as_mut() { change = Some(qn.apply_move(board, m, &child)); } }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let mut score = -self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m));
            if score == crate::search::eval::DRAW_SCORE && crate::search::safety::is_stalemate(&child) {
                score -= 1;
            }
            if let Some(ch) = change { if let Some(qn) = self.nnue_quant.as_mut() { qn.revert(ch); } }
            if score > best_score {
                second_score = best_score; second_move = bestmove;
                best_score = score; bestmove = Some(m);
            } else if score > second_score { second_score = score; second_move = Some(m); }
            if score > alpha { alpha = score; }
        }

        if self.smp_safe {
            if let (Some(_bm), Some(sb)) = (bestmove, second_move) {
                if let Some(dl) = self.deadline { if Instant::now() < dl {
                    let mut child = board.clone(); child.play(sb);
                    let confirm_depth = depth.saturating_sub(1);
                    let sc = -self.alphabeta(&child, confirm_depth, -beta, -alpha, 1, move_index(sb));
                    if sc > best_score + 30 { best_score = sc; bestmove = Some(sb); }
                }}
            }
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
    pub fn set_null_min_depth(&mut self, d: u32) { self.null_min_depth = d; }
    pub fn set_hist_min_depth(&mut self, d: u32) { self.hist_min_depth = d; }
    pub fn set_root_see_top_k(&mut self, k: usize) { self.root_see_top_k = k; }

    fn eval_cp_internal(&self, board: &Board) -> i32 { self.eval_current(board) }

    // --- Debug helpers for tests ---
    pub fn debug_move_index(&self, m: Move) -> usize { move_index(m) }
    pub fn debug_order_for_parent(&self, board: &Board, parent_move_idx: usize) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        self.order_moves_internal(board, &mut moves, parent_move_idx, 0, 99);
        moves
    }

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

#[derive(Clone, Copy, Debug)]
pub enum TailPolicy { Pvs, Full }
