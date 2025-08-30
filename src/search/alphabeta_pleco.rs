#![cfg(feature = "board-pleco")]
use pleco::{Board as PlecoBoard, BitMove as PMove, Player, PieceType, Piece};
use crate::search::tt_pleco::{TtPleco, Entry as TtEntry, Bound as TtBound};
use std::sync::Arc;
use std::time::{Duration, Instant};
use rayon::prelude::*;
use std::time::Duration as StdDuration;
use crate::search::eval::{MATE_SCORE, DRAW_SCORE};
use crate::search::see_pleco;

pub struct PlecoSearcher {
    nodes: u64,
    deadline: Option<Instant>,
    tt: Arc<TtPleco>,
    killers: Vec<[Option<PMove>; 2]>,
    history: Vec<i32>,
    counter_move: Vec<usize>,
    cont_hist: Vec<i32>,
    threads: usize,
    use_killers: bool,
    use_lmr: bool,
    use_nullmove: bool,
    use_aspiration: bool,
    aspiration_window_cp: i32,
    last_depth: u32,
    abort: Option<Arc<std::sync::atomic::AtomicBool>>,
    smp_mode: SmpMode,
    // Heuristic knobs for diversification (used by Lazy SMP helpers)
    lmr_aggr: i32,          // extra LMR reduction for helpers
    null_r_bonus: i32,      // extra null-move reduction R for helpers
    tt_first: bool,         // whether to hoist TT move to front
    order_offset: usize,    // rotate tail by offset to diversify ordering
    helper_mode: bool,      // enables aggressive helper-only pruning (LMP/Futility)
    max_seldepth: u32,      // deepest ply reached (selective depth)
    tm_finish_one: bool,    // time manager policy: true = finish-one-depth, false = spend budget
    tm_factor: f32,         // multiplier for predicting next iteration cost
    see_prune: bool,        // enable SEE-based pruning in qsearch
    see_ordering: bool,     // enable SEE in capture ordering
    se_enable: bool,        // enable TT-based singular extensions
    singular_margin_cp: i32,// margin for singularity test
    singular_hits: u32,     // number of times SE triggered (debug)
    debug_force_singular: bool,
    see_ordering_topk: usize,
    iid_strong: bool,
    eval_mode: PlecoEvalMode,
    hybrid_prewarm: bool,   // enable helper pre-warm in LazyHybrid (default off)
    asp_hint: Option<i32>,  // aspiration last score hint carried across hybrid iterations
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmpMode { Off, InTree, LazyIndep, LazyCoop, LazyHybrid }

impl Default for PlecoSearcher { fn default() -> Self { Self { nodes: 0, deadline: None, tt: Arc::new(TtPleco::default()), killers: vec![[None,None];256], history: vec![0; 64*64*5], counter_move: vec![usize::MAX; 64*64*5], cont_hist: vec![0; 1<<18], threads: 1, use_killers: true, use_lmr: true, use_nullmove: true, use_aspiration: true, aspiration_window_cp: 30, last_depth: 0, abort: None, smp_mode: SmpMode::InTree, lmr_aggr: 0, null_r_bonus: 0, tt_first: true, order_offset: 0, helper_mode: false, max_seldepth: 0, tm_finish_one: true, tm_factor: 1.9, see_prune: true, see_ordering: true, se_enable: true, singular_margin_cp: 32, singular_hits: 0, debug_force_singular: false, see_ordering_topk: 6, iid_strong: true, eval_mode: PlecoEvalMode::Material, hybrid_prewarm: false, asp_hint: None } } }

impl PlecoSearcher {
    pub fn clear(&mut self) { self.nodes = 0; self.killers.iter_mut().for_each(|k| *k = [None, None]); self.history.fill(0); self.counter_move.fill(usize::MAX); self.cont_hist.fill(0); self.tt.bump_generation(); }
    pub fn set_tt_capacity_mb(&mut self, mb: usize) { Arc::get_mut(&mut self.tt).map(|t| t.set_capacity_mb(mb)); }
    pub fn set_threads(&mut self, t: usize) { self.threads = t.max(1); }
    pub fn last_depth(&self) -> u32 { self.last_depth }
    pub fn set_smp_mode(&mut self, m: SmpMode) { self.smp_mode = m; }
    pub fn last_seldepth(&self) -> u32 { self.max_seldepth }
    pub fn set_time_manager(&mut self, finish_one: bool, factor: f32) { self.tm_finish_one = finish_one; self.tm_factor = if factor > 0.1 { factor } else { 1.9 }; }
    pub fn set_see_prune(&mut self, on: bool) { self.see_prune = on; }
    pub fn set_see_ordering(&mut self, on: bool) { self.see_ordering = on; }
    pub fn set_see_ordering_topk(&mut self, k: usize) { self.see_ordering_topk = k; }
    pub fn set_singular_enable(&mut self, on: bool) { self.se_enable = on; }
    pub fn set_singular_margin(&mut self, margin: i32) { self.singular_margin_cp = margin; }
    pub fn set_iid_strong(&mut self, on: bool) { self.iid_strong = on; }
    pub fn debug_set_force_singular(&mut self, on: bool) { self.debug_force_singular = on; }
    pub fn debug_singular_hits(&self) -> u32 { self.singular_hits }
    pub fn set_eval_mode(&mut self, mode: PlecoEvalMode) { self.eval_mode = mode; }

    #[inline]
    fn debug_force_singular_active(&self) -> bool {
        #[cfg(test)]
        { return self.debug_force_singular; }
        #[allow(unreachable_code)]
        { false }
    }

    #[inline]
    fn cont_index(&self, parent_idx: usize, child_idx: usize) -> usize {
        let key = (parent_idx as u64).wrapping_mul(1_000_003) ^ (child_idx as u64).wrapping_mul(0x9E37_79B9);
        (key as usize) & ((1<<18) - 1)
    }

    pub fn search_movetime(&mut self, board: &mut PlecoBoard, millis: u64, depth: u32) -> (Option<PMove>, i32, u64) {
        match self.smp_mode {
            SmpMode::LazyCoop if self.threads > 1 => return self.search_movetime_lazy_coop(board, millis, depth),
            SmpMode::LazyIndep if self.threads > 1 => return self.search_movetime_lazy(board, millis, depth),
            SmpMode::LazyHybrid if self.threads > 1 => return self.search_movetime_lazy_hybrid(board, millis, depth),
            _ => {}
        }
        self.nodes = 0;
        self.deadline = Some(Instant::now() + Duration::from_millis(millis));
        self.abort = Some(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        self.max_seldepth = 0;
        let mut best: Option<PMove> = None; let mut best_score = -MATE_SCORE;
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut last_score = 0;
        let mut last_iter_time = Duration::from_millis(0);
        for d in 1..=max_depth {
            self.tt.bump_generation();
            let iter_start = Instant::now();
            let (bm, sc) = if self.use_aspiration && d > 1 {
                let window = self.aspiration_window_cp.max(10);
                let alpha = last_score - window;
                let beta = last_score + window;
                let (b1, s1) = self.root_iter_window(board, d, alpha, beta);
                if s1 <= alpha || s1 >= beta { self.root_iter(board, d) } else { (b1, s1) }
            } else {
                self.root_iter(board, d)
            };
            best = bm; best_score = sc; last_score = sc;
            self.last_depth = d;
            last_iter_time = iter_start.elapsed();
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_score, self.nodes)
    }

    // Cooperative Lazy SMP: partition root move list across workers per iteration
    fn search_movetime_lazy_coop(&mut self, board: &mut PlecoBoard, millis: u64, depth: u32) -> (Option<PMove>, i32, u64) {
        self.nodes = 0;
        if millis > 0 {
            self.deadline = Some(Instant::now() + Duration::from_millis(millis));
        }
        if self.abort.is_none() {
            self.abort = Some(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        }
        self.max_seldepth = 0;
        let mut best: Option<PMove> = None; let mut best_score = self.asp_hint.unwrap_or(-MATE_SCORE);
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut last_score = 0;
        let mut last_iter_time = Duration::from_millis(0);
        for d in 1..=max_depth {
            self.tt.bump_generation();
            // In lazy-coop, prefer completing the iteration
            let iter_start = Instant::now();
            // Build and order root moves
            let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
            if ml.is_empty() { break; }
            let tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
            self.order_moves(board, &mut ml, tt_best, 0, None);
            // Seed PV with first move (TT-best) to raise alpha early, with aspiration window
            use std::sync::atomic::{AtomicI32, Ordering};
            let shared_tt = self.tt.clone();
            let pv = ml[0];
            let mut b1 = board.clone(); b1.apply_move(pv);
            let mut seed = Self::default();
            seed.tt = shared_tt.clone();
            seed.threads = 1; seed.use_killers = self.use_killers; seed.use_lmr = self.use_lmr; seed.use_nullmove = self.use_nullmove; seed.use_aspiration = self.use_aspiration; seed.aspiration_window_cp = self.aspiration_window_cp; seed.deadline = self.deadline; seed.smp_mode = SmpMode::Off;
            let use_asp = self.use_aspiration && d > 1;
            let window = self.aspiration_window_cp.max(10);
            let mut asp_alpha = -MATE_SCORE; let mut asp_beta = MATE_SCORE;
            if use_asp { asp_alpha = best_score - window; asp_beta = best_score + window; }
            let mut pv_sc = -seed.alphabeta(&mut b1, d.saturating_sub(1), -asp_beta, -asp_alpha, 1, Some(self.move_hist_index(pv)));
            self.nodes += seed.nodes; if seed.max_seldepth > self.max_seldepth { self.max_seldepth = seed.max_seldepth; }
            if use_asp && (pv_sc <= asp_alpha || pv_sc >= asp_beta) {
                let mut b2 = board.clone(); b2.apply_move(pv);
                let mut seed2 = Self::default();
                seed2.tt = shared_tt.clone();
                seed2.threads = 1; seed2.use_killers = self.use_killers; seed2.use_lmr = self.use_lmr; seed2.use_nullmove = self.use_nullmove; seed2.use_aspiration = self.use_aspiration; seed2.aspiration_window_cp = self.aspiration_window_cp; seed2.deadline = self.deadline; seed2.smp_mode = SmpMode::Off;
                pv_sc = -seed2.alphabeta(&mut b2, d.saturating_sub(1), -MATE_SCORE, MATE_SCORE, 1, Some(self.move_hist_index(pv)));
                self.nodes += seed2.nodes;
            }
            let alpha_shared = AtomicI32::new(pv_sc);
            // Dynamic chunking and stronger early cancellation
            let tails: Vec<PMove> = ml.into_iter().skip(1).collect();
            let abort_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            // Use finer-grained chunks to improve completion odds under deadline
            let gran = (self.threads * 4).max(1);
            let chunk = (tails.len() + gran - 1) / gran;
            let results: Vec<(PMove, i32, u64, u32)> = tails.par_chunks(chunk.max(1)).flat_map(|chunk_moves| {
                let mut out = Vec::with_capacity(chunk_moves.len());
                for &m in chunk_moves {
                    if abort_flag.load(std::sync::atomic::Ordering::Relaxed) { break; }
                    // Abort very late to allow iteration completion more often
                    if let Some(dl) = self.deadline { if dl.saturating_duration_since(Instant::now()) < StdDuration::from_millis(1) { abort_flag.store(true, std::sync::atomic::Ordering::Relaxed); break; } }
                    let mut c = board.clone(); c.apply_move(m);
                    let mut w = Self::default();
                    w.tt = shared_tt.clone();
                    w.threads = 1; w.use_killers = self.use_killers; w.use_lmr = self.use_lmr; w.use_nullmove = self.use_nullmove; w.use_aspiration = self.use_aspiration; w.aspiration_window_cp = self.aspiration_window_cp + 10; w.deadline = self.deadline; w.tm_finish_one = self.tm_finish_one; w.tm_factor = self.tm_factor; w.smp_mode = SmpMode::Off;
                    let a = alpha_shared.load(Ordering::Relaxed);
                    let sc = if use_asp { -w.alphabeta(&mut c, d.saturating_sub(1), -asp_beta, -a, 1, Some(self.move_hist_index(m))) } else { -w.alphabeta(&mut c, d.saturating_sub(1), -MATE_SCORE, -a, 1, Some(self.move_hist_index(m))) };
                    let mut cur = a;
                    while sc > cur {
                        match alpha_shared.compare_exchange(cur, sc, Ordering::Relaxed, Ordering::Relaxed) {
                            Ok(_) => break,
                            Err(obs) => { if obs >= sc { break; } cur = obs; }
                        }
                    }
                    out.push((m, sc, w.nodes, w.max_seldepth));
                }
                out
            }).collect();
            // Reduce results
            for (m, sc, n, sd) in results { self.nodes += n; if sc > best_score { best_score = sc; best = Some(m); } if sd > self.max_seldepth { self.max_seldepth = sd; } }
            let completed = self.deadline.map(|dl| Instant::now() < dl).unwrap_or(true);
            if completed { self.last_depth = d; }
            last_iter_time = iter_start.elapsed();
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_score, self.nodes)
    }

    // Hybrid Lazy SMP: use independent helper to pre-warm TT at depth d+2, then LazyCoop to complete depth d
    fn search_movetime_lazy_hybrid(&mut self, board: &mut PlecoBoard, millis: u64, depth: u32) -> (Option<PMove>, i32, u64) {
        self.nodes = 0;
        self.deadline = Some(Instant::now() + Duration::from_millis(millis));
        self.abort = Some(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        self.max_seldepth = 0;
        let mut best: Option<PMove> = None; let mut best_score = -MATE_SCORE;
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut last_iter_time = Duration::from_millis(0);
        self.asp_hint = None;
        for d in 1..=max_depth {
            self.tt.bump_generation();
            let iter_start = Instant::now();
            // Pre-warm TT via an independent helper search at depth d+2 for a small time slice
            if self.hybrid_prewarm {
                if let Some(dl) = self.deadline {
                    // Only pre-warm early to avoid starving later iterations
                    if d <= 4 {
                        let remaining = dl.saturating_duration_since(Instant::now());
                        let slice = (remaining.as_millis() as u64 / 8).min(50).max(10);
                        let shared_tt = self.tt.clone();
                        let mut helper = Self::default();
                        helper.tt = shared_tt.clone();
                        helper.threads = 1; helper.use_killers = self.use_killers; helper.use_lmr = true; helper.use_nullmove = true; helper.use_aspiration = true; helper.aspiration_window_cp = self.aspiration_window_cp + 20; helper.deadline = Some(Instant::now() + Duration::from_millis(slice)); helper.tm_finish_one = false; helper.tm_factor = self.tm_factor; helper.smp_mode = SmpMode::Off; helper.lmr_aggr = 1; helper.null_r_bonus = 1; helper.helper_mode = true;
                        let _ = helper.search_movetime(&mut board.clone(), slice, d.saturating_add(2));
                        self.nodes += helper.nodes;
                    }
                }
            }
            // Complete current depth cooperatively
            let (bm, sc, nodes) = self.search_movetime_lazy_coop(board, 0, d); // uses existing deadline
            self.nodes += nodes;
            if let Some(m) = bm { best = Some(m); best_score = sc; }
            self.asp_hint = Some(best_score);
            self.last_depth = d;
            last_iter_time = iter_start.elapsed();
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_score, self.nodes)
    }

    fn search_movetime_lazy(&mut self, board: &mut PlecoBoard, millis: u64, depth: u32) -> (Option<PMove>, i32, u64) {
        let shared_tt = self.tt.clone();
        let threads = self.threads;
        let max_depth = if depth == 0 { 99 } else { depth };
        let deadline = Some(Instant::now() + Duration::from_millis(millis));
        let results: Vec<(usize, Option<PMove>, i32, u64, u32, u32)> = (0..threads).into_par_iter().map(|wid| {
            let mut w = Self::default();
            w.tt = shared_tt.clone();
            w.threads = 1;
            w.use_killers = self.use_killers;
            w.use_lmr = self.use_lmr;
            w.use_nullmove = self.use_nullmove;
            w.use_aspiration = self.use_aspiration;
            // Diversify aspiration window, LMR, null move, and ordering
            w.aspiration_window_cp = self.aspiration_window_cp + (wid as i32 % 3) * 20;
            if wid > 0 { w.lmr_aggr = 1 + ((wid as i32) % 2); w.null_r_bonus = 1; w.tt_first = (wid % 2) == 0; w.order_offset = wid as usize; w.helper_mode = true; }
            w.deadline = deadline;
            w.smp_mode = SmpMode::Off;
            let mut b = board.clone();
            let (bm, sc, nodes) = w.search_movetime(&mut b, millis, max_depth);
            (wid, bm, sc, nodes, w.last_depth(), w.last_seldepth())
        }).collect();
        // Choose the deepest worker; break ties preferring worker 0, then higher score
        let mut best = results[0].clone();
        for r in &results {
            if r.4 > best.4 || (r.4 == best.4 && r.0 == 0 && best.0 != 0) { best = r.clone(); }
        }
        self.nodes = results.iter().map(|r| r.3).sum();
        self.last_depth = best.4; self.max_seldepth = results.iter().map(|r| r.5).max().unwrap_or(0);
        (best.1, best.2, self.nodes)
    }

    #[inline]
    fn killer_bonus(&self, ply: usize, m: PMove) -> i32 {
        if !self.use_killers { return 0; }
        let p = ply.min(self.killers.len()-1);
        let slot = &self.killers[p];
        if slot[0] == Some(m) { 50 } else if slot[1] == Some(m) { 30 } else { 0 }
    }

    fn order_moves(&self, board: &PlecoBoard, moves: &mut Vec<PMove>, tt_best: Option<PMove>, ply: usize, parent_idx: Option<usize>) {
        if self.tt_first {
            if let Some(ttm) = tt_best { if let Some(pos) = moves.iter().position(|&x| x == ttm) { let mv = moves.remove(pos); moves.insert(0, mv); } }
        }
        if moves.len() <= 1 { return; }
        let cm_target = parent_idx.and_then(|pi| self.counter_move.get(pi).copied()).unwrap_or(usize::MAX);
        let topk = self.see_ordering_topk;
        let mut cap_scores: Vec<(PMove, i32)> = Vec::new();
        if self.see_ordering && topk > 0 {
            for &m in moves.iter().skip(1) { if m.is_capture() { cap_scores.push((m, self.mvv_lva(board, m))); } }
            cap_scores.sort_by_key(|&(_, sc)| -sc);
        }
        let mut topk_vec: Vec<PMove> = Vec::new();
        if self.see_ordering && topk > 0 {
            for &(m, _) in cap_scores.iter().take(topk.min(cap_scores.len())) { topk_vec.push(m); }
        }
        moves[1..].sort_by_key(|&m| {
            let cap = if m.is_capture() { 1 } else { 0 };
            let mvv = if cap == 1 { self.mvv_lva(board, m) } else { 0 };
            let hist = self.history_score(m);
            let kb = self.killer_bonus(ply, m);
            let mi = self.move_hist_index(m);
            let cm = if cm_target != usize::MAX && mi == cm_target { 200 } else { 0 };
            let cont = if let Some(pi) = parent_idx { self.cont_hist[self.cont_index(pi, mi)] } else { 0 };
            let see_b = if cap == 1 && self.see_ordering {
                if topk == 0 || topk_vec.iter().any(|&x| x == m) {
                    if let Some(g) = see_pleco::see_gain_cp(board, m) { g / 8 } else { 0 }
                } else { 0 }
            } else { 0 };
            -(cap * 10 + kb + hist + mvv + cm + cont + see_b)
        });
        // Diversification: rotate tail by offset
        if self.order_offset > 0 && moves.len() > 2 {
            let tail = &mut moves[1..];
            let k = self.order_offset % tail.len();
            tail.rotate_left(k);
        }
    }

    #[inline]
    fn history_score(&self, m: PMove) -> i32 {
        let idx = self.move_hist_index(m);
        *self.history.get(idx).unwrap_or(&0)
    }

    #[inline]
    fn move_hist_index(&self, m: PMove) -> usize {
        // Use direct square indices (0..63) instead of formatting strings
        fn sq_index(sq: pleco::SQ) -> usize {
            use pleco::SQ;
            match sq {
                SQ::A1=>0,SQ::B1=>1,SQ::C1=>2,SQ::D1=>3,SQ::E1=>4,SQ::F1=>5,SQ::G1=>6,SQ::H1=>7,
                SQ::A2=>8,SQ::B2=>9,SQ::C2=>10,SQ::D2=>11,SQ::E2=>12,SQ::F2=>13,SQ::G2=>14,SQ::H2=>15,
                SQ::A3=>16,SQ::B3=>17,SQ::C3=>18,SQ::D3=>19,SQ::E3=>20,SQ::F3=>21,SQ::G3=>22,SQ::H3=>23,
                SQ::A4=>24,SQ::B4=>25,SQ::C4=>26,SQ::D4=>27,SQ::E4=>28,SQ::F4=>29,SQ::G4=>30,SQ::H4=>31,
                SQ::A5=>32,SQ::B5=>33,SQ::C5=>34,SQ::D5=>35,SQ::E5=>36,SQ::F5=>37,SQ::G5=>38,SQ::H5=>39,
                SQ::A6=>40,SQ::B6=>41,SQ::C6=>42,SQ::D6=>43,SQ::E6=>44,SQ::F6=>45,SQ::G6=>46,SQ::H6=>47,
                SQ::A7=>48,SQ::B7=>49,SQ::C7=>50,SQ::D7=>51,SQ::E7=>52,SQ::F7=>53,SQ::G7=>54,SQ::H7=>55,
                SQ::A8=>56,SQ::B8=>57,SQ::C8=>58,SQ::D8=>59,SQ::E8=>60,SQ::F8=>61,SQ::G8=>62,SQ::H8=>63,
                _ => 0,
            }
        }
        let from = sq_index(m.get_src());
        let to = sq_index(m.get_dest());
        let pi = match m.promo_piece() { PieceType::N => 1, PieceType::B => 2, PieceType::R => 3, PieceType::Q => 4, _ => 0 };
        (from * 64 + to) * 5 + pi
    }

    #[inline]
    fn piece_value_cp(pt: PieceType) -> i32 {
        match pt { PieceType::P => 100, PieceType::N => 320, PieceType::B => 330, PieceType::R => 500, PieceType::Q => 900, _ => 0 }
    }

    #[inline]
    fn mvv_lva(&self, board: &PlecoBoard, m: PMove) -> i32 {
        let to = m.get_dest(); let from = m.get_src();
        let v_piece = board.piece_at_sq(to);
        let a_piece = board.piece_at_sq(from);
        let v = if v_piece != Piece::None { Self::piece_value_cp(v_piece.type_of()) } else { 0 };
        let a = if a_piece != Piece::None { Self::piece_value_cp(a_piece.type_of()) } else { 0 };
        v * 10 - a
    }

    fn root_iter(&mut self, board: &mut PlecoBoard, depth: u32) -> (Option<PMove>, i32) {
        self.root_iter_window(board, depth, -MATE_SCORE, MATE_SCORE)
    }

    fn root_iter_window(&mut self, board: &mut PlecoBoard, depth: u32, alpha0: i32, beta: i32) -> (Option<PMove>, i32) {
        let mut alpha = alpha0;
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        if ml.is_empty() { return (None, self.eval_terminal(board)); }
        let tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        self.order_moves(board, &mut ml, tt_best, 0, None);
        // Root SMP split (in-tree SMP only; split only when heavy and time allows)
        if self.smp_mode == SmpMode::InTree && self.threads > 1 {
            let heavy = depth >= 5 && ml.len() >= 16;
            let time_ok = self.deadline.map(|dl| dl.saturating_duration_since(Instant::now()) > StdDuration::from_millis(50)).unwrap_or(true);
            if heavy && time_ok {
            let shared_tt = self.tt.clone();
            // PV seed: first move
            let first = ml[0];
            let mut b1 = board.clone(); b1.apply_move(first);
            let mut seed = Self { tt: shared_tt.clone(), ..Self::default() };
            seed.threads = 1; seed.use_killers = self.use_killers; seed.use_lmr = self.use_lmr; seed.use_nullmove = self.use_nullmove; seed.use_aspiration = self.use_aspiration; seed.aspiration_window_cp = self.aspiration_window_cp; seed.deadline = self.deadline; seed.smp_mode = SmpMode::Off;
            let abort_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            seed.abort = Some(abort_flag.clone());
            let mut best_sc = -seed.alphabeta(&mut b1, depth - 1, -beta, -alpha, 1, Some(self.move_hist_index(first)));
            self.nodes += seed.nodes;
            let mut best = first;
            let tails: Vec<PMove> = ml.into_iter().skip(1).collect();
            use std::sync::atomic::{AtomicI32, Ordering};
            let alpha_shared = AtomicI32::new(best_sc);
            let results: Vec<(PMove, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone(); c.apply_move(m);
                let mut w = Self { tt: shared_tt.clone(), ..Self::default() };
                w.threads = 1; w.use_killers = self.use_killers; w.use_lmr = self.use_lmr; w.use_nullmove = self.use_nullmove; w.use_aspiration = self.use_aspiration; w.aspiration_window_cp = self.aspiration_window_cp; w.deadline = self.deadline; w.abort = Some(abort_flag.clone()); w.smp_mode = SmpMode::Off;
                let a = alpha_shared.load(Ordering::Relaxed);
                let score = -w.alphabeta(&mut c, depth - 1, -beta, -a, 1, Some(self.move_hist_index(m)));
                // update alpha
                let mut cur = a;
                while score > cur {
                    match alpha_shared.compare_exchange(cur, score, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(obs) => { if obs >= score { break; } cur = obs; }
                    }
                }
                if score >= beta { abort_flag.store(true, Ordering::Relaxed); }
                (m, score, w.nodes)
            }).collect();
            for (m, s, n) in results { self.nodes += n; if s > best_sc { best_sc = s; best = m; } }
            // Store root exact
            self.tt.put(TtEntry { key: board.zobrist(), depth, score: best_sc, best: Some(best), bound: TtBound::Exact, gen: 0 });
            return (Some(best), best_sc);
            }
        }
        // Serial
        let mut best: Option<PMove> = None; let mut best_sc = -MATE_SCORE;
        for m in ml.iter() {
            board.apply_move(*m);
            let sc = -self.alphabeta(board, depth.saturating_sub(1), -beta, -alpha, 1, Some(self.move_hist_index(*m)));
            board.undo_move();
            if sc > best_sc { best_sc = sc; best = Some(*m); }
            if sc > alpha { alpha = sc; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        (best, best_sc)
    }

    // Evaluate all legal root moves at a fixed depth and return scores (higher is better)
    pub fn score_root_moves(&mut self, board: &mut PlecoBoard, depth: u32) -> Vec<(PMove, i32)> {
        let mut out: Vec<(PMove, i32)> = Vec::new();
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        if ml.is_empty() { return out; }
        let tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        self.order_moves(board, &mut ml, tt_best, 0, None);
        for &m in ml.iter() {
            board.apply_move(m);
            let sc = -self.alphabeta(board, depth.saturating_sub(1), -MATE_SCORE, MATE_SCORE, 1, Some(self.move_hist_index(m)));
            board.undo_move();
            out.push((m, sc));
        }
        out.sort_by_key(|&(_, sc)| -sc);
        out
    }

    fn alphabeta(&mut self, board: &mut PlecoBoard, depth: u32, mut alpha: i32, beta: i32, ply: u32, parent_idx: Option<usize>) -> i32 {
        self.nodes += 1;
        if ply > self.max_seldepth { self.max_seldepth = ply; }
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval(board); } }
        if let Some(ref f) = self.abort { if f.load(std::sync::atomic::Ordering::Relaxed) { return self.eval(board); } }
        if depth == 0 { return self.qsearch(board, alpha, beta, ply); }
        // Null-move pruning
        if self.use_nullmove && depth >= 3 && !board.in_check() {
            let mut nb = board.clone();
            // Pleco supports null moves via apply_null_move/undo_null_move if available
            let did_null = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { nb.apply_null_move(); })).is_ok();
            if did_null {
                let mut r = 2 + (depth / 4) as u32;
                if self.null_r_bonus > 0 { r = r.saturating_add(self.null_r_bonus as u32); }
                if r >= depth { r = depth - 1; }
                let sc = -self.alphabeta(&mut nb, depth - 1 - r, -beta, -beta + 1, ply + 1, None);
                if sc >= beta { return sc; }
            }
        }
        // TT probe
        if let Some(e) = self.tt.get(board.zobrist()) {
            if e.depth >= depth { match e.bound { TtBound::Exact => return e.score, TtBound::Lower => if e.score >= beta { return e.score; }, TtBound::Upper => if e.score <= alpha { return e.score; } } }
        }
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        if ml.is_empty() { return self.eval_terminal(board); }
        // Internal Iterative Deepening: if no TT move and depth is sufficient, do a shallow search to fill TT
        let mut tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        if tt_best.is_none() && depth >= 3 && ply <= 2 {
            let _ = self.alphabeta(board, depth - 2, alpha, beta, ply, parent_idx);
            tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        }
        // Stronger IID: broaden trigger
        if self.iid_strong && tt_best.is_none() && depth >= 4 && ply <= 4 {
            let _ = self.alphabeta(board, depth - 3, alpha, beta, ply, parent_idx);
            tt_best = self.tt.get(board.zobrist()).and_then(|e| e.best);
        }
        // TT-based singular extension test for TT move
        let mut tt_is_singular = false;
        if self.se_enable {
            if self.debug_force_singular_active() { tt_is_singular = true; self.singular_hits = self.singular_hits.saturating_add(1); }
            else if let Some(ttm) = tt_best {
                if depth >= 5 {
                    if ml.len() <= 1 { tt_is_singular = true; } else {
                        let sing_beta = alpha.saturating_add(self.singular_margin_cp);
                        let red = 2u32.min(depth - 1);
                        let mut all_fail = true;
                        for &m in ml.iter() {
                            if m == ttm { continue; }
                            board.apply_move(m);
                            let sc = -self.alphabeta(board, depth - 1 - red, -sing_beta, -sing_beta + 1, ply + 1, Some(self.move_hist_index(m)));
                            board.undo_move();
                            if sc >= sing_beta { all_fail = false; break; }
                        }
                        if all_fail { tt_is_singular = true; self.singular_hits = self.singular_hits.saturating_add(1); }
                    }
                }
            }
        }
        self.order_moves(board, &mut ml, tt_best, (self.killers.len()-1).min(depth as usize), parent_idx);
        // In-tree split (jamboree-lite): PV seed + parallel tail
        if self.threads > 1 && depth >= 3 && ml.len() >= 12 {
            let shared_tt = self.tt.clone();
            // PV seed
            let first = ml[0];
            let mut b1 = board.clone(); b1.apply_move(first);
            let mut seed = Self { tt: shared_tt.clone(), ..Self::default() };
            seed.threads = 1; seed.use_killers = self.use_killers; seed.use_lmr = self.use_lmr; seed.use_nullmove = self.use_nullmove; seed.use_aspiration = self.use_aspiration; seed.aspiration_window_cp = self.aspiration_window_cp; seed.deadline = self.deadline;
            let mut best = -seed.alphabeta(&mut b1, depth - 1, -beta, -alpha, ply + 1, Some(self.move_hist_index(first)));
            self.nodes += seed.nodes;
            let mut best_move_local: Option<PMove> = Some(first);
            use std::sync::atomic::{AtomicI32, Ordering};
            let alpha_shared = AtomicI32::new(best);
            let tails: Vec<PMove> = ml.into_iter().skip(1).collect();
            let abort_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let results: Vec<(PMove, i32, u64)> = tails.par_iter().map(|&m| {
                let mut c = board.clone(); c.apply_move(m);
                let mut w = Self { tt: shared_tt.clone(), ..Self::default() };
                w.threads = 1; w.use_killers = self.use_killers; w.use_lmr = self.use_lmr; w.use_nullmove = self.use_nullmove; w.use_aspiration = self.use_aspiration; w.aspiration_window_cp = self.aspiration_window_cp; w.deadline = self.deadline; w.abort = Some(abort_flag.clone());
                let a = alpha_shared.load(Ordering::Relaxed);
                let sc = -w.alphabeta(&mut c, depth - 1, -beta, -a, ply + 1, Some(self.move_hist_index(m)));
                // update alpha
                let mut cur = a;
                while sc > cur {
                    match alpha_shared.compare_exchange(cur, sc, Ordering::Relaxed, Ordering::Relaxed) {
                        Ok(_) => break,
                        Err(obs) => { if obs >= sc { break; } cur = obs; }
                    }
                }
                if sc >= beta { abort_flag.store(true, std::sync::atomic::Ordering::Relaxed); }
                (m, sc, w.nodes)
            }).collect();
            for (m, s, n) in results { self.nodes += n; if s > best { best = s; best_move_local = Some(m); } }
            // Store
            self.tt.put(TtEntry { key: board.zobrist(), depth, score: best, best: best_move_local, bound: TtBound::Exact, gen: 0 });
            return best;
        }

        let mut bestmove: Option<PMove> = None;
        for (i, m) in ml.iter().enumerate() {
            // Helper-only pruning: Late Move Pruning (LMP) and Futility for quiets at small depth
            let is_cap = m.is_capture();
            if self.helper_mode && !is_cap && depth <= 3 && (i >= (if depth >= 3 { 6 } else if depth == 2 { 8 } else { 10 })) && !board.in_check() {
                continue;
            }
            if self.helper_mode && !is_cap && depth <= 2 && !board.in_check() {
                let stand = self.eval(board);
                let margin = 100 * depth as i32;
                if stand + margin <= alpha { continue; }
            }
            board.apply_move(*m);
            // Singular extension: extend TT move when determined singular
            let extend = if tt_best.is_some() && Some(*m) == tt_best && tt_is_singular { 1 } else { 0 };
            let sc = if self.use_lmr && depth >= 3 && !m.is_capture() && i >= 3 && extend == 0 {
                let base_red = 1 + self.lmr_aggr.max(0) as u32;
                let mut red_d = if depth >= 6 { base_red + 1 } else { base_red };
                if red_d >= depth { red_d = depth - 1; }
                let red = -self.alphabeta(board, depth - 1 - red_d, -alpha - 1, -alpha, ply + 1, Some(self.move_hist_index(*m)));
                if red > alpha { -self.alphabeta(board, depth - 1, -beta, -alpha, ply + 1, Some(self.move_hist_index(*m))) } else { red }
            } else {
                -self.alphabeta(board, depth - 1 + extend, -beta, -alpha, ply + 1, Some(self.move_hist_index(*m)))
            };
            board.undo_move();
            if sc >= beta {
                self.tt.put(TtEntry { key: board.zobrist(), depth, score: sc, best: Some(*m), bound: TtBound::Lower, gen: 0 });
                if self.use_killers {
                    let ply = (self.killers.len()-1).min(depth as usize);
                    let k = &mut self.killers[ply]; if k[0] != Some(*m) { k[1] = k[0]; k[0] = Some(*m); }
                }
                // Update history/continuation for quiet beta-cut moves
                if !m.is_capture() {
                    let mi = self.move_hist_index(*m);
                    if let Some(h) = self.history.get_mut(mi) { *h += (depth as i32) * (depth as i32); }
                    if let Some(pi) = parent_idx {
                        if let Some(slot) = self.counter_move.get_mut(pi) { *slot = mi; }
                        let idx = self.cont_index(pi, mi);
                        if let Some(ch) = self.cont_hist.get_mut(idx) { *ch += (depth as i32) * (depth as i32); }
                    }
                }
                return beta;
            }
            if sc > alpha { alpha = sc; bestmove = Some(*m); }
        }
        let bound = if bestmove.is_some() { TtBound::Exact } else { TtBound::Upper };
        self.tt.put(TtEntry { key: board.zobrist(), depth, score: alpha, best: bestmove, bound, gen: 0 });
        alpha
    }

    fn qsearch(&mut self, board: &mut PlecoBoard, mut alpha: i32, beta: i32, ply: u32) -> i32 {
        if ply > self.max_seldepth { self.max_seldepth = ply; }
        let stand = self.eval(board);
        if stand >= beta { return beta; }
        if stand > alpha { alpha = stand; }
        let mut caps: Vec<PMove> = board.generate_moves().iter().copied().filter(|m| m.is_capture()).collect();
        caps.sort_by_key(|&m| -self.mvv_lva(board, m));
        for m in caps.into_iter() {
            // SEE-based pruning: skip clearly losing captures when enabled
            if self.see_prune {
                if let Some(gain) = see_pleco::see_gain_cp(board, m) { if gain < 0 { continue; } }
            }
            board.apply_move(m);
            let sc = -self.qsearch(board, -beta, -alpha, ply + 1);
            board.undo_move();
            if sc >= beta { return beta; }
            if sc > alpha { alpha = sc; }
        }
        alpha
    }

    fn eval(&self, board: &PlecoBoard) -> i32 {
        match self.eval_mode {
            PlecoEvalMode::Material => self.eval_material(board),
            PlecoEvalMode::Pst => self.eval_pst(board),
        }
    }

    #[inline]
    fn eval_material(&self, board: &PlecoBoard) -> i32 {
        let mut score = 0i32;
        for &(p, v) in &[(PieceType::P,100),(PieceType::N,320),(PieceType::B,330),(PieceType::R,500),(PieceType::Q,900)] {
            score += board.count_piece(Player::White, p) as i32 * v;
            score -= board.count_piece(Player::Black, p) as i32 * v;
        }
        if board.turn() == Player::White { score } else { -score }
    }

    #[inline]
    fn eval_pst(&self, board: &PlecoBoard) -> i32 {
        // Material + PST mirroring cozy eval
        let mut score = 0i32;
        // Material
        for &(p, v) in &[(PieceType::P,100),(PieceType::N,320),(PieceType::B,330),(PieceType::R,500),(PieceType::Q,900)] {
            score += board.count_piece(Player::White, p) as i32 * v;
            score -= board.count_piece(Player::Black, p) as i32 * v;
        }
        // PST tables ported from search/eval.rs
        const PST_PAWN: [i16; 64] = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10,-20,-20, 10, 10,  5,
             5, -5,-10,  0,  0,-10, -5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5,  5, 10, 25, 25, 10,  5,  5,
            10, 10, 20, 30, 30, 20, 10, 10,
            50, 50, 50, 50, 50, 50, 50, 50,
             0,  0,  0,  0,  0,  0,  0,  0,
        ];
        const PST_KNIGHT: [i16; 64] = [
           -50,-40,-30,-30,-30,-30,-40,-50,
           -40,-20,  0,  0,  0,  0,-20,-40,
           -30,  0, 10, 15, 15, 10,  0,-30,
           -30,  5, 15, 20, 20, 15,  5,-30,
           -30,  0, 15, 20, 20, 15,  0,-30,
           -30,  5, 10, 15, 15, 10,  5,-30,
           -40,-20,  0,  5,  5,  0,-20,-40,
           -50,-40,-30,-30,-30,-30,-40,-50,
        ];
        const PST_BISHOP: [i16; 64] = [
           -20,-10,-10,-10,-10,-10,-10,-20,
           -10,  5,  0,  0,  0,  0,  5,-10,
           -10, 10, 10, 10, 10, 10, 10,-10,
           -10,  0, 10, 10, 10, 10,  0,-10,
           -10,  5,  5, 10, 10,  5,  5,-10,
           -10,  0,  5, 10, 10,  5,  0,-10,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -20,-10,-10,-10,-10,-10,-10,-20,
        ];
        const PST_ROOK: [i16; 64] = [
             0,  0,  5, 10, 10,  5,  0,  0,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             5, 10, 10, 10, 10, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0,
        ];
        const PST_QUEEN: [i16; 64] = [
           -20,-10,-10, -5, -5,-10,-10,-20,
           -10,  0,  0,  0,  0,  0,  0,-10,
           -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
             0,  0,  5,  5,  5,  5,  0, -5,
           -10,  5,  5,  5,  5,  5,  0,-10,
           -10,  0,  5,  0,  0,  0,  0,-10,
           -20,-10,-10, -5, -5,-10,-10,-20,
        ];
        const PST_KING: [i16; 64] = [
            20, 30, 10,  0,  0, 10, 30, 20,
            20, 20,  0,  0,  0,  0, 20, 20,
           -10,-20,-20,-20,-20,-20,-20,-10,
           -20,-30,-30,-40,-40,-30,-30,-20,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
           -30,-40,-40,-50,-50,-40,-40,-30,
        ];
        use pleco::SQ;
        const SQS: [SQ; 64] = [
            SQ::A1,SQ::B1,SQ::C1,SQ::D1,SQ::E1,SQ::F1,SQ::G1,SQ::H1,
            SQ::A2,SQ::B2,SQ::C2,SQ::D2,SQ::E2,SQ::F2,SQ::G2,SQ::H2,
            SQ::A3,SQ::B3,SQ::C3,SQ::D3,SQ::E3,SQ::F3,SQ::G3,SQ::H3,
            SQ::A4,SQ::B4,SQ::C4,SQ::D4,SQ::E4,SQ::F4,SQ::G4,SQ::H4,
            SQ::A5,SQ::B5,SQ::C5,SQ::D5,SQ::E5,SQ::F5,SQ::G5,SQ::H5,
            SQ::A6,SQ::B6,SQ::C6,SQ::D6,SQ::E6,SQ::F6,SQ::G6,SQ::H6,
            SQ::A7,SQ::B7,SQ::C7,SQ::D7,SQ::E7,SQ::F7,SQ::G7,SQ::H7,
            SQ::A8,SQ::B8,SQ::C8,SQ::D8,SQ::E8,SQ::F8,SQ::G8,SQ::H8,
        ];
        for (i, &sq) in SQS.iter().enumerate() {
            let p = board.piece_at_sq(sq);
            if p == Piece::None { continue; }
            let pt = p.type_of();
            let is_white = match p { Piece::WhitePawn|Piece::WhiteKnight|Piece::WhiteBishop|Piece::WhiteRook|Piece::WhiteQueen|Piece::WhiteKing => true, _ => false };
            let file = (i % 8) as usize; let rank = (i / 8) as usize;
            let idx = if is_white { i } else { (7 - rank) * 8 + file };
            let delta = match pt {
                PieceType::P => PST_PAWN[idx] as i32,
                PieceType::N => PST_KNIGHT[idx] as i32,
                PieceType::B => PST_BISHOP[idx] as i32,
                PieceType::R => PST_ROOK[idx] as i32,
                PieceType::Q => PST_QUEEN[idx] as i32,
                PieceType::K => PST_KING[idx] as i32,
                _ => 0,
            };
            if is_white { score += delta; } else { score -= delta; }
        }
        if board.turn() == Player::White { score } else { -score }
    }

    fn eval_terminal(&self, board: &PlecoBoard) -> i32 {
        if board.in_check() { -MATE_SCORE } else { DRAW_SCORE }
    }

    // Helpers to validate ordering and counter-move behavior (exposed for tests)
    pub fn debug_move_hist_index(&self, m: PMove) -> usize { self.move_hist_index(m) }

    pub fn debug_set_counter(&mut self, parent_idx: usize, child_idx: usize) {
        if let Some(slot) = self.counter_move.get_mut(parent_idx) { *slot = child_idx; }
    }

    pub fn debug_order_for_parent(&self, board: &PlecoBoard, parent_idx: usize) -> Vec<PMove> {
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        let tt_best = None;
        self.order_moves(board, &mut ml, tt_best, 0, Some(parent_idx));
        ml
    }

    pub fn debug_order_no_parent(&self, board: &PlecoBoard) -> Vec<PMove> {
        let mut ml: Vec<PMove> = board.generate_moves().iter().copied().collect();
        let tt_best = None;
        self.order_moves(board, &mut ml, tt_best, 0, None);
        ml
    }

    pub fn debug_set_cont(&mut self, parent_idx: usize, child_idx: usize, val: i32) {
        let idx = self.cont_index(parent_idx, child_idx);
        if let Some(slot) = self.cont_hist.get_mut(idx) { *slot = val; }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PlecoEvalMode { Material, Pst }
