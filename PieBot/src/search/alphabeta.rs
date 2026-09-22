use crate::eval::nnue::loader::QuantNnue;
use crate::eval::nnue::network::{ChangeSet, QuantNetwork};
use crate::search::eval::{eval_cp, material_eval_cp, DRAW_SCORE, MATE_SCORE};
use crate::search::tt::{Bound, Entry, Tt};
use crate::search::zobrist;
use cozy_chess::{Board, Color, Move, Square};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
const HIST_PROMO_KINDS: usize = 5; // None, N, B, R, Q
const HIST_SIZE: usize = 64 * 64 * HIST_PROMO_KINDS;
const PIECE_SQUARE_ENTRIES: usize = 6 * 64; // 384
const CONTHIST_SIZE: usize = PIECE_SQUARE_ENTRIES * PIECE_SQUARE_ENTRIES; // 147,456
const HISTORY_MAX: i32 = 400;

pub static LMR_TABLE: std::sync::LazyLock<[[i32; 64]; 64]> = std::sync::LazyLock::new(|| {
    let mut table = [[0i32; 64]; 64];
    for depth in 1..64 {
        for m in 1..64 {
            let d_f = depth as f64;
            let m_f = m as f64;
            let r = 0.5 + (d_f.ln() * m_f.ln()) / 2.75;
            table[depth][m] = r.max(0.0).round() as i32;
        }
    }
    table
});

pub fn lmr_reduction(depth: u32, move_idx: usize, history_score: i32) -> u32 {
    if depth < 3 || move_idx < 3 {
        return 0;
    }
    let d = (depth as usize).min(63);
    let m = move_idx.min(63);
    let mut r = LMR_TABLE[d][m];

    if history_score > 4000 {
        r = r.saturating_sub(1);
    } else if history_score < -4000 {
        r += 1;
    }

    r.max(0) as u32
}

#[inline]
fn piece_to_idx(piece: cozy_chess::Piece) -> usize {
    match piece {
        cozy_chess::Piece::Pawn => 0,
        cozy_chess::Piece::Knight => 1,
        cozy_chess::Piece::Bishop => 2,
        cozy_chess::Piece::Rook => 3,
        cozy_chess::Piece::Queen => 4,
        cozy_chess::Piece::King => 5,
    }
}

const CAPHIST_SIZE: usize = 6 * 64 * 6;

#[inline]
fn capture_history_index(piece: cozy_chess::Piece, to: Square, victim: cozy_chess::Piece) -> usize {
    (piece_to_idx(piece) * 64 + (to as usize)) * 6 + piece_to_idx(victim)
}

#[inline]
fn piece_sq_index(piece: cozy_chess::Piece, sq: Square) -> usize {
    piece_to_idx(piece) * 64 + (sq as usize)
}

#[inline]
fn update_history_score(entry: &mut i32, bonus: i32) {
    let clamped_bonus = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    *entry += clamped_bonus - (*entry * clamped_bonus.abs()) / HISTORY_MAX;
}

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

/// Cushion added to a capture's face value before delta-pruning it. Covers the
/// positional swing a capture can produce beyond the material it wins, so the
/// filter only discards lines that cannot plausibly reach alpha.
const QSEARCH_DELTA_MARGIN_CP: i32 = 200;

#[inline]
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let to = m.to;
    let from = m.from;
    let victim = board.piece_on(to).map(piece_value_cp).unwrap_or(0);
    let attacker = board.piece_on(from).map(piece_value_cp).unwrap_or(0);
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
    /// Deepest fully completed iteration; can trail the requested depth
    /// when a node budget or deadline interrupts the search.
    pub depth: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchAbort {
    /// A user stop, deadline, or node budget interrupted the iteration.
    Limit,
    /// A sibling produced a cutoff and cancelled redundant parallel work.
    Cancelled,
}

type SearchScore = Result<i32, SearchAbort>;

const MATE_TT_THRESHOLD: i32 = MATE_SCORE - 1_024;
const FIFTY_MOVE_CLAIM_PLIES: u32 = 100;

#[inline]
fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_TT_THRESHOLD {
        score + ply
    } else if score <= -MATE_TT_THRESHOLD {
        score - ply
    } else {
        score
    }
}

#[inline]
fn score_from_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_TT_THRESHOLD {
        score - ply
    } else if score <= -MATE_TT_THRESHOLD {
        score + ply
    } else {
        score
    }
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
    external_stop: Option<Arc<AtomicBool>>,
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
    // New: array-based history, continuation history, and counter-move tables
    history_table: Vec<i32>,
    conthist_1ply: Vec<i32>,
    conthist_2ply: Vec<i32>,
    move_stack: Vec<Option<(cozy_chess::Piece, Square)>>,
    counter_move: Vec<usize>,
    capture_history: Vec<i32>,
    deterministic: bool,
    // Eval mode: material-only, PST, or NNUE
    eval_mode: EvalMode,
    // Instrumentation
    last_depth: u32,
    max_seldepth: u32,
    root_history: Vec<Board>,
    search_history: Vec<Board>,
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
            external_stop: None,
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
            conthist_1ply: vec![0; CONTHIST_SIZE],
            conthist_2ply: vec![0; CONTHIST_SIZE],
            move_stack: vec![None; 128],
            counter_move: vec![usize::MAX; HIST_SIZE],
            capture_history: vec![0; CAPHIST_SIZE],
            deterministic: false,
            eval_mode: EvalMode::Pst,
            last_depth: 0,
            max_seldepth: 0,
            root_history: Vec::new(),
            search_history: Vec::new(),
        }
    }
}

impl Searcher {
    pub fn continuation_history_entries_count(&self) -> usize {
        self.conthist_1ply.iter().filter(|&&x| x != 0).count()
            + self.conthist_2ply.iter().filter(|&&x| x != 0).count()
    }

    /// Supply the real game history, including the current root position, so
    /// recursive search can recognize threefold repetitions.
    pub fn set_position_history(&mut self, history: &[Board]) {
        self.root_history.clear();
        self.root_history.extend_from_slice(history);
    }

    pub fn clear_position_history(&mut self) {
        self.root_history.clear();
        self.search_history.clear();
    }

    /// Install a cooperative stop flag. A stopped iteration is discarded and
    /// the last fully completed result (or a legal fallback) is returned.
    pub fn set_stop_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.external_stop = flag;
    }

    pub fn clear_stop_flag(&mut self) {
        self.external_stop = None;
    }

    fn prepare_root_state(&mut self, board: &Board) {
        self.search_history.clone_from(&self.root_history);
        if self.search_history.last() != Some(board) {
            self.search_history.push(board.clone());
        }
        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.refresh(board);
            }
        }
    }

    #[inline]
    fn rule_draw(&self, board: &Board) -> bool {
        crate::search::draw::is_fifty_move_draw(board)
            || crate::search::draw::is_insufficient_material(board)
            || crate::search::draw::is_threefold(board, &self.search_history)
    }

    /// A TT score omits the halfmove clock from its key and is therefore safe
    /// only when the nominal search horizon cannot reach a fifty-move claim.
    /// The entry's best move remains valid positionally and can still be used
    /// for ordering when its score is context-sensitive.
    #[inline]
    fn tt_score_is_rule50_safe(board: &Board, depth: u32) -> bool {
        u32::from(board.halfmove_clock()).saturating_add(depth) < FIFTY_MOVE_CLAIM_PLIES
    }

    #[inline]
    fn poll_abort(&self) -> Result<(), SearchAbort> {
        if self
            .external_stop
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(SearchAbort::Limit);
        }
        if self
            .abort
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(SearchAbort::Cancelled);
        }
        if self.nodes >= self.node_limit {
            return Err(SearchAbort::Limit);
        }
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(SearchAbort::Limit);
        }
        Ok(())
    }

    #[inline]
    fn enter_node(&mut self, ply: i32) -> Result<(), SearchAbort> {
        self.poll_abort()?;
        self.nodes += 1;
        self.max_seldepth = self.max_seldepth.max(ply.max(0) as u32);
        Ok(())
    }

    fn fallback_result(&mut self, board: &Board) -> SearchResult {
        self.prepare_root_state(board);
        let moves = self.debug_order_root(board);
        let score_cp = if moves.is_empty() {
            self.eval_terminal(board, 0)
        } else if self.rule_draw(board) {
            DRAW_SCORE
        } else {
            self.eval_current(board)
        };
        SearchResult { depth: 0,
            bestmove: moves.first().map(|mv| format!("{mv}")),
            score_cp,
            nodes: self.nodes,
        }
    }

    // Choose evaluation mode
    pub fn set_eval_mode(&mut self, mode: EvalMode) {
        self.eval_mode = mode;
    }
    pub fn set_threads(&mut self, t: usize) {
        self.threads = t.max(1);
    }
    pub fn set_order_captures(&mut self, on: bool) {
        self.order_captures = on;
    }
    pub fn set_use_history(&mut self, on: bool) {
        self.use_history = on;
    }
    pub fn set_use_killers(&mut self, on: bool) {
        self.use_killers = on;
    }
    pub fn set_use_lmr(&mut self, on: bool) {
        self.use_lmr = on;
    }
    pub fn set_use_nullmove(&mut self, on: bool) {
        self.use_nullmove = on;
    }
    pub fn set_use_aspiration(&mut self, on: bool) {
        self.use_aspiration = on;
    }
    pub fn set_deterministic(&mut self, on: bool) {
        self.deterministic = on;
    }
    pub fn see_gain_cp(&mut self, board: &Board, uci: &str) -> Option<i32> {
        // Locate a matching legal move by UCI string
        let mut chosen: Option<Move> = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{m}") == uci {
                    chosen = Some(m);
                    break;
                }
            }
            chosen.is_some()
        });
        chosen.and_then(|m| crate::search::see::see_gain_cp(board, m))
    }

    pub fn qsearch_eval_cp(&mut self, board: &Board) -> i32 {
        self.nodes = 0;
        self.max_seldepth = 0;
        self.node_limit = u64::MAX;
        self.deadline = None;
        self.abort = None;
        self.prepare_root_state(board);
        self.qsearch(board, -MATE_SCORE, MATE_SCORE, 0, true)
            .unwrap_or_else(|_| self.eval_current(board))
    }

    // Time-managed iterative deepening up to a maximum depth
    pub fn search_movetime(
        &mut self,
        board: &Board,
        millis: u64,
        depth: u32,
    ) -> (Option<String>, i32, u64) {
        self.nodes = 0;
        self.last_depth = 0;
        self.max_seldepth = 0;
        self.abort = None;
        self.node_limit = u64::MAX;
        let start_time = Instant::now();
        let allotted = Duration::from_millis(millis);
        self.deadline = Some(start_time + allotted);
        self.prepare_root_state(board);
        if self.use_history {
            self.history_table.fill(0);
            self.conthist_1ply.fill(0);
            self.conthist_2ply.fill(0);
            self.counter_move.fill(usize::MAX);
            self.move_stack.fill(None);
            self.capture_history.fill(0);
        }
        let max_depth = if depth == 0 { 99 } else { depth };
        let mut committed = self.fallback_result(board);

        let mut legal_move_count = 0;
        board.generate_moves(|ml| {
            legal_move_count += ml.len();
            false
        });

        for d in 1..=max_depth {
            if d >= 2 && legal_move_count <= 1 {
                break;
            }
            if start_time.elapsed() >= allotted * 55 / 100 {
                break;
            }
            self.tt.bump_generation();
            self.prepare_root_state(board);
            let iteration = if self.use_aspiration && d >= 4 {
                let mut delta = 30;
                let mut alpha = (committed.score_cp - delta).max(-MATE_SCORE);
                let mut beta = (committed.score_cp + delta).min(MATE_SCORE);
                loop {
                    match self.search_depth_window(board, d, alpha, beta) {
                        Ok(result) => {
                            if result.score_cp <= alpha {
                                delta *= 2;
                                alpha = (committed.score_cp - delta).max(-MATE_SCORE);
                                self.prepare_root_state(board);
                            } else if result.score_cp >= beta {
                                delta *= 2;
                                beta = (committed.score_cp + delta).min(MATE_SCORE);
                                self.prepare_root_state(board);
                            } else {
                                break Ok(result);
                            }
                            if delta > 300 {
                                self.prepare_root_state(board);
                                break self.search_depth_internal(board, d);
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
            } else {
                self.search_depth_internal(board, d)
            };
            match iteration {
                Ok(result) => {
                    committed = result;
                    self.last_depth = d;
                }
                Err(_) => break,
            }
        }
        (committed.bestmove, committed.score_cp, self.nodes)
    }

    fn qsearch(
        &mut self,
        board: &Board,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        check_draws: bool,
    ) -> SearchScore {
        self.enter_node(ply)?;
        let in_check = !board.checkers().is_empty();
        let mut legal: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                legal.push(m);
            }
            false
        });
        if legal.is_empty() {
            return Ok(self.eval_terminal(board, ply));
        }
        if check_draws && self.rule_draw(board) {
            return Ok(DRAW_SCORE);
        }

        let mut stand_pat = 0;
        if !in_check {
            let stand = self.eval_current(board);
            if stand >= beta {
                return Ok(beta);
            }
            if stand > alpha {
                alpha = stand;
            }
            stand_pat = stand;
        }

        let mut tactical: Vec<Move> = legal
            .into_iter()
            .filter(|&mv| in_check || self.is_capture(board, mv) || mv.promotion.is_some())
            .collect();
        tactical.sort_by_key(|&mv| {
            let promotion_bonus = if mv.promotion.is_some() { 10_000 } else { 0 };
            -(promotion_bonus + mvv_lva_score(board, mv))
        });

        for m in tactical {
            // Quiescence is the overwhelming majority of the tree, and it was
            // expanding every capture regardless of whether the exchange was
            // survivable or could possibly matter. Both filters are skipped
            // while in check, where the move list is evasions rather than
            // captures and every one of them has to be searched.
            if !in_check {
                // A provably losing exchange cannot improve the side to move's
                // standing; roughly a third of the captures reaching this loop
                // are in that class.
                if crate::search::see::see_gain_cp(board, m).unwrap_or(0) < 0 {
                    continue;
                }
                // Delta pruning. Winning the piece on the destination square
                // outright, plus a margin for the positional swing a capture
                // can bring, still leaves the line short of alpha -- so its
                // exact value cannot affect this node. Promotions are exempt
                // because the gain is the new piece, not the victim.
                if m.promotion.is_none() {
                    let victim = board.piece_on(m.to).map(piece_value_cp).unwrap_or(
                        // En passant takes a pawn that is not on `m.to`.
                        if board.piece_on(m.from) == Some(cozy_chess::Piece::Pawn) {
                            piece_value_cp(cozy_chess::Piece::Pawn)
                        } else {
                            0
                        },
                    );
                    if stand_pat + victim + QSEARCH_DELTA_MARGIN_CP <= alpha {
                        continue;
                    }
                }
            }
            let mut child = board.clone();
            child.play_unchecked(m);
            let mut change = None;
            if self.use_nnue {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    change = Some(qn.apply_move(board, m, &child));
                }
            }
            self.search_history.push(child.clone());
            // `check_draws = false` is scoped only to a synthetic null
            // position. This move creates a real game position, so normal
            // draw adjudication resumes immediately.
            let child_score = self.qsearch(&child, -beta, -alpha, ply + 1, true);
            self.search_history.pop();
            if let Some(ch) = change {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    qn.revert(ch);
                }
            }
            let score = -child_score?;
            if score >= beta {
                return Ok(beta);
            }
            if score > alpha {
                alpha = score;
            }
        }
        Ok(alpha)
    }

    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        self.nodes = 0;
        self.node_limit = u64::MAX;
        self.deadline = None;
        self.abort = None;
        self.last_depth = 0;
        self.max_seldepth = 0;
        self.prepare_root_state(board);
        match self.search_depth_internal(board, depth) {
            Ok(mut result) => {
                self.last_depth = depth;
                result.nodes = self.nodes;
                result
            }
            Err(_) => self.fallback_result(board),
        }
    }

    fn search_depth_internal(
        &mut self,
        board: &Board,
        depth: u32,
    ) -> Result<SearchResult, SearchAbort> {
        self.poll_abort()?;
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        // Root-split parallel search if threads > 1 and depth > 1
        if self.threads > 1
            && depth >= 4
            && !self.deterministic
            && self.node_limit == u64::MAX
            && (!self.use_nnue || self.nnue_quant.is_some())
        {
            return self.search_depth_parallel(board, depth);
        }

        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.refresh(board);
            }
        }
        let orig_alpha = alpha;
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return Ok(SearchResult { depth: 0,
                bestmove: None,
                score_cp: self.eval_terminal(board, 0),
                nodes: self.nodes,
            });
        }
        if self.rule_draw(board) {
            return Ok(SearchResult { depth: 0,
                bestmove: moves.first().map(|mv| format!("{mv}")),
                score_cp: DRAW_SCORE,
                nodes: self.nodes,
            });
        }
        // TT-first
        let mut tt_move = None;
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                    tt_move = Some(mv);
                }
            }
        }
        // Order with captures/history and a small bonus for checking moves
        // Pre-compute scores to avoid board clones during sorting
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White {
                cozy_chess::Color::Black
            } else {
                cozy_chess::Color::White
            };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0;
            for sq in opp_bb {
                occ_mask |= 1u64 << (sq as usize);
            }

            // Build scored tuples (once per move, not once per comparison)
            let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
            for &m in &moves {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures {
                    if (occ_mask & bit) != 0 {
                        1
                    } else {
                        0
                    }
                } else {
                    0
                };
                let mvv = if is_cap == 1 {
                    mvv_lva_score(board, m)
                } else {
                    0
                };

                let gives_check_bonus = {
                    let mut c = board.clone();
                    c.play_unchecked(m);
                    if !(c.checkers()).is_empty() {
                        30
                    } else {
                        0
                    }
                };

                let mi = move_index(m);
                let hist = if self.use_history {
                    self.history_table.get(mi).copied().unwrap_or(0)
                } else {
                    0
                };
                let kb = if self.use_killers {
                    self.killer_bonus(0, m)
                } else {
                    0
                };

                let is_tt = tt_move == Some(m);
                let sort_score = if is_tt {
                    10_000_000
                } else if is_cap == 1 {
                    let see_gain = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    let cap_hist = if self.use_history {
                        let p = board.piece_on(m.from).unwrap();
                        let victim = board.piece_on(m.to).unwrap_or(cozy_chess::Piece::Pawn);
                        self.capture_history[capture_history_index(p, m.to, victim)] / 16
                    } else {
                        0
                    };
                    if see_gain >= 0 {
                        1_000_000 + mvv * 10 + cap_hist + see_gain + gives_check_bonus
                    } else {
                        -1_000_000 + cap_hist + see_gain + gives_check_bonus
                    }
                } else {
                    let killer_weight = if kb > 0 { kb * 1000 } else { 0 };
                    (killer_weight + gives_check_bonus + hist).clamp(-500_000, 500_000)
                };
                scored.push((m, -sort_score));
            }

            // Sort by pre-computed scores
            scored.sort_by_key(|&(_, score)| score);
            moves = scored.into_iter().map(|(m, _)| m).collect();
        }
        for m in moves.into_iter() {
            let mut child = board.clone();
            child.play_unchecked(m);
            let mut change = None;
            if self.use_nnue {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    change = Some(qn.apply_move(board, m, &child));
                }
            }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let moving_piece = board.piece_on(m.from).unwrap();
            if !self.move_stack.is_empty() {
                self.move_stack[0] = Some((moving_piece, m.to));
            }
            self.search_history.push(child.clone());
            let child_score =
                self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m), true);
            self.search_history.pop();
            if !self.move_stack.is_empty() {
                self.move_stack[0] = None;
            }
            if let Some(ch) = change {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    qn.revert(ch);
                }
            }
            let score = -child_score?;
            if score > best_score {
                best_score = score;
                bestmove = Some(m);
            }
            if score > alpha {
                alpha = score;
            }
        }

        // Store root in TT as exact when using full window
        let root_bound = if best_score <= orig_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_put(board, depth, best_score, bestmove, root_bound, 0);

        let bestmove_uci = bestmove.map(|m| format!("{m}"));
        Ok(SearchResult { depth: 0,
            bestmove: bestmove_uci,
            score_cp: best_score,
            nodes: self.nodes,
        })
    }

    fn search_depth_parallel(
        &mut self,
        board: &Board,
        depth: u32,
    ) -> Result<SearchResult, SearchAbort> {
        self.poll_abort()?;
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let orig_alpha = alpha;
        let mut bestmove: Option<Move>;
        let mut best_score: i32;

        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.refresh(board);
            }
        }

        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return Ok(SearchResult { depth: 0,
                bestmove: None,
                score_cp: self.eval_terminal(board, 0),
                nodes: self.nodes,
            });
        }
        if self.rule_draw(board) {
            return Ok(SearchResult { depth: 0,
                bestmove: moves.first().map(|mv| format!("{mv}")),
                score_cp: DRAW_SCORE,
                nodes: self.nodes,
            });
        }

        // TT-first
        let mut tt_move = None;
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                    tt_move = Some(mv);
                }
            }
        }

        // Root ordering matches serial search to make PV-split effective.
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White {
                cozy_chess::Color::Black
            } else {
                cozy_chess::Color::White
            };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0;
            for sq in opp_bb {
                occ_mask |= 1u64 << (sq as usize);
            }

            let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
            for &m in &moves {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures && (occ_mask & bit) != 0 {
                    1
                } else {
                    0
                };
                let mvv = if is_cap == 1 {
                    mvv_lva_score(board, m)
                } else {
                    0
                };
                let gives_check_bonus = {
                    let mut c = board.clone();
                    c.play_unchecked(m);
                    if !(c.checkers()).is_empty() {
                        30
                    } else {
                        0
                    }
                };
                let mi = move_index(m);
                let hist = if self.use_history {
                    self.history_table.get(mi).copied().unwrap_or(0)
                } else {
                    0
                };
                let kb = if self.use_killers {
                    self.killer_bonus(0, m)
                } else {
                    0
                };
                let is_tt = tt_move == Some(m);
                let sort_score = if is_tt {
                    10_000_000
                } else if is_cap == 1 {
                    let see_gain = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    let cap_hist = if self.use_history {
                        let p = board.piece_on(m.from).unwrap();
                        let victim = board.piece_on(m.to).unwrap_or(cozy_chess::Piece::Pawn);
                        self.capture_history[capture_history_index(p, m.to, victim)] / 16
                    } else {
                        0
                    };
                    if see_gain >= 0 {
                        1_000_000 + mvv * 10 + cap_hist + see_gain + gives_check_bonus
                    } else {
                        -1_000_000 + cap_hist + see_gain + gives_check_bonus
                    }
                } else {
                    let killer_weight = if kb > 0 { kb * 1000 } else { 0 };
                    (killer_weight + gives_check_bonus + hist).clamp(-500_000, 500_000)
                };
                scored.push((m, -sort_score));
            }
            scored.sort_by_key(|&(_, score)| score);
            moves = scored.into_iter().map(|(m, _)| m).collect();
        }

        // Root PV-split: seed first move, search tail in parallel with shared alpha.
        let deadline = self.deadline;
        let shared_tt = self.tt.clone();
        let quant_network = self.nnue_quant.as_ref().map(QuantNetwork::clone_for_search);
        let eval_mode = self.eval_mode;
        let use_nnue = self.use_nnue;
        let eval_blend_percent = self.eval_blend_percent;
        let order_captures = self.order_captures;
        let use_history = self.use_history;
        let use_killers = self.use_killers;
        let use_lmr = self.use_lmr;
        let use_nullmove = self.use_nullmove;
        let node_limit = self.node_limit;
        let external_stop = self.external_stop.clone();
        let search_history = self.search_history.clone();

        let make_worker = || {
            let mut w = Searcher::default();
            w.node_limit = node_limit;
            w.deadline = deadline;
            w.order_captures = order_captures;
            w.use_history = use_history;
            w.use_killers = use_killers;
            w.use_lmr = use_lmr;
            w.use_nullmove = use_nullmove;
            w.eval_mode = eval_mode;
            w.eval_blend_percent = eval_blend_percent;
            w.tt = shared_tt.clone();
            w.use_nnue = use_nnue;
            w.threads = 1;
            w.external_stop = external_stop.clone();
            w.search_history = search_history.clone();
            if let Some(network) = &quant_network {
                w.nnue_quant = Some(network.clone_for_search());
            }
            w
        };

        // Seed a few PV moves serially to raise alpha before tail parallelization.
        // This reduces root over-search, especially in fixed-depth test runs.
        let seed_count = if deadline.is_some() {
            1
        } else {
            moves.len().min(4)
        };
        best_score = -MATE_SCORE;
        bestmove = None;
        for (i, &m) in moves.iter().take(seed_count).enumerate() {
            let mut child = board.clone();
            child.play_unchecked(m);
            let mut seed = make_worker();
            seed.search_history.push(child.clone());
            if seed.use_nnue {
                if let Some(qn) = seed.nnue_quant.as_mut() {
                    qn.refresh(&child);
                }
            }
            let ext = if !(child.checkers()).is_empty() { 1 } else { 0 };
            let next_depth = depth.saturating_sub(1) + ext;
            let score_result = if i == 0 {
                seed.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m), true)
                    .map(|score| -score)
            } else {
                match seed.alphabeta(
                    &child,
                    next_depth,
                    -alpha - 1,
                    -alpha,
                    1,
                    move_index(m),
                    true,
                ) {
                    Ok(value) => {
                        let score = -value;
                        if score > alpha {
                            seed.alphabeta(
                                &child,
                                next_depth,
                                -beta,
                                -alpha,
                                1,
                                move_index(m),
                                true,
                            )
                            .map(|value| -value)
                        } else {
                            Ok(score)
                        }
                    }
                    Err(reason) => Err(reason),
                }
            };
            self.nodes += seed.nodes;
            self.max_seldepth = self.max_seldepth.max(seed.max_seldepth);
            let score = score_result?;
            if score > best_score {
                best_score = score;
                bestmove = Some(m);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        if alpha >= beta || seed_count >= moves.len() {
            let root_bound = if best_score <= orig_alpha {
                Bound::Upper
            } else if best_score >= beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.tt_put(board, depth, best_score, bestmove, root_bound, 0);
            return Ok(SearchResult { depth: 0,
                bestmove: bestmove.map(|m| format!("{m}")),
                score_cp: best_score,
                nodes: self.nodes,
            });
        }

        let alpha_shared = AtomicI32::new(alpha);
        let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tails = &moves[seed_count..];
        let results: Vec<(Move, SearchScore, u64, u32)> = tails
            .par_iter()
            .map(|&m| {
                if abort.load(Ordering::Relaxed) {
                    return (m, Err(SearchAbort::Cancelled), 0, 0);
                }
                let mut child = board.clone();
                child.play_unchecked(m);
                let mut w = make_worker();
                w.abort = Some(abort.clone());
                w.search_history.push(child.clone());
                if w.use_nnue {
                    if let Some(qn) = w.nnue_quant.as_mut() {
                        qn.refresh(&child);
                    }
                }
                let gives_check = !(child.checkers()).is_empty();
                let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
                let local_alpha = alpha_shared.load(Ordering::Relaxed);
                let first = w.alphabeta(
                    &child,
                    next_depth,
                    -local_alpha - 1,
                    -local_alpha,
                    1,
                    move_index(m),
                    true,
                );
                let mut score = match first {
                    Ok(value) => -value,
                    Err(reason) => return (m, Err(reason), w.nodes, w.max_seldepth),
                };
                if score > local_alpha {
                    score = match w.alphabeta(
                        &child,
                        next_depth,
                        -beta,
                        -local_alpha,
                        1,
                        move_index(m),
                        true,
                    ) {
                        Ok(value) => -value,
                        Err(reason) => return (m, Err(reason), w.nodes, w.max_seldepth),
                    };
                }
                let mut cur = local_alpha;
                while score > cur {
                    match alpha_shared.compare_exchange(
                        cur,
                        score,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(observed) => {
                            if observed >= score {
                                break;
                            }
                            cur = observed;
                        }
                    }
                }
                if score >= beta {
                    abort.store(true, Ordering::Relaxed);
                }
                (m, Ok(score), w.nodes, w.max_seldepth)
            })
            .collect();

        let mut interrupted = false;
        let mut cutoff = false;
        for (m, result, n, seldepth) in results {
            self.nodes += n;
            self.max_seldepth = self.max_seldepth.max(seldepth);
            match result {
                Ok(score) => {
                    cutoff |= score >= beta;
                    if score > best_score {
                        best_score = score;
                        bestmove = Some(m);
                    }
                }
                Err(SearchAbort::Limit) => interrupted = true,
                Err(SearchAbort::Cancelled) => {}
            }
        }
        if interrupted || (abort.load(Ordering::Relaxed) && !cutoff) {
            return Err(SearchAbort::Limit);
        }

        let root_bound = if best_score <= orig_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_put(board, depth, best_score, bestmove, root_bound, 0);
        Ok(SearchResult { depth: 0,
            bestmove: bestmove.map(|m| format!("{m}")),
            score_cp: best_score,
            nodes: self.nodes,
        })
    }

    fn order_moves_slice(
        &self,
        board: &Board,
        moves: &mut [Move],
        ply: i32,
        parent_move_idx: usize,
    ) {
        if moves.len() <= 1 {
            return;
        }
        let opp = if board.side_to_move() == cozy_chess::Color::White {
            cozy_chess::Color::Black
        } else {
            cozy_chess::Color::White
        };
        let opp_bb = board.colors(opp);
        let mut occ_mask: u64 = 0;
        for sq in opp_bb {
            occ_mask |= 1u64 << (sq as usize);
        }
        let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
        for &m in moves.iter() {
            let to_sq: Square = m.to;
            let bit = 1u64 << (to_sq as usize);
            let is_cap = if self.order_captures {
                if (occ_mask & bit) != 0 {
                    1
                } else {
                    0
                }
            } else {
                0
            };
            let mvv = if is_cap == 1 {
                mvv_lva_score(board, m)
            } else {
                0
            };
            let mi = move_index(m);
            let hist = if self.use_history {
                self.history_table.get(mi).copied().unwrap_or(0)
            } else {
                0
            };
            let conthist = if self.use_history && is_cap == 0 {
                let mut ch = 0;
                if let Some(piece) = board.piece_on(m.from) {
                    let curr_psi = piece_sq_index(piece, m.to);
                    if ply > 0 {
                        if let Some(Some((prev1_p, prev1_sq))) = self.move_stack.get((ply - 1) as usize) {
                            let prev1_psi = piece_sq_index(*prev1_p, *prev1_sq);
                            ch += 2 * self.conthist_1ply[prev1_psi * PIECE_SQUARE_ENTRIES + curr_psi];
                        }
                    }
                    if ply > 1 {
                        if let Some(Some((prev2_p, prev2_sq))) = self.move_stack.get((ply - 2) as usize) {
                            let prev2_psi = piece_sq_index(*prev2_p, *prev2_sq);
                            ch += self.conthist_2ply[prev2_psi * PIECE_SQUARE_ENTRIES + curr_psi];
                        }
                    }
                }
                (ch / 3).clamp(-300, 300)
            } else {
                0
            };
            let cm = if self.use_history && parent_move_idx != usize::MAX {
                if self
                    .counter_move
                    .get(parent_move_idx)
                    .copied()
                    .unwrap_or(usize::MAX)
                    == mi
                {
                    40
                } else {
                    0
                }
            } else {
                0
            };
            let kb = if self.use_killers {
                self.killer_bonus(ply, m)
            } else {
                0
            };
            let sort_score = if is_cap == 1 {
                let see_gain = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                let cap_hist = if self.use_history {
                    let p = board.piece_on(m.from).unwrap();
                    let victim = board.piece_on(m.to).unwrap_or(cozy_chess::Piece::Pawn);
                    self.capture_history[capture_history_index(p, m.to, victim)] / 16
                } else {
                    0
                };
                if see_gain >= 0 {
                    1_000_000 + mvv * 10 + cap_hist + see_gain
                } else {
                    -1_000_000 + cap_hist + see_gain
                }
            } else {
                let killer_weight = if kb > 0 { kb * 1000 } else { 0 };
                let cm_weight = if cm > 0 { 20_000 } else { 0 };
                (killer_weight + cm_weight + hist + conthist).clamp(-500_000, 500_000)
            };
            scored.push((m, -sort_score));
        }
        scored.sort_by_key(|&(_, score)| score);
        for (i, (m, _)) in scored.into_iter().enumerate() {
            moves[i] = m;
        }
    }

    fn alphabeta(
        &mut self,
        board: &Board,
        depth: u32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        parent_move_idx: usize,
        check_draws: bool,
    ) -> SearchScore {
        if depth == 0 {
            return self.qsearch(board, alpha, beta, ply, check_draws);
        }
        self.enter_node(ply)?;
        if check_draws && self.rule_draw(board) {
            // Checkmate ends the game before a fifty-move/repetition claim.
            // We only pay for this legal-move probe at a position that would
            // otherwise be returned as a rule draw.
            let mut has_legal_move = false;
            board.generate_moves(|ml| {
                has_legal_move = ml.into_iter().next().is_some();
                has_legal_move
            });
            return Ok(if has_legal_move {
                DRAW_SCORE
            } else {
                self.eval_terminal(board, ply)
            });
        }
        // Reverse futility: at shallow non-mate-window nodes not in check, a
        // static eval comfortably above beta almost never comes back below it
        // after a real search; return the eval as a fail-soft bound. The
        // margin grows with depth so deeper nodes need a bigger cushion.
        let mut static_eval: Option<i32> = None;
        if self.use_nullmove
            && depth <= 7
            && beta.abs() < MATE_TT_THRESHOLD
            && alpha.abs() < MATE_TT_THRESHOLD
            && board.checkers().is_empty()
        {
            let eval = *static_eval.get_or_insert_with(|| self.eval_current(board));
            if eval - 90 * depth as i32 >= beta {
                return Ok(eval);
            }
        }
        // Null-move pruning with additional guards for shallow depths and endgames
        if self.should_try_null_move(board, depth, beta, parent_move_idx, &mut static_eval) {
            let eval = static_eval.unwrap_or_else(|| self.eval_current(board));
            let r = self.null_move_reduction(depth, eval, beta);
            if let Some(nb) = board.null_move() {
                // A null move changes the side to move without moving a piece.
                // The arch-v2 network is side-to-move relative, so it has to be
                // told: otherwise the whole null subtree evaluates from the
                // parent's perspective. Reverted on every exit path, including
                // aborts, so the accumulator state cannot leak upward.
                let null_change = self.nnue_apply_null_move(&nb);
                let outcome = self.null_move_probe(&nb, depth, r, beta, ply);
                self.nnue_revert_change(null_change);
                if let Some(score) = outcome? {
                    return Ok(score);
                }
            }
        }

        // TT probe (exact-only)
        if Self::tt_score_is_rule50_safe(board, depth) {
            if let Some(en) = self.tt_get(board) {
                if en.depth >= depth {
                    let tt_score = score_from_tt(en.score, ply);
                    match en.bound {
                        Bound::Exact => return Ok(tt_score),
                        Bound::Lower => {
                            if tt_score >= beta {
                                return Ok(tt_score);
                            }
                        }
                        Bound::Upper => {
                            if tt_score <= alpha {
                                return Ok(tt_score);
                            }
                        }
                    }
                }
            }
        }

        // Build movelist and order
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return Ok(self.eval_terminal(board, ply));
        }
        // TT move first
        let mut tt_best: Option<Move> = None;
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                    tt_best = Some(mv);
                }
            }
        }
        // Lazy move ordering: if a TT move exists, defer ordering the remaining moves
        // until after the TT move is searched. If the TT move produces a beta-cutoff,
        // we avoid all move scoring, SEE calculations, and allocations entirely.
        let mut remaining_ordered = false;
        if tt_best.is_none() && (self.order_captures || self.use_history || self.use_killers) {
            self.order_moves_slice(board, &mut moves, ply, parent_move_idx);
            remaining_ordered = true;
        }

        let mut best = -MATE_SCORE;
        let mut best_move_local: Option<Move> = None;
        let orig_alpha = alpha;
        let non_pv = beta <= alpha + 1;
        let mut quiets_tried = [Move {
            from: Square::A1,
            to: Square::A1,
            promotion: None,
        }; 64];
        let mut num_quiets_tried: usize = 0;
        let mut captures_tried = [Move {
            from: Square::A1,
            to: Square::A1,
            promotion: None,
        }; 32];
        let mut num_captures_tried: usize = 0;
        let num_moves = moves.len();
        for idx in 0..num_moves {
            if idx == 1 && !remaining_ordered && (self.order_captures || self.use_history || self.use_killers) {
                self.order_moves_slice(board, &mut moves[1..], ply, parent_move_idx);
                remaining_ordered = true;
            }
            let m = moves[idx];
            let is_capture_move = self.is_capture(board, m);
            if !is_capture_move && num_quiets_tried < 64 {
                quiets_tried[num_quiets_tried] = m;
                num_quiets_tried += 1;
            } else if is_capture_move && num_captures_tried < 32 {
                captures_tried[num_captures_tried] = m;
                num_captures_tried += 1;
            }
            let mut child = board.clone();
            child.play_unchecked(m);
            let gives_check = !(child.checkers()).is_empty();
            // Futility pruning: at shallow depth in non-PV nodes, a quiet non-checking move
            // whose parent static eval plus a depth-scaled margin still cannot
            // reach alpha is skipped before paying eval-update and child-search
            // costs. The first move is always searched; mate windows and
            // in-check parents are exempt.
            if self.use_nullmove
                && non_pv
                && idx > 0
                && depth <= 3
                && !gives_check
                && alpha.abs() < MATE_TT_THRESHOLD
                && beta.abs() < MATE_TT_THRESHOLD
                && board.checkers().is_empty()
                && !self.is_capture(board, m)
                && m.promotion.is_none()
            {
                const FUTILITY_MARGIN: [i32; 4] = [0, 120, 200, 280];
                let eval = *static_eval.get_or_insert_with(|| self.eval_current(board));
                if eval + FUTILITY_MARGIN[depth as usize] <= alpha {
                    continue;
                }
            }

            // Late Move Pruning: in non-PV nodes at shallow depth, skip quiet non-checking
            // moves with neutral or negative history after searching the most promising candidates.
            if self.use_nullmove
                && non_pv
                && depth <= 4
                && !is_capture_move
                && m.promotion.is_none()
                && !gives_check
                && board.checkers().is_empty()
                && alpha.abs() < MATE_TT_THRESHOLD
                && beta.abs() < MATE_TT_THRESHOLD
            {
                let lmp_threshold = 3 + 3 * (depth as usize) * (depth as usize);
                let mi = move_index(m);
                let hist = self.history_table.get(mi).copied().unwrap_or(0);
                if num_quiets_tried > lmp_threshold && hist <= 0 {
                    continue;
                }
            }

            // SEE Pruning: at shallow depths in non-PV nodes, prune moves that lose material
            if self.use_nullmove
                && non_pv
                && idx > 0
                && depth <= 3
                && !gives_check
                && alpha.abs() < MATE_TT_THRESHOLD
                && beta.abs() < MATE_TT_THRESHOLD
                && board.checkers().is_empty()
            {
                if is_capture_move {
                    if let Some(gain) = crate::search::see::see_gain_cp(board, m) {
                        if gain < -100 * (depth as i32) {
                            continue;
                        }
                    }
                } else if depth <= 2 && m.promotion.is_none() {
                    if let Some(gain) = crate::search::see::see_gain_cp(board, m) {
                        if gain < -50 * (depth as i32) {
                            continue;
                        }
                    }
                }
            }

            let mut change = None;
            if self.use_nnue {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    change = Some(qn.apply_move(board, m, &child));
                }
            }
            let moving_piece = board.piece_on(m.from).unwrap();
            if (ply as usize) < self.move_stack.len() {
                self.move_stack[ply as usize] = Some((moving_piece, m.to));
            }
            self.search_history.push(child.clone());
            let ext = if gives_check { 1 } else { 0 };
            // Principal variation search: the first move is searched with the
            // full window; every later move is scouted with a zero window
            // (optionally LMR-reduced) and re-searched at the full window only
            // when the scout fails high inside the window.
            let score_result: SearchScore;
            if idx == 0 {
                score_result = self
                    .alphabeta(
                        &child,
                        depth - 1 + ext,
                        -beta,
                        -alpha,
                        ply + 1,
                        move_index(m),
                        true,
                    )
                    .map(|value| -value);
            } else {
                let r = if self.use_lmr
                    && depth >= 3
                    && idx >= 3
                    && !gives_check
                    && !self.is_capture(board, m)
                    && m.promotion.is_none()
                {
                    let mi = move_index(m);
                    let hist = self.history_table.get(mi).copied().unwrap_or(0);
                    let mut red = lmr_reduction(depth, idx + 1, hist);
                    if self.use_killers && self.killer_bonus(ply, m) > 0 {
                        red = red.saturating_sub(1);
                    }
                    red
                } else {
                    0
                };
                let mut scout = self.alphabeta(
                    &child,
                    depth - 1 - r + ext,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    move_index(m),
                    true,
                );
                // A reduced fail-high is only evidence, not proof: verify at
                // full depth (still zero window) before trusting it.
                if r > 0 {
                    if let Ok(value) = scout {
                        if -value > alpha {
                            scout = self.alphabeta(
                                &child,
                                depth - 1 + ext,
                                -alpha - 1,
                                -alpha,
                                ply + 1,
                                move_index(m),
                                true,
                            );
                        }
                    }
                }
                score_result = match scout {
                    Ok(value) if -value > alpha && -value < beta => self
                        .alphabeta(
                            &child,
                            depth - 1 + ext,
                            -beta,
                            -alpha,
                            ply + 1,
                            move_index(m),
                            true,
                        )
                        .map(|value| -value),
                    Ok(value) => Ok(-value),
                    Err(reason) => Err(reason),
                };
            }
            self.search_history.pop();
            if (ply as usize) < self.move_stack.len() {
                self.move_stack[ply as usize] = None;
            }
            if let Some(change) = change {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    qn.revert(change);
                }
            }
            let score = score_result?;
            if score > best {
                best = score;
                best_move_local = Some(m);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
            // (removed) string-based continuation history
        }
        // Store exact score and best move
        let bound = if best <= orig_alpha {
            Bound::Upper
        } else if best >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.tt_put(board, depth, best, best_move_local, bound, ply);
        if let Some(mv) = best_move_local {
            let mi = move_index(mv);
            let is_cap = self.is_capture(board, mv);
            if self.use_history && bound != Bound::Upper && !is_cap {
                let bonus = ((depth as i32).min(16) * (depth as i32).min(16) * 32).min(1200);
                let malus = -bonus;

                // 1. History table update with gravity
                if let Some(h) = self.history_table.get_mut(mi) {
                    update_history_score(h, bonus);
                }

                // 2. Continuation history for cutoff move
                let curr_p = board.piece_on(mv.from).unwrap();
                let curr_psi = piece_sq_index(curr_p, mv.to);

                if ply > 0 {
                    if let Some(Some((prev1_p, prev1_sq))) = self.move_stack.get((ply - 1) as usize) {
                        let prev1_psi = piece_sq_index(*prev1_p, *prev1_sq);
                        update_history_score(
                            &mut self.conthist_1ply[prev1_psi * PIECE_SQUARE_ENTRIES + curr_psi],
                            bonus,
                        );
                    }
                }
                if ply > 1 {
                    if let Some(Some((prev2_p, prev2_sq))) = self.move_stack.get((ply - 2) as usize) {
                        let prev2_psi = piece_sq_index(*prev2_p, *prev2_sq);
                        update_history_score(
                            &mut self.conthist_2ply[prev2_psi * PIECE_SQUARE_ENTRIES + curr_psi],
                            bonus,
                        );
                    }
                }

                // 3. Apply malus to all quiet moves tried before this cutoff move
                for &qm in &quiets_tried[..num_quiets_tried] {
                    if qm == mv {
                        continue;
                    }
                    let qmi = move_index(qm);
                    if let Some(qh) = self.history_table.get_mut(qmi) {
                        update_history_score(qh, malus);
                    }
                    if let Some(qp) = board.piece_on(qm.from) {
                        let q_psi = piece_sq_index(qp, qm.to);
                        if ply > 0 {
                            if let Some(Some((prev1_p, prev1_sq))) = self.move_stack.get((ply - 1) as usize) {
                                let prev1_psi = piece_sq_index(*prev1_p, *prev1_sq);
                                update_history_score(
                                    &mut self.conthist_1ply[prev1_psi * PIECE_SQUARE_ENTRIES + q_psi],
                                    malus,
                                );
                            }
                        }
                        if ply > 1 {
                            if let Some(Some((prev2_p, prev2_sq))) = self.move_stack.get((ply - 2) as usize) {
                                let prev2_psi = piece_sq_index(*prev2_p, *prev2_sq);
                                update_history_score(
                                    &mut self.conthist_2ply[prev2_psi * PIECE_SQUARE_ENTRIES + q_psi],
                                    malus,
                                );
                            }
                        }
                    }
                }
            } else if self.use_history && bound != Bound::Upper && is_cap {
                let bonus = ((depth as i32).min(16) * (depth as i32).min(16) * 32).min(1200);
                let malus = -bonus;
                let curr_p = board.piece_on(mv.from).unwrap();
                let curr_v = board.piece_on(mv.to).unwrap_or(cozy_chess::Piece::Pawn);
                let chi = capture_history_index(curr_p, mv.to, curr_v);
                if let Some(h) = self.capture_history.get_mut(chi) {
                    update_history_score(h, bonus);
                }
                for &cm in &captures_tried[..num_captures_tried] {
                    if cm == mv {
                        continue;
                    }
                    if let Some(cp) = board.piece_on(cm.from) {
                        let cv = board.piece_on(cm.to).unwrap_or(cozy_chess::Piece::Pawn);
                        let cchi = capture_history_index(cp, cm.to, cv);
                        if let Some(ch) = self.capture_history.get_mut(cchi) {
                            update_history_score(ch, malus);
                        }
                    }
                }
            } else if self.use_history && bound == Bound::Upper {
                // All quiets tried failed low
                let malus = -((depth as i32).min(16) * (depth as i32).min(16) * 16).min(600);
                for &qm in &quiets_tried[..num_quiets_tried] {
                    let qmi = move_index(qm);
                    if let Some(qh) = self.history_table.get_mut(qmi) {
                        update_history_score(qh, malus);
                    }
                }
            }

            if self.use_killers && bound == Bound::Lower {
                self.update_killers(ply, mv);
            }
            if self.use_history && bound == Bound::Lower && parent_move_idx != usize::MAX {
                if let Some(slot) = self.counter_move.get_mut(parent_move_idx) {
                    *slot = mi;
                }
            }
        }
        Ok(best)
    }

    /// The null-move scout and its verification searches.
    ///
    /// Returns `Some(score)` when the null move produced a cutoff and `None`
    /// when the search must proceed normally. Split out of `alphabeta` so the
    /// caller owns a single apply/revert pair around it and every early exit
    /// -- including `?` aborts -- restores the network's side-to-move state.
    fn null_move_probe(
        &mut self,
        nb: &Board,
        depth: u32,
        r: u32,
        beta: i32,
        ply: i32,
    ) -> Result<Option<i32>, SearchAbort> {
        let reduced_depth = depth.saturating_sub(1 + r);
        let score = -self.alphabeta(
            nb,
            reduced_depth,
            -beta,
            -beta + 1,
            ply + 1,
            usize::MAX,
            false,
        )?;
        if score < beta {
            return Ok(None);
        }
        // Below the high-depth band the reduced fail-high is taken at face
        // value. Re-searching it at full depth d-1 is R=0 -- a near-full-width
        // subtree, i.e. most of what null-move pruning exists to avoid -- and
        // because a 150 ms search only reaches depth 6-8, a `depth <= 12`
        // guard fired at essentially every null cutoff in a real game. The
        // zugzwang risk it was standing in for is already covered, and more
        // directly, by should_try_null_move's pieces-only material floor,
        // occupancy floor and is_zugzwang_prone check.
        if depth <= 12 {
            return Ok(Some(score));
        }
        if reduced_depth > 0 {
            let verify_r = r.saturating_sub(1).max(1);
            let verify_depth = depth.saturating_sub(1 + verify_r);
            if verify_depth == 0 {
                return Ok(Some(score));
            }
            let verify = -self.alphabeta(
                nb,
                verify_depth,
                -beta,
                -beta + 1,
                ply + 1,
                usize::MAX,
                false,
            )?;
            if verify >= beta {
                return Ok(Some(verify));
            }
        } else {
            return Ok(Some(score));
        }
        Ok(None)
    }

    #[inline]
    fn nnue_apply_null_move(&mut self, after: &Board) -> Option<ChangeSet> {
        if !self.use_nnue {
            return None;
        }
        self.nnue_quant.as_mut().map(|qn| qn.apply_null_move(after))
    }

    #[inline]
    fn nnue_revert_change(&mut self, change: Option<ChangeSet>) {
        if let Some(change) = change {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.revert(change);
            }
        }
    }

    fn null_move_reduction(&self, depth: u32, eval: i32, beta: i32) -> u32 {
        let mut r = if depth <= 4 { 1 } else { 2 };
        if depth >= 8 {
            r = 3;
        }
        if depth >= 11 {
            r = 4;
        }
        let eval_margin = eval - beta;
        if eval_margin > 300 {
            r += 1;
        }
        if eval_margin > 600 {
            r += 1;
        }
        if depth <= 12 && r > 2 {
            r = 2;
        }
        r = r.min(depth.saturating_sub(1));
        r.max(1)
    }

    fn should_try_null_move(
        &self,
        board: &Board,
        depth: u32,
        beta: i32,
        parent_move_idx: usize,
        static_eval: &mut Option<i32>,
    ) -> bool {
        if !self.use_nullmove {
            return false;
        }
        if depth < 3 {
            return false;
        }
        if !(board.checkers()).is_empty() {
            return false;
        }
        if parent_move_idx == usize::MAX {
            return false;
        }
        if self.is_zugzwang_prone(board) {
            return false;
        }
        let eval = static_eval.get_or_insert_with(|| self.eval_current(board));
        if self.get_mate_distance(*eval) < 800 {
            return false;
        }
        let beta_mate_dist = MATE_SCORE - beta.abs();
        if beta_mate_dist < 800 {
            return false;
        }
        if *eval < beta {
            return false;
        }
        let stm = board.side_to_move();
        let material_cp = self.side_material_cp(board, stm);
        if depth <= 12 && material_cp <= 800 {
            return false;
        }
        if depth <= 5 && *eval + 80 < beta {
            return false;
        }
        let mut occupied_count = 0;
        for _ in board.occupied() {
            occupied_count += 1;
        }
        if occupied_count <= 8 {
            return false;
        }
        true
    }

    fn is_zugzwang_prone(&self, board: &Board) -> bool {
        let stm = board.side_to_move();
        let our_pieces = board.colors(stm);
        let our_king = board.pieces(cozy_chess::Piece::King) & our_pieces;
        let our_pawns = board.pieces(cozy_chess::Piece::Pawn) & our_pieces;
        (our_pieces ^ our_king ^ our_pawns).is_empty()
    }

    fn get_mate_distance(&self, eval: i32) -> i32 {
        MATE_SCORE - eval.abs()
    }

    fn side_material_cp(&self, board: &Board, color: Color) -> i32 {
        let pieces = board.colors(color);
        let mut total = 0;
        for &piece in &[
            cozy_chess::Piece::Pawn,
            cozy_chess::Piece::Knight,
            cozy_chess::Piece::Bishop,
            cozy_chess::Piece::Rook,
            cozy_chess::Piece::Queen,
        ] {
            let bb = board.pieces(piece) & pieces;
            for _ in bb {
                total += piece_value_cp(piece);
            }
        }
        total
    }

    fn eval_terminal(&self, board: &Board, ply: i32) -> i32 {
        if !(board.checkers()).is_empty() {
            return -MATE_SCORE + ply;
        }
        DRAW_SCORE
    }
}

#[cfg(test)]
mod mate_tt_score_tests {
    use super::{score_from_tt, score_to_tt, Searcher};
    use crate::search::eval::MATE_SCORE;
    use crate::search::tt::Bound;
    use cozy_chess::Board;

    #[test]
    fn mate_score_is_adjusted_when_probed_at_a_different_ply() {
        let board = Board::default();
        let mut searcher = Searcher::default();
        searcher.tt_put(&board, 4, MATE_SCORE - 7, None, Bound::Exact, 3);
        let entry = searcher.tt_get(&board).expect("stored mate entry");
        assert_eq!(score_from_tt(entry.score, 8), MATE_SCORE - 12);

        searcher.tt_put(&board, 4, -MATE_SCORE + 7, None, Bound::Exact, 3);
        let loss = searcher.tt_get(&board).expect("stored loss entry");
        assert_eq!(score_from_tt(loss.score, 8), -MATE_SCORE + 12);
    }

    #[test]
    fn ordinary_centipawn_scores_are_not_ply_adjusted() {
        assert_eq!(score_from_tt(score_to_tt(173, 4), 29), 173);
        assert_eq!(score_from_tt(score_to_tt(-842, 4), 29), -842);
    }
}

impl Searcher {
    fn tt_key(board: &Board) -> u64 {
        zobrist::compute(board)
    }
    fn tt_get(&self, board: &Board) -> Option<Entry> {
        self.tt.get(Self::tt_key(board))
    }
    fn tt_put(
        &mut self,
        board: &Board,
        depth: u32,
        score: i32,
        best: Option<Move>,
        bound: Bound,
        ply: i32,
    ) {
        // Rule-sensitive scores must not later poison the same piece position
        // at a low halfmove clock. Depth zero makes this a move-ordering-only
        // entry because depth-zero nodes go directly to qsearch before probing.
        let stored_depth = if Self::tt_score_is_rule50_safe(board, depth) {
            depth
        } else {
            0
        };
        let e = Entry {
            key: Self::tt_key(board),
            depth: stored_depth,
            score: score_to_tt(score, ply),
            best,
            bound,
            gen: 0,
        };
        self.tt.put(e);
    }

    pub fn search_with_params(&mut self, board: &Board, params: SearchParams) -> SearchResult {
        // Configure this search
        self.nodes = 0;
        self.last_depth = 0;
        self.max_seldepth = 0;
        self.abort = None;
        self.node_limit = params.max_nodes.unwrap_or(u64::MAX);
        if !params.use_tt {
            self.tt = Arc::new(Tt::new());
        }
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
            self.history_table.fill(0);
            self.conthist_1ply.fill(0);
            self.conthist_2ply.fill(0);
            self.counter_move.fill(usize::MAX);
            self.move_stack.fill(None);
            self.capture_history.fill(0);
        }
        let start_time = Instant::now();
        self.deadline = params.movetime.map(|d| start_time + d);
        self.prepare_root_state(board);
        let mut committed = self.fallback_result(board);
        let max_depth = if params.depth == 0 { 99 } else { params.depth };

        let mut legal_move_count = 0;
        board.generate_moves(|ml| {
            legal_move_count += ml.len();
            false
        });

        for d in 1..=max_depth {
            if d >= 2 && legal_move_count <= 1 {
                break;
            }
            if let Some(movetime) = params.movetime {
                if start_time.elapsed() >= movetime * 55 / 100 {
                    break;
                }
            }
            self.tt.bump_generation();
            self.prepare_root_state(board);
            let iteration = if self.use_aspiration && d >= 4 {
                let mut delta = params.aspiration_window_cp.max(15);
                let mut alpha = (committed.score_cp - delta).max(-MATE_SCORE);
                let mut beta = (committed.score_cp + delta).min(MATE_SCORE);
                loop {
                    match self.search_depth_window(board, d, alpha, beta) {
                        Ok(result) => {
                            if result.score_cp <= alpha {
                                delta *= 2;
                                alpha = (committed.score_cp - delta).max(-MATE_SCORE);
                                self.prepare_root_state(board);
                            } else if result.score_cp >= beta {
                                delta *= 2;
                                beta = (committed.score_cp + delta).min(MATE_SCORE);
                                self.prepare_root_state(board);
                            } else {
                                break Ok(result);
                            }
                            if delta > 300 {
                                self.prepare_root_state(board);
                                break self.search_depth_internal(board, d);
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
            } else {
                self.search_depth_internal(board, d)
            };
            match iteration {
                Ok(result) => {
                    committed = result;
                    self.last_depth = d;
                }
                Err(_) => break,
            }
        }
        committed.nodes = self.nodes;
        committed.depth = self.last_depth;
        committed
    }

    fn search_depth_window(
        &mut self,
        board: &Board,
        depth: u32,
        alpha0: i32,
        beta0: i32,
    ) -> Result<SearchResult, SearchAbort> {
        self.poll_abort()?;
        let mut alpha = alpha0;
        let beta = beta0;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        if self.threads > 1 && depth >= 4 {
            return self.search_depth_internal(board, depth);
        }

        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.refresh(board);
            }
        }
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return Ok(SearchResult { depth: 0,
                bestmove: None,
                score_cp: self.eval_terminal(board, 0),
                nodes: self.nodes,
            });
        }
        if self.rule_draw(board) {
            return Ok(SearchResult { depth: 0,
                bestmove: moves.first().map(|mv| format!("{mv}")),
                score_cp: DRAW_SCORE,
                nodes: self.nodes,
            });
        }
        let mut tt_move = None;
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                    tt_move = Some(mv);
                }
            }
        }
        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White {
                cozy_chess::Color::Black
            } else {
                cozy_chess::Color::White
            };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0;
            for sq in opp_bb {
                occ_mask |= 1u64 << (sq as usize);
            }

            // Build scored tuples (once per move, not once per comparison)
            let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
            for &m in &moves {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures {
                    if (occ_mask & bit) != 0 {
                        1
                    } else {
                        0
                    }
                } else {
                    0
                };

                // Pre-compute gives_check (one clone per move, not per comparison)
                let gives_check_bonus = {
                    let mut c = board.clone();
                    c.play_unchecked(m);
                    if !(c.checkers()).is_empty() {
                        30
                    } else {
                        0
                    }
                };

                let mi = move_index(m);
                let hist = if self.use_history {
                    self.history_table.get(mi).copied().unwrap_or(0)
                } else {
                    0
                };
                let mvv = if is_cap == 1 {
                    mvv_lva_score(board, m)
                } else {
                    0
                };
                let kb = if self.use_killers {
                    self.killer_bonus(0, m)
                } else {
                    0
                };
                let is_tt = tt_move == Some(m);
                let sort_score = if is_tt {
                    10_000_000
                } else if is_cap == 1 {
                    let see_gain = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    let cap_hist = if self.use_history {
                        let p = board.piece_on(m.from).unwrap();
                        let victim = board.piece_on(m.to).unwrap_or(cozy_chess::Piece::Pawn);
                        self.capture_history[capture_history_index(p, m.to, victim)] / 16
                    } else {
                        0
                    };
                    if see_gain >= 0 {
                        1_000_000 + mvv * 10 + cap_hist + see_gain + gives_check_bonus
                    } else {
                        -1_000_000 + cap_hist + see_gain + gives_check_bonus
                    }
                } else {
                    let killer_weight = if kb > 0 { kb * 1000 } else { 0 };
                    (killer_weight + gives_check_bonus + hist).clamp(-500_000, 500_000)
                };
                scored.push((m, -sort_score));
            }

            // Sort by pre-computed scores
            scored.sort_by_key(|&(_, score)| score);
            moves = scored.into_iter().map(|(m, _)| m).collect();
        }
        for m in moves.into_iter() {
            let mut child = board.clone();
            child.play_unchecked(m);
            let mut change = None;
            if self.use_nnue {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    change = Some(qn.apply_move(board, m, &child));
                }
            }
            let gives_check = !(child.checkers()).is_empty();
            let next_depth = depth.saturating_sub(1) + if gives_check { 1 } else { 0 };
            let moving_piece = board.piece_on(m.from).unwrap();
            if !self.move_stack.is_empty() {
                self.move_stack[0] = Some((moving_piece, m.to));
            }
            self.search_history.push(child.clone());
            let child_score =
                self.alphabeta(&child, next_depth, -beta, -alpha, 1, move_index(m), true);
            self.search_history.pop();
            if !self.move_stack.is_empty() {
                self.move_stack[0] = None;
            }
            if let Some(ch) = change {
                if let Some(qn) = self.nnue_quant.as_mut() {
                    qn.revert(ch);
                }
            }
            let score = -child_score?;
            if score > best_score {
                best_score = score;
                bestmove = Some(m);
            }
            if score > alpha {
                alpha = score;
            }
        }
        let bestmove_uci = bestmove.map(|m| format!("{m}"));
        Ok(SearchResult { depth: 0,
            bestmove: bestmove_uci,
            score_cp: best_score,
            nodes: self.nodes,
        })
    }

    fn is_capture(&self, board: &Board, m: Move) -> bool {
        let opp = if board.side_to_move() == cozy_chess::Color::White {
            cozy_chess::Color::Black
        } else {
            cozy_chess::Color::White
        };
        board.color_on(m.to) == Some(opp)
            || (board.piece_on(m.from) == Some(cozy_chess::Piece::Pawn)
                && board.piece_on(m.to).is_none()
                && (m.from as usize % 8) != (m.to as usize % 8))
    }

    fn update_killers(&mut self, ply: i32, m: Move) {
        let p = ply as usize;
        if p >= self.killers.len() {
            return;
        }
        let slot = &mut self.killers[p];
        if slot[0] == Some(m) {
            return;
        }
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
        if p >= self.killers.len() {
            return 0;
        }
        let slot = &self.killers[p];
        if slot[0] == Some(m) {
            50
        } else if slot[1] == Some(m) {
            30
        } else {
            0
        }
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

    pub fn debug_order_root(&self, board: &Board) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml {
                moves.push(m);
            }
            false
        });
        if moves.is_empty() {
            return moves;
        }

        let mut tt_move = None;
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos);
                    moves.insert(0, mv);
                    tt_move = Some(mv);
                }
            }
        }

        if self.order_captures || self.use_history || self.use_killers {
            let opp = if board.side_to_move() == cozy_chess::Color::White {
                cozy_chess::Color::Black
            } else {
                cozy_chess::Color::White
            };
            let opp_bb = board.colors(opp);
            let mut occ_mask: u64 = 0;
            for sq in opp_bb {
                occ_mask |= 1u64 << (sq as usize);
            }

            let mut scored: Vec<(Move, i32)> = Vec::with_capacity(moves.len());
            for &m in &moves {
                let to_sq: Square = m.to;
                let bit = 1u64 << (to_sq as usize);
                let is_cap = if self.order_captures && (occ_mask & bit) != 0 {
                    1
                } else {
                    0
                };
                let mvv = if is_cap == 1 {
                    mvv_lva_score(board, m)
                } else {
                    0
                };
                let gives_check_bonus = {
                    let mut child = board.clone();
                    child.play_unchecked(m);
                    if !(child.checkers()).is_empty() {
                        30
                    } else {
                        0
                    }
                };
                let mi = move_index(m);
                let hist = if self.use_history {
                    self.history_table.get(mi).copied().unwrap_or(0)
                } else {
                    0
                };
                let kb = if self.use_killers {
                    self.killer_bonus(0, m)
                } else {
                    0
                };
                let is_tt = tt_move == Some(m);
                let sort_score = if is_tt {
                    10_000_000
                } else if is_cap == 1 {
                    let see_gain = crate::search::see::see_gain_cp(board, m).unwrap_or(0);
                    let cap_hist = if self.use_history {
                        let p = board.piece_on(m.from).unwrap();
                        let victim = board.piece_on(m.to).unwrap_or(cozy_chess::Piece::Pawn);
                        self.capture_history[capture_history_index(p, m.to, victim)] / 16
                    } else {
                        0
                    };
                    if see_gain >= 0 {
                        1_000_000 + mvv * 10 + cap_hist + see_gain + gives_check_bonus
                    } else {
                        -1_000_000 + cap_hist + see_gain + gives_check_bonus
                    }
                } else {
                    let killer_weight = if kb > 0 { kb * 1000 } else { 0 };
                    (killer_weight + gives_check_bonus + hist).clamp(-500_000, 500_000)
                };
                scored.push((m, -sort_score));
            }

            scored.sort_by_key(|&(_, score)| score);
            moves = scored.into_iter().map(|(m, _)| m).collect();
        }
        moves
    }

    pub fn get_threads(&self) -> usize {
        self.threads
    }
    pub fn last_depth(&self) -> u32 {
        self.last_depth
    }
    pub fn last_seldepth(&self) -> u32 {
        self.max_seldepth
    }

    pub fn set_use_nnue(&mut self, on: bool) {
        self.use_nnue = on;
    }
    pub fn set_nnue_network(&mut self, nn: Option<crate::eval::nnue::Nnue>) {
        self.nnue = nn;
    }
    pub fn set_nnue_quant_model(&mut self, model: QuantNnue) {
        self.nnue_quant = Some(QuantNetwork::new(model));
    }
    pub fn clear_nnue_quant(&mut self) {
        self.nnue_quant = None;
    }
    pub fn set_eval_blend_percent(&mut self, p: u8) {
        self.eval_blend_percent = p.min(100);
    }
    pub fn search_movetime_lazy_smp(
        &mut self,
        board: &Board,
        millis: u64,
        depth: u32,
    ) -> (Option<String>, i32, u64) {
        self.search_movetime(board, millis, depth)
    }

    #[inline]
    fn nnue_eval_cp(&self, board: &Board) -> Option<i32> {
        if !self.use_nnue {
            return None;
        }
        let raw = if let Some(qn) = &self.nnue_quant {
            // Hot path: rely on refreshed/incremental accumulator state.
            Some(qn.eval_current())
        } else {
            self.nnue.as_ref().map(|nn| nn.evaluate(board))
        }?;
        Some(if board.side_to_move() == cozy_chess::Color::White {
            raw
        } else {
            -raw
        })
    }

    #[inline]
    fn blend_pst_nnue(&self, pst_cp: i32, nnue_cp: i32) -> i32 {
        let nnue_w = self.eval_blend_percent as i32;
        let pst_w = 100 - nnue_w;
        (nnue_cp * nnue_w + pst_cp * pst_w) / 100
    }

    #[inline]
    fn eval_current(&self, board: &Board) -> i32 {
        match self.eval_mode {
            EvalMode::Material => material_eval_cp(board),
            EvalMode::Pst => {
                let pst = eval_cp(board);
                if let Some(nnue) = self.nnue_eval_cp(board) {
                    self.blend_pst_nnue(pst, nnue)
                } else {
                    pst
                }
            }
            EvalMode::Nnue => {
                let pst = eval_cp(board);
                if let Some(nnue) = self.nnue_eval_cp(board) {
                    self.blend_pst_nnue(pst, nnue)
                } else {
                    pst
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EvalMode {
    Material,
    Pst,
    Nnue,
}

#[cfg(test)]
mod draw_policy_regressions {
    use super::*;

    #[test]
    fn real_children_of_a_synthetic_null_node_restore_draw_checks() {
        // `check_draws = false` represents the synthetic null position itself.
        // Every real rook move reaches halfmove 100 and must restore normal
        // fifty-move adjudication in the child rather than inheriting the
        // synthetic-node exception through the whole subtree.
        let board = Board::from_fen("k7/8/8/8/8/8/7R/7K w - - 99 1", false).unwrap();
        let mut searcher = Searcher::default();
        searcher.prepare_root_state(&board);

        let score = searcher
            .alphabeta(&board, 1, -MATE_SCORE, MATE_SCORE, 0, 0, false)
            .unwrap();

        assert_eq!(score, DRAW_SCORE);
    }

    #[test]
    fn low_clock_exact_tt_score_cannot_cut_off_near_fifty_move_draw() {
        let low_clock = Board::from_fen("k7/8/8/8/8/8/7R/7K w - - 0 1", false).unwrap();
        let near_draw = Board::from_fen("k7/8/8/8/8/8/7R/7K w - - 99 50", false).unwrap();
        let tt_move = Move {
            from: Square::H2,
            to: Square::H3,
            promotion: None,
        };
        let mut searcher = Searcher::default();
        searcher.tt_put(&low_clock, 4, 777, Some(tt_move), Bound::Exact, 0);

        // Halfmove clocks intentionally do not alter the position hash, so the
        // entry remains useful as a move-ordering hint at the near-draw node.
        assert_eq!(
            searcher.debug_order_root(&near_draw).first(),
            Some(&tt_move)
        );
        searcher.prepare_root_state(&near_draw);

        let score = searcher
            .alphabeta(&near_draw, 1, -MATE_SCORE, MATE_SCORE, 0, 0, true)
            .unwrap();

        assert_eq!(score, DRAW_SCORE);
    }

    #[test]
    fn near_draw_tt_result_cannot_poison_the_same_low_clock_position() {
        let low_clock = Board::from_fen("k7/8/8/8/8/8/7R/7K w - - 0 1", false).unwrap();
        let near_draw = Board::from_fen("k7/8/8/8/8/8/7R/7K w - - 99 50", false).unwrap();

        let mut reference = Searcher::default();
        reference.prepare_root_state(&low_clock);
        let expected = reference
            .alphabeta(&low_clock, 1, -MATE_SCORE, MATE_SCORE, 0, 0, true)
            .unwrap();
        assert_ne!(expected, DRAW_SCORE);

        let mut searcher = Searcher::default();
        searcher.prepare_root_state(&near_draw);
        assert_eq!(
            searcher
                .alphabeta(&near_draw, 1, -MATE_SCORE, MATE_SCORE, 0, 0, true)
                .unwrap(),
            DRAW_SCORE
        );
        searcher.prepare_root_state(&low_clock);
        let actual = searcher
            .alphabeta(&low_clock, 1, -MATE_SCORE, MATE_SCORE, 0, 0, true)
            .unwrap();

        assert_eq!(actual, expected);
    }
}
