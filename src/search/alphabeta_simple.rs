use cozy_chess::{Board, Move, Square};
use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::search::tt::{Tt, Entry, Bound};
use crate::search::zobrist;
use crate::search::eval::{eval_cp, material_eval_cp, MATE_SCORE, DRAW_SCORE};

#[derive(Clone, Copy, Debug)]
pub enum EvalMode { Material, Pst }

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchParams {
    pub depth: u32,
    pub use_tt: bool,
    pub max_nodes: Option<u64>,
    pub movetime: Option<Duration>,
    pub order_captures: bool,
}

#[derive(Default, Debug, Clone)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub score_cp: i32,
    pub nodes: u64,
}

pub struct Searcher {
    tt: Arc<Tt>,
    nodes: u64,
    node_limit: u64,
    deadline: Option<Instant>,
    eval_mode: EvalMode,
    order_captures: bool,
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
            eval_mode: EvalMode::Pst,
            order_captures: true,
        }
    }
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
fn is_en_passant(board: &Board, m: Move) -> bool {
    // EP: pawn moves diagonally to empty square
    if let Some((_, p)) = piece_at(board, m.from) {
        if p == cozy_chess::Piece::Pawn {
            // file changed and target square empty
            let from = format!("{}", m.from);
            let to = format!("{}", m.to);
            let file_diff = from.as_bytes()[0] != to.as_bytes()[0];
            if file_diff && piece_at(board, m.to).is_none() {
                return true;
            }
        }
    }
    false
}

#[inline]
fn is_capture(board: &Board, m: Move) -> bool {
    // Direct capture if destination has opponent piece
    if let Some((col_to, _)) = piece_at(board, m.to) {
        return col_to != board.side_to_move();
    }
    // En passant special case
    is_en_passant(board, m)
}

#[inline]
fn mvv_lva_score(board: &Board, m: Move) -> i32 {
    let to = m.to; let from = m.from;
    let victim = piece_at(board, to).map(|(_, p)| piece_value_cp(p)).unwrap_or(0);
    let attacker = piece_at(board, from).map(|(_, p)| piece_value_cp(p)).unwrap_or(0);
    victim * 10 - attacker
}

impl Searcher {
    pub fn set_tt_capacity_mb(&mut self, mb: usize) {
        let mut tt = Tt::new();
        tt.set_capacity_mb(mb);
        self.tt = Arc::new(tt);
    }

    pub fn set_eval_mode(&mut self, mode: EvalMode) { self.eval_mode = mode; }
    pub fn set_order_captures(&mut self, on: bool) { self.order_captures = on; }

    #[inline]
    fn eval_current(&self, board: &Board) -> i32 {
        match self.eval_mode {
            EvalMode::Material => material_eval_cp(board),
            EvalMode::Pst => eval_cp(board),
        }
    }

    fn tt_key(board: &Board) -> u64 { zobrist::compute(board) }
    fn tt_get(&self, board: &Board) -> Option<Entry> { self.tt.get(Self::tt_key(board)) }
    fn tt_put(&mut self, board: &Board, depth: u32, score: i32, best: Option<Move>, bound: Bound) {
        let e = Entry { key: Self::tt_key(board), depth, score, best, bound, gen: 0 };
        self.tt.put(e);
    }

    pub fn qsearch_eval_cp(&mut self, board: &Board) -> i32 {
        self.qsearch(board, -MATE_SCORE, MATE_SCORE)
    }

    fn qsearch(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval_current(board); } }
        // Terminal: no legal moves
        {
            let mut has_legal = false;
            board.generate_moves(|_| { has_legal = true; true });
            if !has_legal { return self.eval_terminal(board, 0); }
        }

        let in_check = !(board.checkers()).is_empty();
        if !in_check {
            // stand pat
            let stand = self.eval_current(board);
            if stand >= beta { return beta; }
            if stand > alpha { alpha = stand; }
        }

        // Captures (and EP) first
        let mut caps: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| {
            for m in ml { if is_capture(board, m) { caps.push(m); } }
            false
        });
        caps.sort_by_key(|&m| -mvv_lva_score(board, m));
        for m in caps {
            let mut child = board.clone(); child.play(m);
            let score = -self.qsearch(&child, -beta, -alpha);
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }

        if in_check {
            // Explore evasions conservatively (all legal moves) when in check
            let mut moves: Vec<Move> = Vec::with_capacity(64);
            board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
            for m in moves {
                let mut child = board.clone(); child.play(m);
                let score = -self.qsearch(&child, -beta, -alpha);
                if score >= beta { return beta; }
                if score > alpha { alpha = score; }
            }
        }
        alpha
    }

    pub fn search_with_params(&mut self, board: &Board, params: SearchParams) -> SearchResult {
        self.nodes = 0;
        self.node_limit = params.max_nodes.unwrap_or(u64::MAX);
        if !params.use_tt { self.tt = Arc::new(Tt::new()); }
        self.deadline = params.movetime.map(|d| Instant::now() + d);
        self.order_captures = params.order_captures;
        let max_depth = if params.depth == 0 { 99 } else { params.depth };
        let mut best: Option<String> = None;
        let mut last_score = 0;
        for d in 1..=max_depth {
            let r = self.search_depth(board, d);
            best = r.bestmove.clone();
            last_score = r.score_cp;
            if self.nodes >= self.node_limit { break; }
            if let Some(dl) = self.deadline { if Instant::now() >= dl { break; } }
        }
        if best.is_none() {
            let mut mv: Option<Move> = None;
            board.generate_moves(|ml| { for m in ml { mv = Some(m); break; } mv.is_some() });
            if let Some(m) = mv { best = Some(format!("{}", m)); }
        }
        SearchResult { bestmove: best, score_cp: last_score, nodes: self.nodes }
    }

    pub fn search_movetime(&mut self, board: &Board, millis: u64, depth: u32) -> (Option<String>, i32, u64) {
        let mut p = SearchParams::default();
        p.depth = depth; p.use_tt = true; p.movetime = Some(Duration::from_millis(millis)); p.order_captures = true;
        let r = self.search_with_params(board, p);
        (r.bestmove, r.score_cp, r.nodes)
    }

    fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        let mut alpha = -MATE_SCORE;
        let beta = MATE_SCORE;
        let mut bestmove: Option<Move> = None;
        let mut best_score = -MATE_SCORE;

        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return SearchResult { bestmove: None, score_cp: self.eval_terminal(board, 0), nodes: self.nodes }; }

        // TT move first (trust Exact only)
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best { if matches!(en.bound, Bound::Exact) {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) {
                    let mv = moves.remove(pos); moves.insert(0, mv);
                }
            } }
        }

        // Simple ordering: captures first by MVV-LVA
        if self.order_captures {
            moves.sort_by_key(|&m| {
                let cap = if is_capture(board, m) { 1 } else { 0 };
                let mvv = if cap == 1 { mvv_lva_score(board, m) } else { 0 };
                -(cap * 1000 + mvv)
            });
        }

        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let score = -self.alphabeta(&child, depth - 1, -beta, -alpha, 1);
            if score > best_score { best_score = score; bestmove = Some(m); }
            if score > alpha { alpha = score; }
        }
        let bestmove_uci = bestmove.map(|m| format!("{}", m));
        SearchResult { bestmove: bestmove_uci, score_cp: best_score, nodes: self.nodes }
    }

    fn alphabeta(&mut self, board: &Board, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        if self.nodes >= self.node_limit { return self.eval_current(board); }
        if let Some(dl) = self.deadline { if Instant::now() >= dl { return self.eval_current(board); } }

        // In-check horizon extension (extend depth 0 by 1)
        let in_check_now = !(board.checkers()).is_empty();
        let depth = if depth == 0 && in_check_now { 1 } else { depth };
        if depth == 0 { return self.qsearch(board, alpha, beta); }

        // TT probe
        if let Some(en) = self.tt_get(board) {
            if en.depth >= depth {
                match en.bound {
                    Bound::Exact => return en.score,
                    Bound::Lower => if en.score >= beta { return en.score; },
                    Bound::Upper => if en.score <= alpha { return en.score; },
                }
            }
        }

        // Generate and order
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if moves.is_empty() { return self.eval_terminal(board, ply); }
        // TT move first (trusted exact only)
        if let Some(en) = self.tt_get(board) {
            if let Some(ttm) = en.best { if matches!(en.bound, Bound::Exact) {
                if let Some(pos) = moves.iter().position(|&mv| mv == ttm) { let mv = moves.remove(pos); moves.insert(0, mv); }
            } }
        }
        if self.order_captures {
            moves.sort_by_key(|&m| {
                let cap = if is_capture(board, m) { 1 } else { 0 };
                let mvv = if cap == 1 { mvv_lva_score(board, m) } else { 0 };
                -(cap * 1000 + mvv)
            });
        }

        let mut best = -MATE_SCORE;
        let mut best_move_local: Option<Move> = None;
        let orig_alpha = alpha;
        for m in moves.into_iter() {
            let mut child = board.clone(); child.play(m);
            let score = -self.alphabeta(&child, depth - 1, -beta, -alpha, ply + 1);
            if score > best { best = score; best_move_local = Some(m); }
            if best > alpha { alpha = best; }
            if alpha >= beta { break; }
        }
        let bound = if best <= orig_alpha { Bound::Upper } else if best >= beta { Bound::Lower } else { Bound::Exact };
        self.tt_put(board, depth, best, best_move_local, bound);
        best
    }

    fn eval_terminal(&self, board: &Board, ply: i32) -> i32 {
        if !(board.checkers()).is_empty() { return -MATE_SCORE + ply; }
        DRAW_SCORE
    }

    // --- Debug helpers for tests (parity with baseline) ---
    pub fn debug_order_for_parent(&self, board: &Board) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();
        board.generate_moves(|ml| { for m in ml { moves.push(m); } false });
        if self.order_captures {
            moves.sort_by_key(|&m| {
                let cap = if is_capture(board, m) { 1 } else { 0 };
                let mvv = if cap == 1 { mvv_lva_score(board, m) } else { 0 };
                -(cap * 1000 + mvv)
            });
        }
        moves
    }
}
