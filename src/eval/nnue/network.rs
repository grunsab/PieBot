use crate::eval::nnue::features::{HalfKpA, HALFKP_PIECE_ORDER};
use crate::eval::nnue::loader::QuantNnue;
use cozy_chess::{Board, Color, Move, Piece};
use std::collections::HashSet;

/// Quantized NNUE wrapper; currently a placeholder that will be wired to the search.
pub struct QuantNetwork {
    pub model: QuantNnue,
    pub feats: HalfKpA,
    // Incremental state
    acc: Vec<i32>,
    active: HashSet<usize>,
    wk_idx: usize,
    bk_idx: usize,
}

pub enum ChangeSet {
    Delta {
        added: Vec<usize>,
        removed: Vec<usize>,
    },
    Snapshot {
        acc: Vec<i32>,
        active: HashSet<usize>,
        wk_idx: usize,
        bk_idx: usize,
    },
}

impl QuantNetwork {
    pub fn new(model: QuantNnue) -> Self {
        let feats = HalfKpA;
        let dim = feats.dim();
        assert_eq!(
            model.meta.input_dim, dim,
            "Quant model input_dim must equal HalfKP dim"
        );
        let acc = vec![0i32; model.meta.hidden_dim];
        Self {
            model,
            feats,
            acc,
            active: HashSet::new(),
            wk_idx: 0,
            bk_idx: 0,
        }
    }

    pub fn refresh(&mut self, board: &Board) {
        // Recompute active set and accumulators from scratch
        let act = self.feats.active_indices(board);
        self.active.clear();
        self.active.extend(act.iter().copied());
        self.wk_idx = square_index(board, Color::White, Piece::King);
        self.bk_idx = square_index(board, Color::Black, Piece::King);
        // accum = b1 + sum_w1(active)
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;
        for j in 0..h {
            self.acc[j] = self.model.b1[j] as i32;
        }
        for &idx in &self.active {
            let base = idx; // column
            for j in 0..h {
                let w = self.model.w1[j * n + base] as i32;
                self.acc[j] += w;
            }
        }
    }

    pub fn eval_current(&self) -> i32 {
        self.eval_from_acc()
    }

    pub fn eval_full(&self, board: &Board) -> i32 {
        // Full recompute path; used for parity testing
        let act = self.feats.active_indices(board);
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;
        let mut y = vec![0i32; h];
        for j in 0..h {
            y[j] = self.model.b1[j] as i32;
        }
        for &idx in &act {
            for j in 0..h {
                y[j] += self.model.w1[j * n + idx] as i32;
            }
        }
        // ReLU and head
        let mut out: i64 = self.model.b2[0] as i64;
        for j in 0..h {
            let v = y[j].max(0) as i64;
            out += (self.model.w2[j] as i64) * v;
        }
        out as i32
    }

    pub fn apply_move(&mut self, before: &Board, _mv: Move, after: &Board) -> ChangeSet {
        // If either king moved, snapshot + refresh for safety
        let wk_before = square_index(before, Color::White, Piece::King);
        let bk_before = square_index(before, Color::Black, Piece::King);
        let wk_after = square_index(after, Color::White, Piece::King);
        let bk_after = square_index(after, Color::Black, Piece::King);
        if wk_before != wk_after || bk_before != bk_after {
            let snap = ChangeSet::Snapshot {
                acc: self.acc.clone(),
                active: self.active.clone(),
                wk_idx: self.wk_idx,
                bk_idx: self.bk_idx,
            };
            self.refresh(after);
            return snap;
        }
        // Diff-based update for non-king moves (handles promotions, ep, captures) by recomputing active sets
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;

        let before_set = self.active.clone();
        let mut after_set: HashSet<usize> = HashSet::new();
        for (side, k_idx) in [(Color::White, wk_after), (Color::Black, bk_after)] {
            for (pi, p) in HALFKP_PIECE_ORDER.iter().enumerate() {
                let bb = after.colors(side) & after.pieces(*p);
                for sq in bb {
                    let sq_idx = sq as usize;
                    let idx = (((if side == Color::White { 0 } else { 1 }) * 64 + k_idx)
                        * HALFKP_PIECE_ORDER.len()
                        + pi)
                        * 64
                        + sq_idx;
                    after_set.insert(idx);
                }
            }
        }
        let removed: Vec<usize> = before_set.difference(&after_set).copied().collect();
        let added: Vec<usize> = after_set.difference(&before_set).copied().collect();
        // Apply removals
        for idx in &removed {
            if self.active.remove(idx) {
                for j in 0..h {
                    self.acc[j] -= self.model.w1[j * n + *idx] as i32;
                }
            }
        }
        // Apply additions
        for idx in &added {
            if self.active.insert(*idx) {
                for j in 0..h {
                    self.acc[j] += self.model.w1[j * n + *idx] as i32;
                }
            }
        }
        ChangeSet::Delta { added, removed }
    }

    pub fn revert(&mut self, change: ChangeSet) {
        match change {
            ChangeSet::Snapshot {
                acc,
                active,
                wk_idx,
                bk_idx,
            } => {
                self.acc = acc;
                self.active = active;
                self.wk_idx = wk_idx;
                self.bk_idx = bk_idx;
            }
            ChangeSet::Delta { added, removed } => {
                let h = self.model.meta.hidden_dim;
                let n = self.model.meta.input_dim;
                // Undo additions by subtracting
                for idx in added {
                    if self.active.remove(&idx) {
                        for j in 0..h {
                            self.acc[j] -= self.model.w1[j * n + idx] as i32;
                        }
                    }
                }
                // Undo removals by adding back
                for idx in removed {
                    if self.active.insert(idx) {
                        for j in 0..h {
                            self.acc[j] += self.model.w1[j * n + idx] as i32;
                        }
                    }
                }
            }
        }
    }

    fn eval_from_acc(&self) -> i32 {
        let h = self.model.meta.hidden_dim;
        let mut out: i64 = self.model.b2[0] as i64;
        for j in 0..h {
            let v = self.acc[j].max(0) as i64;
            out += (self.model.w2[j] as i64) * v;
        }
        out as i32
    }
}

fn square_index(board: &Board, side: Color, piece: Piece) -> usize {
    let sq = (board.colors(side) & board.pieces(piece))
        .into_iter()
        .next()
        .unwrap();
    // Cozy-chess Square implements Into<u8>, returns 0-63 in row-major order
    sq as usize
}

fn active_indices_side_diff(board: &Board, side: Color, k_idx: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(16);
    for (pi, p) in HALFKP_PIECE_ORDER.iter().enumerate() {
        let bb = board.colors(side) & board.pieces(*p);
        for sq in bb {
            let sq_idx = sq as usize;
            let idx = (((if side == Color::White { 0 } else { 1 }) * 64 + k_idx)
                * HALFKP_PIECE_ORDER.len()
                + pi)
                * 64
                + sq_idx;
            out.push(idx);
        }
    }
    out
}
