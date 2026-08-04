use crate::eval::nnue::features::{HalfKpSchema, PieceFeatureIndices};
use crate::eval::nnue::loader::QuantNnue;
use cozy_chess::{Board, Color, Move, Piece, Square};

const MAX_MOVE_FEATURES: usize = 4;

/// Quantized NNUE wrapper with full refresh + incremental apply/revert support.
pub struct QuantNetwork {
    pub model: QuantNnue,
    pub schema: HalfKpSchema,
    // Incremental state
    acc: Vec<i32>,
    wk_idx: usize,
    bk_idx: usize,
}

/// Opaque undo token returned by [`QuantNetwork::apply_move`].
pub struct ChangeSet(ChangeKind);

enum ChangeKind {
    Delta {
        added: FeatureDelta,
        removed: FeatureDelta,
    },
    Snapshot {
        acc: Vec<i32>,
        wk_idx: usize,
        bk_idx: usize,
    },
}

#[derive(Clone, Copy)]
struct FeatureDelta {
    indices: [usize; MAX_MOVE_FEATURES],
    len: u8,
}

impl FeatureDelta {
    #[inline]
    const fn new() -> Self {
        Self {
            indices: [0; MAX_MOVE_FEATURES],
            len: 0,
        }
    }

    #[inline]
    fn extend(&mut self, features: PieceFeatureIndices) {
        let start = self.len as usize;
        let end = start + features.len();
        debug_assert!(end <= MAX_MOVE_FEATURES);
        self.indices[start..end].copy_from_slice(features.as_slice());
        self.len = end as u8;
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len as usize]
    }
}

impl QuantNetwork {
    pub fn new(model: QuantNnue) -> Self {
        let schema = HalfKpSchema::from_input_dim(model.meta.input_dim).unwrap_or_else(|| {
            panic!(
                "Quant model input_dim {} must match a supported HalfKP schema",
                model.meta.input_dim
            )
        });
        let acc = vec![0i32; model.meta.hidden_dim];
        Self {
            model,
            schema,
            acc,
            wk_idx: 0,
            bk_idx: 0,
        }
    }

    pub fn refresh(&mut self, board: &Board) {
        // Recompute active features and accumulators from scratch.
        let act = self.schema.active_indices(board);
        self.wk_idx = square_index(board, Color::White, Piece::King);
        self.bk_idx = square_index(board, Color::Black, Piece::King);
        // accum = b1 + sum_w1(active).  Iterate by output row so each row of
        // the row-major weight matrix is reused while it is cache-hot.
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;
        for j in 0..h {
            let row = &self.model.w1[j * n..(j + 1) * n];
            let mut sum = self.model.b1[j] as i32;
            for &idx in &act {
                sum += row[idx] as i32;
            }
            self.acc[j] = sum;
        }
    }

    pub fn eval_current(&self) -> i32 {
        self.eval_from_acc()
    }

    pub fn eval_full(&self, board: &Board) -> i32 {
        // Full recompute path; used for parity testing
        let act = self.schema.active_indices(board);
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;
        let mut y = vec![0i32; h];
        for j in 0..h {
            let row = &self.model.w1[j * n..(j + 1) * n];
            let mut sum = self.model.b1[j] as i32;
            for &idx in &act {
                sum += row[idx] as i32;
            }
            y[j] = sum;
        }
        // ReLU and head
        let mut out: i64 = self.model.b2[0] as i64;
        for j in 0..h {
            let v = y[j].max(0) as i64;
            out += (self.model.w2[j] as i64) * v;
        }
        self.scale_output(out)
    }

    pub fn apply_move(&mut self, before: &Board, mv: Move, after: &Board) -> ChangeSet {
        // King moves re-key many HalfKP features. They are rare enough that a
        // full refresh is both simpler and safer (and includes Chess960
        // castling, where the encoded destination is the rook square).
        let wk_before = square_index(before, Color::White, Piece::King);
        let bk_before = square_index(before, Color::Black, Piece::King);
        let wk_after = square_index(after, Color::White, Piece::King);
        let bk_after = square_index(after, Color::Black, Piece::King);
        if wk_before != wk_after
            || bk_before != bk_after
            || self.wk_idx != wk_before
            || self.bk_idx != bk_before
        {
            return self.snapshot_and_refresh(after);
        }

        let Some(moving_piece) = before.piece_on(mv.from) else {
            return self.snapshot_and_refresh(after);
        };
        let Some(moving_color) = before.color_on(mv.from) else {
            return self.snapshot_and_refresh(after);
        };
        if moving_piece == Piece::King || moving_color != before.side_to_move() {
            return self.snapshot_and_refresh(after);
        }

        let placed_piece = mv.promotion.unwrap_or(moving_piece);
        if after.piece_on(mv.to) != Some(placed_piece)
            || after.color_on(mv.to) != Some(moving_color)
            || after.piece_on(mv.from).is_some()
        {
            return self.snapshot_and_refresh(after);
        }

        let mut removed = FeatureDelta::new();
        let mut added = FeatureDelta::new();
        removed.extend(self.piece_features(moving_color, moving_piece, mv.from));
        added.extend(self.piece_features(moving_color, placed_piece, mv.to));

        if let Some(captured_piece) = before.piece_on(mv.to) {
            let Some(captured_color) = before.color_on(mv.to) else {
                return self.snapshot_and_refresh(after);
            };
            if captured_piece == Piece::King || captured_color == moving_color {
                return self.snapshot_and_refresh(after);
            }
            removed.extend(self.piece_features(captured_color, captured_piece, mv.to));
        } else if moving_piece == Piece::Pawn && mv.from.file() != mv.to.file() {
            // En-passant: the captured pawn is on the destination file and the
            // moving pawn's original rank.
            let captured_square = Square::new(mv.to.file(), mv.from.rank());
            let captured_color = !moving_color;
            if before.piece_on(captured_square) != Some(Piece::Pawn)
                || before.color_on(captured_square) != Some(captured_color)
                || after.piece_on(captured_square).is_some()
            {
                return self.snapshot_and_refresh(after);
            }
            removed.extend(self.piece_features(captured_color, Piece::Pawn, captured_square));
        }

        self.apply_delta(&added, &removed);
        ChangeSet(ChangeKind::Delta { added, removed })
    }

    pub fn revert(&mut self, change: ChangeSet) {
        match change.0 {
            ChangeKind::Snapshot {
                acc,
                wk_idx,
                bk_idx,
            } => {
                self.acc = acc;
                self.wk_idx = wk_idx;
                self.bk_idx = bk_idx;
            }
            ChangeKind::Delta { added, removed } => {
                self.apply_delta(&removed, &added);
            }
        }
    }

    #[inline]
    fn piece_features(&self, color: Color, piece: Piece, square: Square) -> PieceFeatureIndices {
        self.schema
            .piece_indices(self.wk_idx, self.bk_idx, color, piece, square)
    }

    #[cold]
    fn snapshot_and_refresh(&mut self, after: &Board) -> ChangeSet {
        let change = ChangeSet(ChangeKind::Snapshot {
            acc: self.acc.clone(),
            wk_idx: self.wk_idx,
            bk_idx: self.bk_idx,
        });
        self.refresh(after);
        change
    }

    #[inline]
    fn apply_delta(&mut self, added: &FeatureDelta, removed: &FeatureDelta) {
        let h = self.model.meta.hidden_dim;
        let n = self.model.meta.input_dim;
        for j in 0..h {
            let row = &self.model.w1[j * n..(j + 1) * n];
            let mut value = self.acc[j];
            for &idx in removed.as_slice() {
                value -= row[idx] as i32;
            }
            for &idx in added.as_slice() {
                value += row[idx] as i32;
            }
            self.acc[j] = value;
        }
    }

    fn eval_from_acc(&self) -> i32 {
        let h = self.model.meta.hidden_dim;
        let mut out: i64 = self.model.b2[0] as i64;
        for j in 0..h {
            let v = self.acc[j].max(0) as i64;
            out += (self.model.w2[j] as i64) * v;
        }
        self.scale_output(out)
    }

    #[inline]
    fn scale_output(&self, raw: i64) -> i32 {
        let s = self.model.w1_scale * self.model.w2_scale;
        if s.is_finite() && s > 0.0 {
            ((raw as f32) * s).round() as i32
        } else {
            raw as i32
        }
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
