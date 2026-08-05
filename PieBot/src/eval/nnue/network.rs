use crate::eval::nnue::features::{HalfKpSchema, PieceFeatureIndices};
use crate::eval::nnue::loader::QuantNnue;
use cozy_chess::{Board, Color, Move, Piece, Square};
use std::sync::Arc;

const MAX_MOVE_FEATURES: usize = 4;

/// Quantized NNUE wrapper with full refresh + incremental apply/revert support.
///
/// The immutable source model is intentionally not mutable through the runtime
/// wrapper: changing it would invalidate the feature-major cache.
///
/// ```compile_fail
/// use piebot::eval::nnue::network::QuantNetwork;
/// use std::sync::Arc;
///
/// fn invalidate_runtime_cache(network: &mut QuantNetwork) {
///     let _ = Arc::get_mut(&mut network.model);
/// }
/// ```
pub struct QuantNetwork {
    model: Arc<QuantNnue>,
    pub schema: HalfKpSchema,
    // PIENNQ01 stores W1 hidden-major for compatibility with the dense
    // exporter. Incremental updates instead need all hidden weights for one
    // active feature, so keep a feature-major runtime copy to make each delta
    // a pair of contiguous 64-byte reads rather than 64 cache-line-strided
    // reads through the multi-megabyte model.
    w1_feature_major: Arc<[i8]>,
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
        let w1_feature_major =
            transpose_w1_feature_major(&model.w1, model.meta.input_dim, model.meta.hidden_dim)
                .into();
        let acc = vec![0i32; model.meta.hidden_dim];
        let model = Arc::new(model);
        Self {
            model,
            schema,
            w1_feature_major,
            acc,
            wk_idx: 0,
            bk_idx: 0,
        }
    }

    /// Clone mutable accumulator state for a search worker while sharing both
    /// immutable allocations: the source model and runtime-transposed W1.
    pub(crate) fn clone_for_search(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            schema: self.schema,
            w1_feature_major: Arc::clone(&self.w1_feature_major),
            acc: self.acc.clone(),
            wk_idx: self.wk_idx,
            bk_idx: self.bk_idx,
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
        for &idx in removed.as_slice() {
            let start = idx * h;
            let weights = &self.w1_feature_major[start..start + h];
            for (value, &weight) in self.acc.iter_mut().zip(weights) {
                *value -= weight as i32;
            }
        }
        for &idx in added.as_slice() {
            let start = idx * h;
            let weights = &self.w1_feature_major[start..start + h];
            for (value, &weight) in self.acc.iter_mut().zip(weights) {
                *value += weight as i32;
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

fn transpose_w1_feature_major(row_major: &[i8], input_dim: usize, hidden_dim: usize) -> Vec<i8> {
    assert_eq!(row_major.len(), input_dim * hidden_dim);
    let mut feature_major = vec![0; row_major.len()];
    for hidden in 0..hidden_dim {
        let row = &row_major[hidden * input_dim..(hidden + 1) * input_dim];
        for (feature, &weight) in row.iter().enumerate() {
            feature_major[feature * hidden_dim + hidden] = weight;
        }
    }
    feature_major
}

fn square_index(board: &Board, side: Color, piece: Piece) -> usize {
    let sq = (board.colors(side) & board.pieces(piece))
        .into_iter()
        .next()
        .unwrap();
    // Cozy-chess Square implements Into<u8>, returns 0-63 in row-major order
    sq as usize
}

#[cfg(test)]
mod tests {
    use super::{transpose_w1_feature_major, QuantNetwork};
    use crate::eval::nnue::features::halfkp_v2_dim;
    use crate::eval::nnue::loader::{QuantMeta, QuantNnue};
    use std::sync::Arc;

    #[test]
    fn feature_major_runtime_layout_preserves_every_weight() {
        let hidden_dim = 3;
        let input_dim = 5;
        let row_major: Vec<i8> = (0..hidden_dim * input_dim)
            .map(|value| value as i8 - 7)
            .collect();

        let feature_major = transpose_w1_feature_major(&row_major, input_dim, hidden_dim);

        for hidden in 0..hidden_dim {
            for feature in 0..input_dim {
                assert_eq!(
                    feature_major[feature * hidden_dim + hidden],
                    row_major[hidden * input_dim + feature],
                    "feature={feature} hidden={hidden}",
                );
            }
        }
    }

    #[test]
    fn search_clone_shares_immutable_allocations_without_cache_invalidation() {
        let input_dim = halfkp_v2_dim();
        let hidden_dim = 2;
        let network = QuantNetwork::new(QuantNnue {
            meta: QuantMeta {
                version: 1,
                input_dim,
                hidden_dim,
                output_dim: 1,
            },
            w1_scale: 1.0,
            w2_scale: 1.0,
            w1: vec![0; input_dim * hidden_dim],
            b1: vec![0; hidden_dim],
            w2: vec![0; hidden_dim],
            b2: vec![0],
        });

        let worker = network.clone_for_search();

        assert!(Arc::ptr_eq(&network.model, &worker.model));
        assert!(Arc::ptr_eq(
            &network.w1_feature_major,
            &worker.w1_feature_major,
        ));
    }
}
