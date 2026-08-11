use crate::eval::nnue::features::{
    dp_active_indices, dp_piece_index, HalfKpSchema, PieceFeatureIndices,
};
use crate::eval::nnue::loader::{QuantNnue, QuantNnueV2};
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
    /// Arch-v2 backend (dual-perspective accumulators, SCReLU head). When
    /// present, every public method dispatches here and the legacy fields
    /// above are inert.
    v2: Option<V2State>,
}

/// Number of plies the accumulator stack is sized for up front. Deeper lines
/// grow it on demand; 64 covers the overwhelming majority of search paths
/// while keeping the per-worker footprint small (64 * 2 * 1024 * 2 B = 256 KB
/// at production width, against 160 self-play workers on the training box).
const INITIAL_STACK_PLIES: usize = 64;

/// Runtime state for a PIENNQ02 dual-perspective model. Accumulators are
/// anchored to color (white/black), not side-to-move; the stm-first
/// concatenation happens at evaluation time from the stored `stm`.
struct V2State {
    model: Arc<QuantNnueV2>,
    /// Ply-indexed accumulator stack. Slot `t` spans `[t*2*h, (t+1)*2*h)`,
    /// white perspective first then black; `top` is the live slot.
    ///
    /// Applying a move writes a *new* slot from its parent, which makes revert
    /// an index decrement. The alternative -- mutating one accumulator in
    /// place and subtracting the delta back out -- reads each move's two 2 KB
    /// weight rows a second time, and does it after the entire child subtree
    /// has evicted them from cache. w1 is ~84 MB at h1024, so those misses are
    /// what the update actually costs.
    stack: Vec<i16>,
    top: usize,
    wk_idx: usize,
    bk_idx: usize,
    stm: Color,
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
    /// Arch-v2 pushed a new accumulator slot. Undoing is a `top` decrement
    /// plus restoring the scalar keys -- no weight rows are touched, so this
    /// is the same cost whether the slot came from a delta or a full refresh.
    PushV2 {
        prev_stm: Color,
        prev_wk: usize,
        prev_bk: usize,
    },
    /// Side-to-move-only transition (null move) on the arch-v2 backend.
    NullV2 {
        prev_stm: Color,
    },
    /// Nothing to undo. Returned for transitions the active backend does not
    /// model, so callers can stay branch-free.
    Inert,
}

#[derive(Clone, Copy)]
struct FeatureDelta {
    indices: [usize; MAX_MOVE_FEATURES],
    len: u8,
}

impl FeatureDelta {
    #[inline]
    fn push(&mut self, idx: usize) {
        self.indices[self.len as usize] = idx;
        self.len += 1;
    }

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
        if let Some(v2_model) = model.v2.clone() {
            let hidden = v2_model.hidden_dim;
            let v2 = V2State {
                model: v2_model,
                stack: vec![0i16; INITIAL_STACK_PLIES * 2 * hidden],
                top: 0,
                wk_idx: 0,
                bk_idx: 0,
                stm: Color::White,
            };
            let model = Arc::new(model);
            return Self {
                model,
                schema: HalfKpSchema::FullPerspective,
                w1_feature_major: Arc::from(Vec::new().into_boxed_slice()),
                acc: Vec::new(),
                wk_idx: 0,
                bk_idx: 0,
                v2: Some(v2),
            };
        }
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
            v2: None,
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
            v2: self.v2.as_ref().map(|v2| V2State {
                model: Arc::clone(&v2.model),
                stack: v2.stack.clone(),
                top: v2.top,
                wk_idx: v2.wk_idx,
                bk_idx: v2.bk_idx,
                stm: v2.stm,
            }),
        }
    }

    pub fn refresh(&mut self, board: &Board) {
        if let Some(v2) = &mut self.v2 {
            v2.refresh(board);
            return;
        }
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
        if let Some(v2) = &self.v2 {
            return v2.eval_from_acc();
        }
        self.eval_from_acc()
    }

    pub fn eval_full(&self, board: &Board) -> i32 {
        if let Some(v2) = &self.v2 {
            return v2.eval_full(board);
        }
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
        if self.v2.is_some() {
            return self.apply_move_v2(before, mv, after);
        }
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

    /// Record a null move: the side to move changes but no piece moves.
    ///
    /// Arch-v2 is side-to-move relative -- the head concatenates the stm
    /// perspective first and reads a different half of `w2` for each color --
    /// so it must be told, even though both accumulators are untouched.
    /// Skipping this makes the null child evaluate as the exact negation of
    /// its parent, discarding the network's tempo asymmetry at every
    /// null-move node. The legacy backend keeps a white-POV accumulator whose
    /// sign the caller applies, so it is genuinely stm-independent and this is
    /// inert there.
    pub fn apply_null_move(&mut self, after: &Board) -> ChangeSet {
        match &mut self.v2 {
            Some(v2) => {
                let prev_stm = v2.stm;
                v2.stm = after.side_to_move();
                ChangeSet(ChangeKind::NullV2 { prev_stm })
            }
            None => ChangeSet(ChangeKind::Inert),
        }
    }

    pub fn revert(&mut self, change: ChangeSet) {
        match change.0 {
            ChangeKind::Inert => {}
            ChangeKind::NullV2 { prev_stm } => {
                let v2 = self.v2.as_mut().expect("v2 change on v2 backend");
                v2.stm = prev_stm;
            }
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
            ChangeKind::PushV2 {
                prev_stm,
                prev_wk,
                prev_bk,
            } => {
                let v2 = self.v2.as_mut().expect("v2 change on v2 backend");
                debug_assert!(v2.top > 0, "v2 accumulator stack underflow on revert");
                v2.top -= 1;
                v2.stm = prev_stm;
                v2.wk_idx = prev_wk;
                v2.bk_idx = prev_bk;
            }
        }
    }

    fn apply_move_v2(&mut self, before: &Board, mv: Move, after: &Board) -> ChangeSet {
        let wk_before = square_index(before, Color::White, Piece::King);
        let bk_before = square_index(before, Color::Black, Piece::King);
        let wk_after = square_index(after, Color::White, Piece::King);
        let bk_after = square_index(after, Color::Black, Piece::King);
        let v2 = self.v2.as_mut().expect("apply_move_v2 requires v2 backend");
        if wk_before != wk_after
            || bk_before != bk_after
            || v2.wk_idx != wk_before
            || v2.bk_idx != bk_before
        {
            return v2.push_refresh(after);
        }

        let Some(moving_piece) = before.piece_on(mv.from) else {
            return v2.push_refresh(after);
        };
        let Some(moving_color) = before.color_on(mv.from) else {
            return v2.push_refresh(after);
        };
        if moving_piece == Piece::King || moving_color != before.side_to_move() {
            return v2.push_refresh(after);
        }

        let placed_piece = mv.promotion.unwrap_or(moving_piece);
        if after.piece_on(mv.to) != Some(placed_piece)
            || after.color_on(mv.to) != Some(moving_color)
            || after.piece_on(mv.from).is_some()
        {
            return v2.push_refresh(after);
        }

        let mut removed = [FeatureDelta::new(), FeatureDelta::new()];
        let mut added = [FeatureDelta::new(), FeatureDelta::new()];
        v2.push_piece(&mut removed, moving_color, moving_piece, mv.from);
        v2.push_piece(&mut added, moving_color, placed_piece, mv.to);

        if let Some(captured_piece) = before.piece_on(mv.to) {
            let Some(captured_color) = before.color_on(mv.to) else {
                return v2.push_refresh(after);
            };
            if captured_piece == Piece::King || captured_color == moving_color {
                return v2.push_refresh(after);
            }
            v2.push_piece(&mut removed, captured_color, captured_piece, mv.to);
        } else if moving_piece == Piece::Pawn && mv.from.file() != mv.to.file() {
            let captured_square = Square::new(mv.to.file(), mv.from.rank());
            let captured_color = !moving_color;
            if before.piece_on(captured_square) != Some(Piece::Pawn)
                || before.color_on(captured_square) != Some(captured_color)
                || after.piece_on(captured_square).is_some()
            {
                return v2.push_refresh(after);
            }
            v2.push_piece(&mut removed, captured_color, Piece::Pawn, captured_square);
        }

        let change = ChangeSet(ChangeKind::PushV2 {
            prev_stm: v2.stm,
            prev_wk: v2.wk_idx,
            prev_bk: v2.bk_idx,
        });
        v2.push_delta(&added, &removed);
        v2.stm = after.side_to_move();
        change
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

/// SCReLU dot product: `sum_j clamp(acc[j], 0, QA)^2 * w[j]`.
///
/// Partial sums widen to i64 every `SCRELU_CHUNK` terms so the inner loop can
/// stay in narrow lanes.
const SCRELU_CHUNK: usize = 128;

/// Largest QA for which `clamp(acc, 0, QA) * w` is exact in i16 across the full
/// int8 weight range: `qa * 128 <= i16::MAX`. The shipped models quantize at
/// QA=255, which is exactly this bound.
const MAX_I16_MADD_QA: i32 = (i16::MAX as i32) / 128;

#[inline]
fn screlu_dot(acc: &[i16], w: &[i8], qa: i32) -> i64 {
    debug_assert_eq!(acc.len(), w.len());
    debug_assert!(qa > 0);
    if qa <= MAX_I16_MADD_QA {
        screlu_dot_madd(acc, w, qa as i16)
    } else {
        screlu_dot_wide(acc, w, qa)
    }
}

/// Fast path, shaped so the autovectorizer emits a widening multiply-accumulate
/// (`vpmaddwd` on AVX2, `smlal`/`smlal2` on NEON).
///
/// The trick is keeping `v * w` in i16: the multiply then runs 16 lanes at a
/// time instead of the 8 an i32 formulation allows, and the widening i16xi16
/// -> i32 accumulate is a single instruction. Exactness rests on two bounds:
///   * `|v * w| <= QA * 128 = 32_640 <= i16::MAX`, guaranteed by the
///     `MAX_I16_MADD_QA` dispatch above;
///   * `|v * (v * w)| <= 255 * 32_640 = 8_323_200`, so `SCRELU_CHUNK = 128`
///     terms reach at most 1.065e9, well inside i32.
/// Every intermediate is therefore exact, not merely non-panicking, and the
/// result is bit-identical to `screlu_dot_wide`.
#[inline]
fn screlu_dot_madd(acc: &[i16], w: &[i8], qa: i16) -> i64 {
    let mut total: i64 = 0;
    for (acc_chunk, w_chunk) in acc.chunks(SCRELU_CHUNK).zip(w.chunks(SCRELU_CHUNK)) {
        let mut partial: i32 = 0;
        for (&a, &weight) in acc_chunk.iter().zip(w_chunk) {
            let v = a.clamp(0, qa);
            let vw = v * weight as i16;
            partial += v as i32 * vw as i32;
        }
        total += partial as i64;
    }
    total
}

/// Exact for any QA. Everything widens to i64 up front, so this cannot
/// overflow regardless of quantization scale; it is only reached by models
/// quantized above `MAX_I16_MADD_QA`, which none currently are.
#[cold]
fn screlu_dot_wide(acc: &[i16], w: &[i8], qa: i32) -> i64 {
    let qa = qa as i64;
    let mut total: i64 = 0;
    for (&a, &weight) in acc.iter().zip(w) {
        let v = (a as i64).clamp(0, qa);
        total += v * v * weight as i64;
    }
    total
}

impl V2State {
    /// Grow the stack so slot `t` exists. Called before any slice is taken, so
    /// a reallocation can never invalidate a live borrow.
    #[inline]
    fn ensure_slot(&mut self, t: usize) {
        let need = (t + 1) * 2 * self.model.hidden_dim;
        if self.stack.len() < need {
            self.stack.resize(need, 0);
        }
    }

    /// The two perspective accumulators in slot `t`, white first.
    #[inline]
    fn slot(&self, t: usize) -> (&[i16], &[i16]) {
        let h = self.model.hidden_dim;
        let base = t * 2 * h;
        (
            &self.stack[base..base + h],
            &self.stack[base + h..base + 2 * h],
        )
    }

    fn refresh(&mut self, board: &Board) {
        self.top = 0;
        self.refresh_into(0, board);
    }

    /// Rebuild both perspective accumulators from scratch into slot `t`.
    /// Does not move `top`; callers decide whether this replaces the live slot
    /// (`refresh`) or pushes a new one (`push_refresh`).
    fn refresh_into(&mut self, t: usize, board: &Board) {
        self.wk_idx = square_index(board, Color::White, Piece::King);
        self.bk_idx = square_index(board, Color::Black, Piece::King);
        self.stm = board.side_to_move();
        self.ensure_slot(t);
        let h = self.model.hidden_dim;
        let base = t * 2 * h;
        let model = &*self.model;
        for (p, perspective) in [Color::White, Color::Black].into_iter().enumerate() {
            let dst = &mut self.stack[base + p * h..base + (p + 1) * h];
            dst.copy_from_slice(&model.b1);
            for idx in dp_active_indices(board, perspective) {
                let row = &model.w1[idx * h..(idx + 1) * h];
                for (value, &weight) in dst.iter_mut().zip(row) {
                    *value = value.wrapping_add(weight);
                }
            }
        }
    }

    /// SCReLU head over the stm-first concatenated accumulators. Returns
    /// white-POV centipawns to match the legacy backend's contract; the
    /// network itself is side-to-move relative.
    fn eval_from_acc(&self) -> i32 {
        let (white, black) = self.slot(self.top);
        let out_stm = self.head(white, black, self.stm);
        if self.stm == Color::White {
            out_stm
        } else {
            -out_stm
        }
    }

    fn eval_full(&self, board: &Board) -> i32 {
        // Full recompute path; used for parity testing.
        let h = self.model.hidden_dim;
        let mut acc_white = self.model.b1.clone();
        let mut acc_black = self.model.b1.clone();
        for (perspective, acc) in [
            (Color::White, &mut acc_white),
            (Color::Black, &mut acc_black),
        ] {
            for idx in dp_active_indices(board, perspective) {
                let row = &self.model.w1[idx * h..(idx + 1) * h];
                for (value, &weight) in acc.iter_mut().zip(row) {
                    *value = value.wrapping_add(weight);
                }
            }
        }
        let stm = board.side_to_move();
        let out_stm = self.head(&acc_white, &acc_black, stm);
        if stm == Color::White {
            out_stm
        } else {
            -out_stm
        }
    }

    fn head(&self, acc_white: &[i16], acc_black: &[i16], stm: Color) -> i32 {
        let h = self.model.hidden_dim;
        let qa = self.model.qa as i64;
        let (first, second) = match stm {
            Color::White => (acc_white, acc_black),
            Color::Black => (acc_black, acc_white),
        };
        let sum = screlu_dot(first, &self.model.w2[..h], self.model.qa)
            + screlu_dot(second, &self.model.w2[h..2 * h], self.model.qa);
        // Weights carry QA^2 * QB; b2 is stored in the same domain.
        let numerator = (sum + self.model.b2 as i64) * self.model.scale as i64;
        (numerator / (qa * qa * self.model.qb as i64)) as i32
    }

    /// Append the feature this piece contributes to each perspective.
    fn push_piece(
        &self,
        deltas: &mut [FeatureDelta; 2],
        color: Color,
        piece: Piece,
        square: Square,
    ) {
        deltas[0].push(dp_piece_index(
            Color::White,
            self.wk_idx,
            color,
            piece,
            square,
        ));
        deltas[1].push(dp_piece_index(
            Color::Black,
            self.bk_idx,
            color,
            piece,
            square,
        ));
    }

    /// Write the post-move accumulators into a fresh slot above `top`, leaving
    /// the parent's slot untouched so revert is a decrement.
    fn push_delta(&mut self, added: &[FeatureDelta; 2], removed: &[FeatureDelta; 2]) {
        let h = self.model.hidden_dim;
        let top = self.top;
        self.ensure_slot(top + 1);
        let model = &*self.model;
        let w1 = &model.w1;
        // Parent slot ends exactly where the child slot begins, so one split
        // hands out a shared read of the parent and a unique write of the child.
        let (parent_slots, child_slot) = self.stack.split_at_mut((top + 1) * 2 * h);
        let parent = &parent_slots[top * 2 * h..];
        for p in 0..2 {
            let src = &parent[p * h..(p + 1) * h];
            let dst = &mut child_slot[p * h..(p + 1) * h];
            let rem = removed[p].as_slice();
            let add = added[p].as_slice();
            // A quiet move is exactly one feature out and one in per
            // perspective, which is the overwhelming majority of applications.
            // Fusing the copy with both row reads walks the accumulator once
            // and lets the two row misses overlap rather than serialize. w1 is
            // ~84 MB at h1024, so those misses, not the arithmetic, are what
            // this loop actually costs.
            if let ([r_idx], [a_idx]) = (rem, add) {
                let r = &w1[r_idx * h..(r_idx + 1) * h];
                let a = &w1[a_idx * h..(a_idx + 1) * h];
                for (((value, &s), &rw), &aw) in dst.iter_mut().zip(src).zip(r).zip(a) {
                    *value = s.wrapping_sub(rw).wrapping_add(aw);
                }
                continue;
            }
            dst.copy_from_slice(src);
            for &idx in rem {
                let row = &w1[idx * h..(idx + 1) * h];
                for (value, &weight) in dst.iter_mut().zip(row) {
                    *value = value.wrapping_sub(weight);
                }
            }
            for &idx in add {
                let row = &w1[idx * h..(idx + 1) * h];
                for (value, &weight) in dst.iter_mut().zip(row) {
                    *value = value.wrapping_add(weight);
                }
            }
        }
        self.top = top + 1;
    }

    /// King moves re-key every HalfKP feature, so the accumulators are rebuilt
    /// rather than patched. Pushing that rebuild into a new slot keeps revert
    /// uniform -- and free -- instead of cloning the parent accumulators.
    #[cold]
    fn push_refresh(&mut self, after: &Board) -> ChangeSet {
        let change = ChangeSet(ChangeKind::PushV2 {
            prev_stm: self.stm,
            prev_wk: self.wk_idx,
            prev_bk: self.bk_idx,
        });
        let t = self.top + 1;
        self.refresh_into(t, after);
        self.top = t;
        change
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
    use super::{screlu_dot, transpose_w1_feature_major, QuantNetwork};
    use crate::eval::nnue::features::halfkp_v2_dim;
    use crate::eval::nnue::loader::{QuantMeta, QuantNnue};
    use std::sync::Arc;

    /// Independent, deliberately naive definition of the SCReLU head term.
    /// Everything widens to i64 immediately, so no intermediate can overflow
    /// and this stays valid for any `qa`. The optimized kernel must reproduce
    /// it exactly.
    fn screlu_dot_reference(acc: &[i16], w: &[i8], qa: i32) -> i64 {
        let mut total: i64 = 0;
        for (&a, &weight) in acc.iter().zip(w) {
            let v = (a as i64).clamp(0, qa as i64);
            total += v * v * weight as i64;
        }
        total
    }

    /// Deterministic LCG; the kernel is integer-exact so a fixed stream is a
    /// complete test, and a fixed seed keeps failures reproducible.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn screlu_dot_matches_reference_on_adversarial_inputs() {
        // Values chosen to sit on every boundary the optimized kernel cares
        // about: accumulators below zero, at zero, inside the clamp, exactly at
        // QA and past it, and at the i16 extremes; weights at both int8 limits.
        let accs: Vec<i16> = vec![
            i16::MIN,
            -32000,
            -256,
            -1,
            0,
            1,
            127,
            254,
            255,
            256,
            1000,
            32000,
            i16::MAX,
        ];
        let weights: Vec<i8> = vec![i8::MIN, -127, -64, -1, 0, 1, 63, 126, i8::MAX];

        // Every (acc, weight) pairing, which includes the worst case for the
        // i16 intermediate: v = QA = 255 against w = -128.
        let mut acc_vec = Vec::new();
        let mut w_vec = Vec::new();
        for &a in &accs {
            for &w in &weights {
                acc_vec.push(a);
                w_vec.push(w);
            }
        }

        for qa in [1, 2, 64, 127, 254, 255] {
            assert_eq!(
                screlu_dot_reference(&acc_vec, &w_vec, qa),
                screlu_dot(&acc_vec, &w_vec, qa),
                "qa={qa}",
            );
        }
    }

    #[test]
    fn screlu_dot_matches_reference_at_production_width() {
        // 2048 lanes is the shipped shape (two hidden-1024 accumulators), so
        // this exercises whatever chunking and tail handling the kernel uses.
        let mut state = 0x5EED_1234_ABCD_0001u64;
        let acc: Vec<i16> = (0..2048)
            .map(|_| (lcg(&mut state) % 65536) as u16 as i16)
            .collect();
        let w: Vec<i8> = (0..2048).map(|_| (lcg(&mut state) % 256) as u8 as i8).collect();

        assert_eq!(
            screlu_dot_reference(&acc, &w, 255),
            screlu_dot(&acc, &w, 255),
        );
    }

    #[test]
    fn screlu_dot_matches_reference_on_ragged_lengths() {
        let mut state = 0xC0FF_EE00_1234_5678u64;
        for len in [0usize, 1, 7, 15, 16, 17, 127, 128, 129, 255, 257] {
            let acc: Vec<i16> = (0..len)
                .map(|_| (lcg(&mut state) % 65536) as u16 as i16)
                .collect();
            let w: Vec<i8> = (0..len).map(|_| (lcg(&mut state) % 256) as u8 as i8).collect();
            assert_eq!(
                screlu_dot_reference(&acc, &w, 255),
                screlu_dot(&acc, &w, 255),
                "len={len}",
            );
        }
    }

    #[test]
    fn screlu_dot_is_exact_above_the_i16_intermediate_limit() {
        // The fast path multiplies v*w in i16, which is only safe while
        // qa * 128 <= i16::MAX, i.e. qa <= 255. Models quantized with a larger
        // QA must still evaluate exactly, via whatever fallback the kernel
        // keeps. Without this the overflow would be silent.
        let acc: Vec<i16> = vec![i16::MAX, 5000, 1024, 512, 300, 256, 0, -5];
        let w: Vec<i8> = vec![i8::MIN, 127, -100, 64, -1, 0, 32, 100];
        for qa in [256, 300, 1024, 4096] {
            assert_eq!(
                screlu_dot_reference(&acc, &w, qa),
                screlu_dot(&acc, &w, qa),
                "qa={qa}",
            );
        }
    }

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
            v2: None,
        });

        let worker = network.clone_for_search();

        assert!(Arc::ptr_eq(&network.model, &worker.model));
        assert!(Arc::ptr_eq(
            &network.w1_feature_major,
            &worker.w1_feature_major,
        ));
    }
}
