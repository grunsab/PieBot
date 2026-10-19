//! Arch-v3 runtime (`PIENNQ03`, see `training/nnue/train_v3.py`).
//!
//! Each perspective's king square is oriented (ranks flipped for Black) and
//! mirrored to files a-d; together these pick a "key" (bucket * 2 + mirror).
//! Every piece, kings included, contributes one feature under that key.
//! Applying a move diffs the two boards' piece bitboards, so captures,
//! castling, en passant and promotions all go through one path; a
//! perspective is rebuilt only when its key changes, from a per-key refresh
//! cache. Slots are ply-indexed like arch-v2: applying a move records how to
//! build the child slot, and the slot is built when an evaluation needs it
//! (immediately, unless lazy mode is on).

use crate::eval::nnue::loader::QuantNnueV3;
use crate::eval::nnue::network::{
    push_delta_capture_fused, push_delta_quiet_fused_plain, screlu_dot_pub, vec_add_assign_i16,
    vec_sub_assign_i16,
};
use cozy_chess::{Board, Color, Piece};
use std::sync::Arc;

const MAX_DELTA: usize = 4;

/// Piece bitboards indexed `[color][piece]` (cozy-chess order), kings included.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Placement3 {
    pieces: [[u64; 6]; 2],
}

impl Placement3 {
    #[inline]
    pub(crate) fn of(board: &Board) -> Self {
        let mut pieces = [[0u64; 6]; 2];
        for (ci, c) in [Color::White, Color::Black].into_iter().enumerate() {
            for p in Piece::ALL {
                pieces[ci][p as usize] = board.colored_pieces(c, p).0;
            }
        }
        Self { pieces }
    }

    #[inline]
    fn king(&self, color: usize) -> usize {
        self.pieces[color][Piece::King as usize].trailing_zeros() as usize
    }

    #[inline]
    fn count(&self) -> u32 {
        self.pieces.iter().flatten().map(|b| b.count_ones()).sum()
    }
}

#[derive(Clone, Copy, Default)]
struct Delta {
    add: [u32; MAX_DELTA],
    sub: [u32; MAX_DELTA],
    na: u8,
    ns: u8,
}

#[derive(Clone)]
enum Pending {
    Built,
    /// Child slot from its parent: per perspective either a delta or a rebuild
    /// (rebuild reads `place` and the slot's key).
    Move {
        delta: [Option<Delta>; 2],
        place: Placement3,
    },
}

#[derive(Clone)]
struct Finny {
    acc: Vec<i16>,
    place: Option<Placement3>,
}

#[derive(Clone)]
pub(crate) struct V3State {
    pub(crate) model: Arc<QuantNnueV3>,
    h: usize,
    stack: Vec<i16>,
    pending: Vec<Pending>,
    /// Per slot: keys (one per perspective) and piece count, set at record time.
    slot_keys: Vec<[u8; 2]>,
    slot_pieces: Vec<u32>,
    top: usize,
    pub(crate) stm: Color,
    pub(crate) lazy: bool,
    finny: Vec<Finny>,
}

/// Undo token payload for a v3 push.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PushV3 {
    pub(crate) prev_stm: Color,
}

impl V3State {
    pub(crate) fn new(model: Arc<QuantNnueV3>) -> Self {
        let h = model.hidden_dim;
        Self {
            model,
            h,
            stack: vec![0; 64 * 2 * h],
            pending: vec![Pending::Built; 64],
            slot_keys: vec![[0; 2]; 64],
            slot_pieces: vec![0; 64],
            top: 0,
            stm: Color::White,
            lazy: false,
            finny: Vec::new(),
        }
    }

    #[inline]
    fn key(&self, place: &Placement3, persp: usize) -> u8 {
        let k = place.king(persp);
        let k = if persp == 1 { k ^ 56 } else { k };
        let mirror = (k & 7) >= 4;
        let k = if mirror { k ^ 7 } else { k };
        let bucket = self.model.king_bucket_layout[(k >> 3) * 4 + (k & 7)];
        bucket * 2 + mirror as u8
    }

    #[inline]
    fn feature(key: u8, persp: usize, color: usize, piece: usize, sq: usize) -> u32 {
        let bucket = (key >> 1) as usize;
        let mut s = if persp == 1 { sq ^ 56 } else { sq };
        if key & 1 == 1 {
            s ^= 7;
        }
        let rel = if persp == 0 { color } else { 1 - color };
        (bucket * 768 + (rel * 6 + piece) * 64 + s) as u32
    }

    #[inline]
    fn ensure_slot(&mut self, t: usize) {
        if self.pending.len() <= t {
            let n = (t + 1).max(self.pending.len() * 2);
            self.stack.resize(n * 2 * self.h, 0);
            self.pending.resize(n, Pending::Built);
            self.slot_keys.resize(n, [0; 2]);
            self.slot_pieces.resize(n, 0);
        }
    }

    pub(crate) fn refresh(&mut self, board: &Board) {
        let place = Placement3::of(board);
        self.top = 0;
        self.ensure_slot(0);
        self.slot_keys[0] = [self.key(&place, 0), self.key(&place, 1)];
        self.slot_pieces[0] = place.count();
        self.stm = board.side_to_move();
        for p in 0..2 {
            self.rebuild_into(0, p, &place);
        }
        self.pending[0] = Pending::Built;
    }

    pub(crate) fn apply_move(&mut self, before: &Board, after: &Board) -> PushV3 {
        let pb = Placement3::of(before);
        let pa = Placement3::of(after);
        let prev = PushV3 { prev_stm: self.stm };
        let t = self.top + 1;
        self.ensure_slot(t);
        let parent_keys = self.slot_keys[self.top];
        let keys = [self.key(&pa, 0), self.key(&pa, 1)];
        let mut delta = [None, None];
        for p in 0..2 {
            if keys[p] != parent_keys[p] {
                continue; // rebuild
            }
            let mut d = Delta::default();
            let mut ok = true;
            for c in 0..2 {
                for pc in 0..6 {
                    let (b, a) = (pb.pieces[c][pc], pa.pieces[c][pc]);
                    let mut gone = b & !a;
                    while gone != 0 {
                        let sq = gone.trailing_zeros() as usize;
                        gone &= gone - 1;
                        if (d.ns as usize) < MAX_DELTA {
                            d.sub[d.ns as usize] = Self::feature(keys[p], p, c, pc, sq);
                            d.ns += 1;
                        } else {
                            ok = false;
                        }
                    }
                    let mut came = a & !b;
                    while came != 0 {
                        let sq = came.trailing_zeros() as usize;
                        came &= came - 1;
                        if (d.na as usize) < MAX_DELTA {
                            d.add[d.na as usize] = Self::feature(keys[p], p, c, pc, sq);
                            d.na += 1;
                        } else {
                            ok = false;
                        }
                    }
                }
            }
            delta[p] = if ok { Some(d) } else { None };
        }
        self.slot_keys[t] = keys;
        self.slot_pieces[t] = pa.count();
        self.pending[t] = Pending::Move { delta, place: pa };
        self.top = t;
        self.stm = after.side_to_move();
        if !self.lazy {
            self.build(t);
        }
        prev
    }

    pub(crate) fn revert(&mut self, prev: PushV3) {
        debug_assert!(self.top > 0, "v3 accumulator stack underflow");
        self.top -= 1;
        self.stm = prev.prev_stm;
    }

    pub(crate) fn null_move(&mut self, after: &Board) -> Color {
        let prev = self.stm;
        self.stm = after.side_to_move();
        prev
    }

    pub(crate) fn ensure_top(&mut self) {
        if matches!(self.pending[self.top], Pending::Built) {
            return;
        }
        let mut c = self.top;
        while c > 0 && !matches!(self.pending[c], Pending::Built) {
            c -= 1;
        }
        for t in c + 1..=self.top {
            self.build(t);
        }
    }

    fn build(&mut self, t: usize) {
        let pending = std::mem::replace(&mut self.pending[t], Pending::Built);
        match pending {
            Pending::Built => {}
            Pending::Move { delta, place } => {
                for p in 0..2 {
                    match delta[p] {
                        Some(d) => self.delta_into(t, p, &d),
                        None => self.rebuild_into(t, p, &place),
                    }
                }
            }
        }
    }

    fn delta_into(&mut self, t: usize, p: usize, d: &Delta) {
        let h = self.h;
        let w1 = &self.model.w1;
        let row = |i: u32| &w1[i as usize * h..(i as usize + 1) * h];
        let (parents, child) = self.stack.split_at_mut(t * 2 * h);
        let src = &parents[(t - 1) * 2 * h + p * h..(t - 1) * 2 * h + (p + 1) * h];
        let dst = &mut child[p * h..(p + 1) * h];
        match (d.ns, d.na) {
            (1, 1) => push_delta_quiet_fused_plain(dst, src, row(d.sub[0]), row(d.add[0])),
            (2, 1) => push_delta_capture_fused(dst, src, row(d.sub[0]), row(d.sub[1]), row(d.add[0])),
            _ => {
                dst.copy_from_slice(src);
                for &i in &d.sub[..d.ns as usize] {
                    vec_sub_assign_i16(dst, row(i));
                }
                for &i in &d.add[..d.na as usize] {
                    vec_add_assign_i16(dst, row(i));
                }
            }
        }
    }

    /// Build perspective `p` of slot `t` from the refresh cache for its key.
    fn rebuild_into(&mut self, t: usize, p: usize, place: &Placement3) {
        let h = self.h;
        let key = self.slot_keys[t][p];
        let nkeys = self.model.n_king_buckets * 2;
        if self.finny.is_empty() {
            self.finny = vec![Finny { acc: Vec::new(), place: None }; 2 * nkeys];
        }
        let model = &*self.model;
        let entry = &mut self.finny[p * nkeys + key as usize];
        let old = match entry.place {
            Some(pl) => pl,
            None => {
                entry.acc.clear();
                entry.acc.extend_from_slice(&model.b1);
                Placement3 { pieces: [[0; 6]; 2] }
            }
        };
        for c in 0..2 {
            for pc in 0..6 {
                let (b, a) = (old.pieces[c][pc], place.pieces[c][pc]);
                let mut gone = b & !a;
                while gone != 0 {
                    let sq = gone.trailing_zeros() as usize;
                    gone &= gone - 1;
                    let i = Self::feature(key, p, c, pc, sq) as usize;
                    vec_sub_assign_i16(&mut entry.acc, &model.w1[i * h..(i + 1) * h]);
                }
                let mut came = a & !b;
                while came != 0 {
                    let sq = came.trailing_zeros() as usize;
                    came &= came - 1;
                    let i = Self::feature(key, p, c, pc, sq) as usize;
                    vec_add_assign_i16(&mut entry.acc, &model.w1[i * h..(i + 1) * h]);
                }
            }
        }
        entry.place = Some(*place);
        let base = t * 2 * h;
        self.stack[base + p * h..base + (p + 1) * h].copy_from_slice(&entry.acc);
    }

    fn head(&self, white: &[i16], black: &[i16], stm: Color, pieces: u32) -> i32 {
        let m = &*self.model;
        let h = self.h;
        let bucket = (((pieces as i64 - 2) / 4).clamp(0, m.n_output_buckets as i64 - 1)) as usize;
        let w = &m.w2[bucket * 2 * h..(bucket + 1) * 2 * h];
        let (first, second) = if stm == Color::White { (white, black) } else { (black, white) };
        let sum = screlu_dot_pub(first, &w[..h], m.qa) + screlu_dot_pub(second, &w[h..], m.qa);
        let num = (sum + m.b2[bucket] as i64) * m.scale as i64;
        (num / (m.qa as i64 * m.qa as i64 * m.qb as i64)) as i32
    }

    /// White-POV centipawns of the live slot (which must be built).
    pub(crate) fn eval_from_acc(&self) -> i32 {
        let h = self.h;
        let base = self.top * 2 * h;
        let (w, b) = (&self.stack[base..base + h], &self.stack[base + h..base + 2 * h]);
        let v = self.head(w, b, self.stm, self.slot_pieces[self.top]);
        if self.stm == Color::White {
            v
        } else {
            -v
        }
    }

    /// Full recompute for a board (parity testing); white-POV centipawns.
    pub(crate) fn eval_full(&self, board: &Board) -> i32 {
        let place = Placement3::of(board);
        let h = self.h;
        let mut accs = [self.model.b1.clone(), self.model.b1.clone()];
        for p in 0..2 {
            let key = self.key(&place, p);
            for c in 0..2 {
                for pc in 0..6 {
                    let mut bb = place.pieces[c][pc];
                    while bb != 0 {
                        let sq = bb.trailing_zeros() as usize;
                        bb &= bb - 1;
                        let i = Self::feature(key, p, c, pc, sq) as usize;
                        vec_add_assign_i16(&mut accs[p], &self.model.w1[i * h..(i + 1) * h]);
                    }
                }
            }
        }
        let stm = board.side_to_move();
        let v = self.head(&accs[0], &accs[1], stm, place.count());
        if stm == Color::White {
            v
        } else {
            -v
        }
    }
}
