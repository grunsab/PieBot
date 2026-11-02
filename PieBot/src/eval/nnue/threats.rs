//! Threat input features (PIENNQ03 version 6).
//!
//! Reference: `training/nnue/threats.py`, which defines the features and
//! their index layout; this module must produce the same indices. In short, a
//! feature is (attacker, from, to, attacked) seen from one perspective, with
//! colours relative to it and squares oriented like the piece features (ranks
//! flipped for Black, files mirrored when its king is on files e-h). Kings
//! neither attack nor get attacked, pawns attack only knights and rooks,
//! bishops and rooks attack P/N/B/R, knights and queens P/N/B/R/Q, and a
//! same-type pair counts only for oriented from > to. The pair rules follow
//! Stockfish's FullThreats (59,808 features).
//!
//! Incremental updates work on "raw" threats (absolute colours and squares),
//! which don't depend on the perspective. After a move, a threat can only
//! change if its attacker or target square changed, or if it comes from a
//! slider whose attack set touches a changed square (otherwise every square
//! on its rays up to the first blocker is unchanged, so its attacks are too).
//! Those candidates are listed on both boards and diffed.

use crate::eval::nnue::network_v3::Placement3;
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves, BitBoard, Board,
    Color, Square,
};
use std::sync::OnceLock;

pub const N_THREAT_FEATURES: usize = 59808;
/// 30 non-king pieces, each attacking at most 8 pieces (one per ray or
/// knight jump; a pawn at most 2), so 240 is the most any position has.
pub const MAX_ACTIVE: usize = 256;

const KING: usize = 5;
/// Attacked types per attacker type, in slot order.
const TARGETS: [&[usize]; 6] = [&[1, 3], &[0, 1, 2, 3, 4], &[0, 1, 2, 3], &[0, 1, 2, 3], &[0, 1, 2, 3, 4], &[]];
const NONE: u32 = u32::MAX;

struct Tables {
    /// Offset of each from-square inside its attacker type's run.
    sq_base: [[u32; 64]; 6],
    /// Rank of `to` among the empty-board attacks of `from` (pawns: either direction).
    to_rank: Vec<u8>,
    /// `[rel attacker colour][attacker type][rel attacked colour][attacked type]`:
    /// block plus slot offset, or NONE when the pair doesn't exist.
    pair_base: [[[[u32; 6]; 2]; 6]; 2],
}

fn sq(i: usize) -> Square {
    Square::index(i)
}

fn empty_attacks(pt: usize, from: usize, up: bool) -> u64 {
    let s = sq(from);
    match pt {
        0 => {
            let r = from >> 3;
            if (1..=6).contains(&r) {
                get_pawn_attacks(s, if up { Color::White } else { Color::Black }).0
            } else {
                0
            }
        }
        1 => get_knight_moves(s).0,
        2 => get_bishop_moves(s, BitBoard::EMPTY).0,
        3 => get_rook_moves(s, BitBoard::EMPTY).0,
        4 => (get_bishop_moves(s, BitBoard::EMPTY) | get_rook_moves(s, BitBoard::EMPTY)).0,
        _ => get_king_moves(s).0,
    }
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut sq_base = [[0u32; 64]; 6];
        let mut to_rank = vec![0u8; 6 * 64 * 64];
        let mut per_type = [0u32; 6];
        for pt in 0..6 {
            let mut off = 0u32;
            for from in 0..64 {
                sq_base[pt][from] = off;
                for up in [true, false] {
                    let m = empty_attacks(pt, from, up);
                    let mut bb = m;
                    while bb != 0 {
                        let to = bb.trailing_zeros() as usize;
                        bb &= bb - 1;
                        to_rank[(pt * 64 + from) * 64 + to] = (m & ((1u64 << to) - 1)).count_ones() as u8;
                    }
                }
                off += empty_attacks(pt, from, true).count_ones();
            }
            per_type[pt] = off;
        }
        let mut pair_base = [[[[NONE; 6]; 2]; 6]; 2];
        let mut block = 0u32;
        for ca in 0..2 {
            for pa in 0..6 {
                let n = TARGETS[pa].len() as u32;
                for cb in 0..2 {
                    for (slot, &pb) in TARGETS[pa].iter().enumerate() {
                        pair_base[ca][pa][cb][pb] = block + (cb as u32 * n + slot as u32) * per_type[pa];
                    }
                }
                block += 2 * n * per_type[pa];
            }
        }
        assert_eq!(block as usize, N_THREAT_FEATURES);
        Tables { sq_base, to_rank, pair_base }
    })
}

/// Target bitboard per attacker type: the attacked types it may have.
#[inline]
fn target_mask(place: &Placement3, pt: usize) -> u64 {
    let p = &place.pieces;
    let of = |t: usize| p[0][t] | p[1][t];
    match pt {
        0 => of(1) | of(3),
        2 | 3 => of(0) | of(1) | of(2) | of(3),
        1 | 4 => of(0) | of(1) | of(2) | of(3) | of(4),
        _ => 0,
    }
}

#[inline]
fn attacks(pt: usize, color: usize, from: usize, occ: u64) -> u64 {
    let s = sq(from);
    let o = BitBoard(occ);
    match pt {
        0 => get_pawn_attacks(s, if color == 0 { Color::White } else { Color::Black }).0,
        1 => get_knight_moves(s).0,
        2 => get_bishop_moves(s, o).0,
        3 => get_rook_moves(s, o).0,
        4 => (get_bishop_moves(s, o) | get_rook_moves(s, o)).0,
        _ => get_king_moves(s).0,
    }
}

/// A fixed-capacity list on the stack: no allocation, and no zeroing (the
/// buffer is left uninitialised past `n`).
pub(crate) struct List {
    buf: [std::mem::MaybeUninit<u32>; MAX_ACTIVE],
    n: usize,
}

impl List {
    #[inline]
    pub(crate) fn new() -> Self {
        Self { buf: [std::mem::MaybeUninit::uninit(); MAX_ACTIVE], n: 0 }
    }

    #[inline]
    fn push(&mut self, v: u32) {
        // No position has more than 240 threats, so this never drops one.
        if self.n < MAX_ACTIVE {
            self.buf[self.n].write(v);
            self.n += 1;
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[u32] {
        // SAFETY: the first `n` entries were written by `push`.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u32, self.n) }
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u32] {
        // SAFETY: as in `as_slice`.
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut u32, self.n) }
    }
}

/// A placement with a square -> piece lookup, built once per threat pass.
struct View<'a> {
    place: &'a Placement3,
    occ: u64,
    /// Piece code (colour * 6 + type) per square; unspecified on empty squares.
    mb: [u8; 64],
}

impl<'a> View<'a> {
    #[inline]
    fn new(place: &'a Placement3) -> Self {
        let mut mb = [0u8; 64];
        let mut occ = 0;
        for c in 0..2 {
            for t in 0..6 {
                let mut bb = place.pieces[c][t];
                occ |= bb;
                while bb != 0 {
                    mb[bb.trailing_zeros() as usize] = (c * 6 + t) as u8;
                    bb &= bb - 1;
                }
            }
        }
        Self { place, occ, mb }
    }

    /// Piece code on an occupied square.
    #[inline]
    fn at(&self, s: usize) -> u8 {
        self.mb[s]
    }
}

/// Raw threat: from | to << 6 | attacker code << 12 | attacked code << 16.
#[inline]
fn raw(from: usize, to: usize, a: usize, b: u8) -> u32 {
    (from | to << 6 | a << 12) as u32 | (b as u32) << 16
}

/// Every threat from the piece on `from` (code `a`), appended to `out`.
#[inline]
fn push_from(v: &View, from: usize, a: usize, out: &mut List) {
    let (c, pt) = (a / 6, a % 6);
    let mut hit = attacks(pt, c, from, v.occ) & target_mask(v.place, pt);
    while hit != 0 {
        let to = hit.trailing_zeros() as usize;
        hit &= hit - 1;
        out.push(raw(from, to, a, v.at(to)));
    }
}

/// All raw threats of `place`.
fn raw_threats(place: &Placement3, out: &mut List) {
    let v = View::new(place);
    for c in 0..2 {
        for pt in 0..KING {
            let mut bb = place.pieces[c][pt];
            while bb != 0 {
                let from = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                push_from(&v, from, c * 6 + pt, out);
            }
        }
    }
}

/// The raw threats of `place` that can differ after the squares in `changed`
/// changed: every threat from a changed square or from a slider that sees a
/// changed square, and every pawn or knight threat onto a changed square.
/// (A slider threat onto a changed square comes from a slider that sees it.)
fn raw_threats_near(place: &Placement3, changed: u64, out: &mut List) {
    let v = View::new(place);
    let occ = v.occ;
    let p = &place.pieces;
    let of = |t: usize| p[0][t] | p[1][t];
    let (diag, line) = (of(2) | of(4), of(3) | of(4));
    let mut full = changed & occ & !of(KING);
    let mut cs = changed;
    while cs != 0 {
        let s = cs.trailing_zeros() as usize;
        cs &= cs - 1;
        let sqr = sq(s);
        full |= (get_bishop_moves(sqr, BitBoard(occ)).0 & diag) | (get_rook_moves(sqr, BitBoard(occ)).0 & line);
        if occ & (1u64 << s) == 0 {
            continue;
        }
        let b = v.at(s);
        let bt = (b % 6) as usize;
        if bt == KING {
            continue;
        }
        // Knights attack P/N/B/R/Q; pawns only N and R.
        let mut kn = get_knight_moves(sqr).0 & of(1) & !changed;
        while kn != 0 {
            let f = kn.trailing_zeros() as usize;
            kn &= kn - 1;
            out.push(raw(f, s, v.at(f) as usize, b));
        }
        if bt == 1 || bt == 3 {
            // A white pawn attacks s from the squares a black pawn on s would attack.
            for (c, from_c) in [(0usize, Color::Black), (1usize, Color::White)] {
                let mut pw = get_pawn_attacks(sqr, from_c).0 & p[c][0] & !changed;
                while pw != 0 {
                    let f = pw.trailing_zeros() as usize;
                    pw &= pw - 1;
                    out.push(raw(f, s, c * 6, b));
                }
            }
        }
    }
    while full != 0 {
        let f = full.trailing_zeros() as usize;
        full &= full - 1;
        push_from(&v, f, v.at(f) as usize, out);
    }
}

/// Feature index of a raw threat for a perspective, or None when the
/// same-type rule drops it.
#[inline]
fn feature_of(t: &Tables, r: u32, persp: usize, mirror: bool) -> Option<u32> {
    let flip = (persp * 56) ^ if mirror { 7 } else { 0 };
    let from = (r & 63) as usize ^ flip;
    let to = ((r >> 6) & 63) as usize ^ flip;
    let a = ((r >> 12) & 15) as usize;
    let b = ((r >> 16) & 15) as usize;
    let (ca, pa) = (a / 6 ^ persp, a % 6);
    let (cb, pb) = (b / 6 ^ persp, b % 6);
    if pa == pb && from < to {
        return None;
    }
    let base = t.pair_base[ca][pa][cb][pb];
    debug_assert_ne!(base, NONE);
    Some(base + t.sq_base[pa][from] + t.to_rank[(pa * 64 + from) * 64 + to] as u32)
}

/// Whether a perspective's squares are mirrored (its oriented king on files e-h).
#[inline]
pub(crate) fn mirror_of(place: &Placement3, persp: usize) -> bool {
    let k = place.pieces[persp][KING].trailing_zeros() as usize;
    let k = if persp == 1 { k ^ 56 } else { k };
    (k & 7) >= 4
}

/// Active features of a placement for one perspective (unsorted).
pub(crate) fn active(place: &Placement3, persp: usize, out: &mut List) {
    let t = tables();
    let mirror = mirror_of(place, persp);
    let mut raws = List::new();
    raw_threats(place, &mut raws);
    for &r in raws.as_slice() {
        if let Some(f) = feature_of(t, r, persp, mirror) {
            out.push(f);
        }
    }
}

/// Raw threats removed and added between two placements.
pub(crate) struct RawDiff {
    pub(crate) removed: List,
    pub(crate) added: List,
}

impl RawDiff {
    #[inline]
    pub(crate) fn new() -> Self {
        Self { removed: List::new(), added: List::new() }
    }
}

#[cfg(test)]
fn raw_diff(before: &Placement3, after: &Placement3) -> RawDiff {
    let mut d = RawDiff::new();
    raw_diff_into(before, after, &mut d);
    d
}

/// Fill `d` (empty) with the raw threats removed and added between two
/// placements. It writes in place because the lists are 1 KB each.
pub(crate) fn raw_diff_into(before: &Placement3, after: &Placement3, d: &mut RawDiff) {
    let mut changed = 0u64;
    for c in 0..2 {
        for pt in 0..6 {
            changed |= before.pieces[c][pt] ^ after.pieces[c][pt];
        }
    }
    let (mut old, mut new) = (List::new(), List::new());
    raw_threats_near(before, changed, &mut old);
    raw_threats_near(after, changed, &mut new);
    old.as_mut_slice().sort_unstable();
    new.as_mut_slice().sort_unstable();
    let (o, n) = (old.as_slice(), new.as_slice());
    let (mut i, mut j) = (0, 0);
    while i < o.len() || j < n.len() {
        if j == n.len() || (i < o.len() && o[i] < n[j]) {
            d.removed.push(o[i]);
            i += 1;
        } else if i == o.len() || n[j] < o[i] {
            d.added.push(n[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
}

/// Map a raw diff to one perspective's feature changes.
pub(crate) fn diff_features(d: &RawDiff, persp: usize, mirror: bool, added: &mut List, removed: &mut List) {
    let t = tables();
    for &r in d.added.as_slice() {
        if let Some(f) = feature_of(t, r, persp, mirror) {
            added.push(f);
        }
    }
    for &r in d.removed.as_slice() {
        if let Some(f) = feature_of(t, r, persp, mirror) {
            removed.push(f);
        }
    }
}

/// Sorted active threat features of `board` for perspective 0 (White) or 1 (Black).
pub fn features(board: &Board, persp: usize) -> Vec<u32> {
    let mut out = List::new();
    active(&Placement3::of(board), persp, &mut out);
    let mut v = out.as_slice().to_vec();
    v.sort_unstable();
    v
}

/// Sorted active features from piece bitboards `[colour][type]` (cozy-chess
/// order, colour 0 = White), for perspective 0 or 1. The trainer calls this
/// with its side to move as colour 0 on its stm-oriented board.
pub fn features_of_pieces(pieces: [[u64; 6]; 2], persp: usize) -> Vec<u32> {
    let mut out = List::new();
    active(&Placement3 { pieces }, persp, &mut out);
    let mut v = out.as_slice().to_vec();
    v.sort_unstable();
    v
}

/// Whether `persp`'s squares are mirrored on `board`.
pub fn mirrored(board: &Board, persp: usize) -> bool {
    mirror_of(&Placement3::of(board), persp)
}

/// Features added and removed by going from `before` to `after` (same mirror).
pub fn changed_features(before: &Board, after: &Board, persp: usize, added: &mut Vec<u32>, removed: &mut Vec<u32>) {
    let (pb, pa) = (Placement3::of(before), Placement3::of(after));
    debug_assert_eq!(mirror_of(&pb, persp), mirror_of(&pa, persp));
    let (mut a, mut r) = (List::new(), List::new());
    let mut d = RawDiff::new();
    raw_diff_into(&pb, &pa, &mut d);
    diff_features(&d, persp, mirror_of(&pa, persp), &mut a, &mut r);
    added.extend_from_slice(a.as_slice());
    removed.extend_from_slice(r.as_slice());
}

/// Ask for the first cache lines of each threat row a coming update reads.
/// (Prefetching whole rows measured 3% slower on the 13900KF.)
#[inline]
pub(crate) fn prefetch_rows(wt: &[i8], rows: &[u32], h: usize) {
    #[cfg(target_arch = "x86_64")]
    for &i in rows {
        // SAFETY: prefetch has no side effects and the address is inside `wt`.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let p = wt.as_ptr().add(i as usize * h) as *const i8;
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p.add(64), _MM_HINT_T0);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (wt, rows, h);
}

const CHUNK: usize = 64;

/// `dst = src - sum(w1 rows sub16) + sum(w1 rows add16) - sum(wt rows sub8) +
/// sum(wt rows add8)`, rows `h` wide, in one pass: each 64-lane chunk stays
/// in registers while every row is applied. Wrapping i16 arithmetic, like
/// the separate row updates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fused_update(
    dst: &mut [i16],
    src: &[i16],
    w1: &[i16],
    sub16: &[u32],
    add16: &[u32],
    wt: &[i8],
    sub8: &[u32],
    add8: &[u32],
    h: usize,
) {
    let mut c = 0;
    while c + CHUNK <= h {
        let mut acc = [0i16; CHUNK];
        acc.copy_from_slice(&src[c..c + CHUNK]);
        for &i in sub16 {
            let r = &w1[i as usize * h + c..i as usize * h + c + CHUNK];
            for k in 0..CHUNK {
                acc[k] = acc[k].wrapping_sub(r[k]);
            }
        }
        for &i in add16 {
            let r = &w1[i as usize * h + c..i as usize * h + c + CHUNK];
            for k in 0..CHUNK {
                acc[k] = acc[k].wrapping_add(r[k]);
            }
        }
        for &i in sub8 {
            let r = &wt[i as usize * h + c..i as usize * h + c + CHUNK];
            for k in 0..CHUNK {
                acc[k] = acc[k].wrapping_sub(r[k] as i16);
            }
        }
        for &i in add8 {
            let r = &wt[i as usize * h + c..i as usize * h + c + CHUNK];
            for k in 0..CHUNK {
                acc[k] = acc[k].wrapping_add(r[k] as i16);
            }
        }
        dst[c..c + CHUNK].copy_from_slice(&acc);
        c += CHUNK;
    }
    // Widths that aren't a multiple of the chunk: the rest lane by lane.
    for k in c..h {
        let mut v = src[k];
        for &i in sub16 {
            v = v.wrapping_sub(w1[i as usize * h + k]);
        }
        for &i in add16 {
            v = v.wrapping_add(w1[i as usize * h + k]);
        }
        for &i in sub8 {
            v = v.wrapping_sub(wt[i as usize * h + k] as i16);
        }
        for &i in add8 {
            v = v.wrapping_add(wt[i as usize * h + k] as i16);
        }
        dst[k] = v;
    }
}

#[cfg(test)]
mod bench {
    //! Per-move cost breakdown of the threat path at width 2048, on random games:
    //! cargo test --release --lib threats::bench -- --ignored --nocapture
    use super::*;
    use std::time::Instant;

    fn pairs() -> Vec<(Placement3, Placement3)> {
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut out = Vec::new();
        for _ in 0..200 {
            let mut b = Board::default();
            for _ in 0..120 {
                let mut mv = Vec::new();
                b.generate_moves(|m| {
                    mv.extend(m);
                    false
                });
                if mv.is_empty() {
                    break;
                }
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let m = mv[(s >> 33) as usize % mv.len()];
                let before = Placement3::of(&b);
                b.play_unchecked(m);
                out.push((before, Placement3::of(&b)));
            }
        }
        out
    }

    #[test]
    #[ignore]
    fn cost_breakdown() {
        let ps = pairs();
        let n = ps.len() as f64;
        let h = 2048;
        let wt: Vec<i8> = (0..N_THREAT_FEATURES * h).map(|i| (i % 7) as i8 - 3).collect();
        let w1: Vec<i16> = (0..768 * 16 * h).map(|i| (i % 5) as i16 - 2).collect();
        let src = vec![0i16; h];
        let mut dst = vec![0i16; h];
        let reps = 5;
        let mut sink = 0usize;

        let now = Instant::now();
        for _ in 0..reps {
            for (a, b) in &ps {
                let d = raw_diff(a, b);
                sink += d.added.n + d.removed.n;
            }
        }
        let t_diff = now.elapsed().as_nanos() as f64 / (n * reps as f64);

        let mut cand = 0usize;
        let now = Instant::now();
        for (a, b) in &ps {
            let mut changed = 0u64;
            for c in 0..2 {
                for pt in 0..6 {
                    changed |= a.pieces[c][pt] ^ b.pieces[c][pt];
                }
            }
            let mut o = List::new();
            raw_threats_near(a, changed, &mut o);
            cand += o.n;
        }
        println!("candidates per board {:.1}, near() {:.0} ns", cand as f64 / n, now.elapsed().as_nanos() as f64 / n);
        let now = Instant::now();
        for (a, _) in &ps {
            let v = View::new(a);
            sink += v.mb[3] as usize;
        }
        println!("View::new {:.0} ns", now.elapsed().as_nanos() as f64 / n);
        let diffs: Vec<RawDiff> = ps.iter().map(|(a, b)| raw_diff(a, b)).collect();
        let now = Instant::now();
        for _ in 0..reps {
            for d in &diffs {
                for p in 0..2 {
                    let (mut ad, mut rm) = (List::new(), List::new());
                    diff_features(d, p, false, &mut ad, &mut rm);
                    sink += ad.n + rm.n;
                }
            }
        }
        let t_map = now.elapsed().as_nanos() as f64 / (n * reps as f64);
        let avg: f64 = diffs.iter().map(|d| (d.added.n + d.removed.n) as f64).sum::<f64>() / n;

        let feats: Vec<(List, List)> = diffs
            .iter()
            .map(|d| {
                let (mut ad, mut rm) = (List::new(), List::new());
                diff_features(d, 0, false, &mut ad, &mut rm);
                (ad, rm)
            })
            .collect();
        let pf = [5u32 * 768 + 100, 5 * 768 + 300];
        let now = Instant::now();
        for _ in 0..reps {
            for _ in &feats {
                fused_update(&mut dst, &src, &w1, &pf[..1], &pf[1..], &wt, &[], &[], h);
                sink += dst[7] as usize;
            }
        }
        let t_piece = now.elapsed().as_nanos() as f64 / (n * reps as f64);
        let now = Instant::now();
        for _ in 0..reps {
            for (ad, rm) in &feats {
                prefetch_rows(&wt, rm.as_slice(), h);
                prefetch_rows(&wt, ad.as_slice(), h);
                fused_update(&mut dst, &src, &w1, &pf[..1], &pf[1..], &wt, rm.as_slice(), ad.as_slice(), h);
                sink += dst[7] as usize;
            }
        }
        let t_fused = now.elapsed().as_nanos() as f64 / (n * reps as f64);
        let hot: Vec<(Vec<u32>, Vec<u32>)> = feats
            .iter()
            .map(|(a, r)| (a.as_slice().iter().map(|f| f % 16).collect(), r.as_slice().iter().map(|f| f % 16 + 16).collect()))
            .collect();
        let now = Instant::now();
        for _ in 0..reps {
            for (ad, rm) in &hot {
                fused_update(&mut dst, &src, &w1, &pf[..1], &pf[1..], &wt, rm, ad, h);
                sink += dst[7] as usize;
            }
        }
        let t_hot = now.elapsed().as_nanos() as f64 / (n * reps as f64);
        println!("piece+threat update, rows cache-hot {t_hot:.0} ns");

        let now = Instant::now();
        for _ in 0..reps {
            for (_, b) in &ps {
                let mut f = List::new();
                active(b, 0, &mut f);
                sink += f.n;
            }
        }
        let t_active = now.elapsed().as_nanos() as f64 / (n * reps as f64);
        println!(
            "moves {n}; raw changes/move {avg:.1}\n raw_diff {t_diff:.0} ns\n map both persp {t_map:.0} ns\n \
             piece-only update (1 persp, h={h}) {t_piece:.0} ns\n piece+threat update (1 persp) {t_fused:.0} ns\n \
             full active() {t_active:.0} ns  (sink {sink})"
        );
    }
}
