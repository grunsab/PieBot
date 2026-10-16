//! Experimental search track: a rewrite of the search core.
//!
//! Structure follows the modern alpha-beta engines: a lockless packed
//! transposition table that also stores the static evaluation, a staged move
//! picker (TT move, good captures, killers, counter move, quiets by history,
//! bad captures), a per-ply search stack, butterfly / continuation / capture /
//! pawn histories, static-evaluation correction histories, singular
//! extensions, and soft/hard time management. The public API is the same as
//! the baseline `alphabeta::Searcher`, so the UCI front end, the A/B tools
//! and the acceptance suites drive either one.

use crate::eval::nnue::loader::QuantNnue;
use crate::eval::nnue::network::{ChangeSet, QuantNetwork};
use crate::search::eval::{eval_cp, material_eval_cp, DRAW_SCORE, MATE_SCORE};
use crate::search::tt::Bound;
use cozy_chess::{BitBoard, Board, Color, Move, Piece, Square};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const MAX_PLY: usize = 128;
pub static STATS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];
#[inline(always)]
fn stat(_i: usize) {
    #[cfg(feature = "search-stats")]
    STATS[_i].fetch_add(1, Ordering::Relaxed);
}
pub fn print_stats() {
    let names = ["search", "qsearch", "raw_eval", "nnue_apply", "see", "scored_moves", "tt_probe", "tt_store", "gen_all", "gen_tact", "correction", "quiet_score"];
    let n0 = STATS[0].load(Ordering::Relaxed) + STATS[1].load(Ordering::Relaxed);
    for (i, n) in names.iter().enumerate() { eprintln!("{n}: {} ({:.2}/node)", STATS[i].load(Ordering::Relaxed), STATS[i].load(Ordering::Relaxed) as f64 / n0.max(1) as f64); }
}
const INF: i32 = MATE_SCORE + 1;
const MATE_IN_MAX: i32 = MATE_SCORE - 2 * MAX_PLY as i32;
/// Tablebase win score (minus ply): above any evaluation, below mates.
const TB_WIN: i32 = 20_000;
const FIFTY_MOVE_CLAIM_PLIES: u32 = 100;
const STACK_OFFSET: usize = 8;

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

macro_rules! tunables {
    ($($name:ident = $def:expr, $min:expr, $max:expr;)*) => {
        /// Search constants exposed for tuning (UCI options in the
        /// experimental build). Process-wide; read with `tp`.
        #[allow(non_camel_case_types, dead_code, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy, Debug)]
        #[repr(usize)]
        pub enum T { $($name),* }
        pub const TUNABLE_NAMES: &[&str] = &[$(stringify!($name)),*];
        pub const TUNABLE_DEFAULTS: &[i32] = &[$($def),*];
        pub const TUNABLE_RANGES: &[(i32, i32)] = &[$(($min, $max)),*];
        pub static TUNABLES: [AtomicI32; TUNABLE_NAMES.len()] = [$(AtomicI32::new($def)),*];
    };
}

tunables! {
    BonusMul = 170, 50, 400;
    BonusOff = 90, 0, 300;
    BonusMax = 1700, 500, 4000;
    MalusMul = 200, 50, 400;
    MalusOff = 80, 0, 300;
    MalusMax = 1400, 500, 4000;
    LmrBase = 820, 0, 2000;
    LmrDiv = 225, 150, 400;
    RazorBase = 450, 100, 1000;
    RazorMul = 280, 50, 600;
    RfpDepth = 10, 4, 16;
    RfpMul = 90, 30, 200;
    RfpCut = 20, 0, 60;
    RfpWorsen = 15, 0, 60;
    RfpImproving = 30, 0, 80;
    NmpBase = 4, 2, 7;
    NmpDepthDiv = 3, 2, 6;
    NmpEvalDiv = 200, 80, 400;
    NmpStaticMul = 20, 0, 50;
    NmpStaticBase = 180, 0, 400;
    IirDepth = 5, 3, 10;
    ProbcutMargin = 210, 80, 400;
    ProbcutImproving = 60, 0, 150;
    ProbcutDepth = 5, 3, 9;
    LmpBase = 3, 1, 8;
    CapFutBase = 250, 50, 500;
    CapFutMul = 230, 50, 400;
    CapSeeMul = 180, 50, 300;
    HistPruneMul = 3500, 1000, 8000;
    HistLmrDiv = 6000, 2000, 15000;
    FutBase = 90, 20, 250;
    FutMul = 110, 40, 250;
    QuietSeeMul = 25, 5, 60;
    SeDepth = 6, 4, 10;
    SeMargin = 60, 20, 150;
    SeTtPvMargin = 70, 0, 150;
    SeDouble = 20, 0, 60;
    LmrTtPv = 1024, 0, 2500;
    LmrCut = 1800, 0, 3500;
    LmrTtCapture = 1024, 0, 2500;
    LmrNotImproving = 600, 0, 2000;
    LmrCheck = 800, 0, 2000;
    LmrCutoffCnt = 900, 0, 2500;
    LmrTtMove = 1800, 0, 3500;
    LmrCapture = 600, 0, 2000;
    LmrStatDiv = 10000, 3000, 25000;
    DeeperBase = 50, 10, 120;
    DeeperMul = 2, 0, 8;
    Shallower = 9, 0, 30;
    QsFutility = 200, 50, 400;
    QsSee = 80, 0, 200;
    AspDelta = 20, 5, 50;
    AspDiv = 16, 6, 64;
    CorrPawn = 2, 0, 4;
    TmSoft = 45, 20, 80;
}

/// Draw bias in centipawns for the side that started the search: a drawn
/// line scores +CONTEMPT for it (and -CONTEMPT for the opponent). Positive
/// values make the engine seek draws, which is right against a much stronger
/// opponent. Not a tunable: self-play cannot measure it.
pub static CONTEMPT: AtomicI32 = AtomicI32::new(0);

#[inline(always)]
fn tp(t: T) -> i32 {
    TUNABLES[t as usize].load(Ordering::Relaxed)
}

/// Set a tunable by its name (case-insensitive). Returns false if unknown or
/// the value is outside its range.
pub fn set_tunable(name: &str, value: i32) -> bool {
    for (i, n) in TUNABLE_NAMES.iter().enumerate() {
        if n.eq_ignore_ascii_case(name) {
            let (lo, hi) = TUNABLE_RANGES[i];
            if value < lo || value > hi {
                return false;
            }
            TUNABLES[i].store(value, Ordering::Relaxed);
            return true;
        }
    }
    false
}

const HIST_MAX: i32 = 16_384;
const CORR_GRAIN: i32 = 256;
const CORR_MAX: i32 = 64 * CORR_GRAIN;
const CORR_SIZE: usize = 16_384;
const PAWN_HIST_SIZE: usize = 1_024;

#[inline]
fn stat_bonus(depth: i32) -> i32 {
    (tp(T::BonusMul) * depth - tp(T::BonusOff)).clamp(0, tp(T::BonusMax))
}

#[inline]
fn stat_malus(depth: i32) -> i32 {
    (tp(T::MalusMul) * depth - tp(T::MalusOff)).clamp(0, tp(T::MalusMax))
}

/// Reductions in 1/1024 plies for the current tunables.
fn lmr_table() -> Box<[[i32; 64]; 64]> {
    let base = tp(T::LmrBase) as f64 / 1024.0;
    let div = tp(T::LmrDiv) as f64 / 100.0;
    let mut table = Box::new([[0i32; 64]; 64]);
    for depth in 1..64 {
        for m in 1..64 {
            let r = base + (depth as f64).ln() * (m as f64).ln() / div;
            table[depth][m] = (r * 1024.0) as i32;
        }
    }
    table
}

pub static LMR_TABLE: std::sync::LazyLock<[[i32; 64]; 64]> = std::sync::LazyLock::new(|| *lmr_table());

/// Integer late-move reduction used by the tests and the shallow-pruning
/// estimate: the base table, plus one ply when not improving, minus one for
/// a strong history and plus one for a bad one.
pub fn lmr_reduction(depth: u32, move_idx: usize, history_score: i32) -> u32 {
    lmr_reduction_improving(depth, move_idx, history_score, true)
}

pub fn lmr_reduction_improving(
    depth: u32,
    move_idx: usize,
    history_score: i32,
    improving: bool,
) -> u32 {
    if depth < 3 || move_idx < 3 {
        return 0;
    }
    let mut r = LMR_TABLE[(depth as usize).min(63)][move_idx.min(63)] / 1024;
    if !improving {
        r += 1;
    }
    if history_score > 2000 {
        r -= 1;
    } else if history_score < -2000 && r >= 1 {
        r += 1;
    }
    r.max(0) as u32
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#[inline]
fn piece_value(p: Piece) -> i32 {
    match p {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 950,
        Piece::King => 0,
    }
}

#[inline]
fn see_value(p: Piece) -> i32 {
    match p {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 950,
        Piece::King => 20_000,
    }
}

#[inline]
fn mated_in(ply: i32) -> i32 {
    -MATE_SCORE + ply
}

#[inline]
fn is_decisive(v: i32) -> bool {
    v.abs() >= MATE_IN_MAX
}

#[inline]
fn score_to_tt(score: i32, ply: i32) -> i32 {
    if score >= MATE_IN_MAX {
        score + ply
    } else if score <= -MATE_IN_MAX {
        score - ply
    } else {
        score
    }
}

#[inline]
fn score_from_tt(score: i32, ply: i32, halfmove: u8) -> i32 {
    if score >= MATE_IN_MAX {
        // A stored mate may be out of reach under the fifty-move rule.
        if MATE_SCORE - score > 99 - halfmove as i32 {
            return MATE_IN_MAX - 1;
        }
        score - ply
    } else if score <= -MATE_IN_MAX {
        if MATE_SCORE + score > 99 - halfmove as i32 {
            return -MATE_IN_MAX + 1;
        }
        score + ply
    } else {
        score
    }
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

#[inline]
fn pawn_key(board: &Board) -> u64 {
    let w = board.colored_pieces(Color::White, Piece::Pawn).0;
    let b = board.colored_pieces(Color::Black, Piece::Pawn).0;
    mix64(w ^ mix64(b ^ 0x9e37_79b9_7f4a_7c15))
}

#[inline]
fn non_pawn_key(board: &Board, c: Color) -> u64 {
    let mut k = 0x1234_5678_9abc_def1u64 ^ c as u64;
    for p in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
        k = mix64(k ^ board.colored_pieces(c, p).0);
    }
    k
}

#[inline]
fn minor_key(board: &Board) -> u64 {
    let mut k = 0x0bad_cafe_dead_beefu64;
    for c in [Color::White, Color::Black] {
        for p in [Piece::Knight, Piece::Bishop, Piece::King] {
            k = mix64(k ^ board.colored_pieces(c, p).0);
        }
    }
    k
}

#[inline]
fn pc_index(color: Color, piece: Piece) -> usize {
    color as usize * 6 + piece as usize
}

#[inline]
fn non_pawn_material(board: &Board, c: Color) -> bool {
    let ours = board.colors(c);
    !((board.pieces(Piece::Knight)
        | board.pieces(Piece::Bishop)
        | board.pieces(Piece::Rook)
        | board.pieces(Piece::Queen))
        & ours)
        .is_empty()
}

#[inline]
fn is_capture(board: &Board, m: Move) -> bool {
    let stm = board.side_to_move();
    board.color_on(m.to) == Some(!stm)
        || (board.piece_on(m.from) == Some(Piece::Pawn) && m.from.file() != m.to.file())
}

#[inline]
fn captured_piece(board: &Board, m: Move) -> Option<Piece> {
    let stm = board.side_to_move();
    if board.color_on(m.to) == Some(!stm) {
        board.piece_on(m.to)
    } else if board.piece_on(m.from) == Some(Piece::Pawn) && m.from.file() != m.to.file() {
        Some(Piece::Pawn)
    } else {
        None
    }
}

/// Squares from which each piece type of the side to move would give a
/// direct check to the enemy king.
fn check_squares(board: &Board) -> [BitBoard; 6] {
    let stm = board.side_to_move();
    let ksq = board.king(!stm);
    let occ = board.occupied();
    let bishop = cozy_chess::get_bishop_moves(ksq, occ);
    let rook = cozy_chess::get_rook_moves(ksq, occ);
    [
        cozy_chess::get_pawn_attacks(ksq, !stm),
        cozy_chess::get_knight_moves(ksq),
        bishop,
        rook,
        bishop | rook,
        BitBoard::EMPTY,
    ]
}

fn has_legal_move(board: &Board) -> bool {
    board.generate_moves(|_| true)
}

#[inline]
fn is_tactical(board: &Board, m: Move) -> bool {
    is_capture(board, m) || m.promotion == Some(Piece::Queen)
}

#[inline]
fn move_to_u16(m: Move) -> u16 {
    let promo = match m.promotion {
        None => 0,
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        _ => 0,
    };
    (m.from as u16) | ((m.to as u16) << 6) | (promo << 12)
}

#[inline]
fn move_from_u16(v: u16) -> Option<Move> {
    if v == 0 {
        return None;
    }
    let promotion = match (v >> 12) & 7 {
        1 => Some(Piece::Knight),
        2 => Some(Piece::Bishop),
        3 => Some(Piece::Rook),
        4 => Some(Piece::Queen),
        _ => None,
    };
    Some(Move {
        from: Square::index((v & 63) as usize),
        to: Square::index(((v >> 6) & 63) as usize),
        promotion,
    })
}

/// Static exchange evaluation: does `m` win at least `threshold` centipawns
/// once every recapture on the destination square has been played out?
pub fn see_ge(board: &Board, m: Move, threshold: i32) -> bool {
    stat(4);
    let from = m.from;
    let to = m.to;
    let stm = board.side_to_move();
    // Castling (king onto own rook) and other non-captures of own pieces.
    if board.color_on(to) == Some(stm) {
        return 0 >= threshold;
    }
    let mover = match board.piece_on(from) {
        Some(p) => p,
        None => return false,
    };
    let mut swap = captured_piece(board, m).map(see_value).unwrap_or(0) - threshold;
    if let Some(promo) = m.promotion {
        swap += see_value(promo) - see_value(Piece::Pawn);
    }
    if swap < 0 {
        return false;
    }
    let first_victim = m.promotion.unwrap_or(mover);
    swap = see_value(first_victim) - swap;
    if swap <= 0 {
        return true;
    }
    let mut occ = board.occupied() ^ from.bitboard() | to.bitboard();
    if mover == Piece::Pawn && m.from.file() != m.to.file() && board.piece_on(to).is_none() {
        // En passant: remove the captured pawn.
        let cap_sq = Square::new(to.file(), from.rank());
        occ ^= cap_sq.bitboard();
    }
    let bishops = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    let rooks = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    let attackers_to = |occ: BitBoard| -> BitBoard {
        (cozy_chess::get_pawn_attacks(to, Color::White) & board.colored_pieces(Color::Black, Piece::Pawn))
            | (cozy_chess::get_pawn_attacks(to, Color::Black)
                & board.colored_pieces(Color::White, Piece::Pawn))
            | (cozy_chess::get_knight_moves(to) & board.pieces(Piece::Knight))
            | (cozy_chess::get_bishop_moves(to, occ) & bishops)
            | (cozy_chess::get_rook_moves(to, occ) & rooks)
            | (cozy_chess::get_king_moves(to) & board.pieces(Piece::King))
    };
    let mut attackers = attackers_to(occ) & occ;
    let mut side = !stm;
    let mut res = 1i32;
    loop {
        let my_attackers = attackers & board.colors(side);
        if my_attackers.is_empty() {
            break;
        }
        res ^= 1;
        let mut found = None;
        for p in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let bb = my_attackers & board.pieces(p);
            if !bb.is_empty() {
                found = Some((p, bb.next_square().unwrap()));
                break;
            }
        }
        let (p, sq) = found.unwrap();
        if p == Piece::King {
            // The king may only capture last: if the other side still has
            // attackers, this capture is illegal and the result flips back.
            if !(attackers & board.colors(!side)).is_empty() {
                res ^= 1;
            }
            break;
        }
        swap = see_value(p) - swap;
        if swap < res {
            break;
        }
        occ ^= sq.bitboard();
        if matches!(p, Piece::Pawn | Piece::Bishop | Piece::Queen) {
            attackers |= cozy_chess::get_bishop_moves(to, occ) & bishops;
        }
        if matches!(p, Piece::Rook | Piece::Queen) {
            attackers |= cozy_chess::get_rook_moves(to, occ) & rooks;
        }
        attackers &= occ;
        side = !side;
    }
    res != 0
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct TtData {
    mv: Option<Move>,
    score: i32,
    eval: i32,
    depth: i32,
    /// None marks an entry that only caches the static evaluation.
    bound: Option<Bound>,
    pv: bool,
}

#[repr(align(64))]
#[derive(Default)]
struct TtBucket {
    slots: [(AtomicU64, AtomicU64); 4],
}

pub struct PackedTt {
    buckets: Vec<TtBucket>,
    gen: AtomicU64,
}

const TT_DEPTH_OFFSET: i32 = 2;

impl PackedTt {
    pub fn with_mb(mb: usize) -> Self {
        let n = ((mb.max(1) << 20) / std::mem::size_of::<TtBucket>()).max(1);
        let mut buckets = Vec::with_capacity(n);
        buckets.resize_with(n, TtBucket::default);
        Self {
            buckets,
            gen: AtomicU64::new(0),
        }
    }

    fn clear(&self) {
        for b in &self.buckets {
            for s in &b.slots {
                s.0.store(0, Ordering::Relaxed);
                s.1.store(0, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    fn bucket(&self, key: u64) -> &TtBucket {
        let idx = ((key as u128 * self.buckets.len() as u128) >> 64) as usize;
        &self.buckets[idx]
    }

    /// Start loading the bucket for `key` so a probe issued after the NNUE
    /// update finds it in cache.
    #[inline(always)]
    fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let p = self.bucket(key) as *const TtBucket as *const i8;
            std::arch::x86_64::_mm_prefetch(p, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    fn new_search(&self) {
        self.gen.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn gen5(&self) -> u64 {
        self.gen.load(Ordering::Relaxed) & 31
    }

    #[inline]
    fn unpack(data: u64) -> TtData {
        let bound = match (data >> 56) & 3 {
            1 => Some(Bound::Upper),
            2 => Some(Bound::Lower),
            3 => Some(Bound::Exact),
            _ => None,
        };
        TtData {
            mv: move_from_u16(data as u16),
            score: (data >> 16) as u16 as i16 as i32,
            eval: (data >> 32) as u16 as i16 as i32,
            depth: ((data >> 48) & 0xff) as i32 - TT_DEPTH_OFFSET,
            bound,
            pv: (data >> 58) & 1 != 0,
        }
    }

    fn probe(&self, key: u64) -> Option<TtData> {
        stat(6);
        let b = self.bucket(key);
        for s in &b.slots {
            let data = s.1.load(Ordering::Relaxed);
            if data != 0 && s.0.load(Ordering::Relaxed) ^ data == key {
                return Some(Self::unpack(data));
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn store(
        &self,
        key: u64,
        mv: Option<Move>,
        score: i32,
        eval: i32,
        depth: i32,
        bound: Option<Bound>,
        pv: bool,
    ) {
        stat(7);
        let b = self.bucket(key);
        let gen = self.gen5();
        let mut victim = 0usize;
        let mut victim_value = i32::MAX;
        for (i, s) in b.slots.iter().enumerate() {
            let data = s.1.load(Ordering::Relaxed);
            if data == 0 {
                victim = i;
                break;
            }
            if s.0.load(Ordering::Relaxed) ^ data == key {
                victim = i;
                break;
            }
            let d = ((data >> 48) & 0xff) as i32;
            let age = ((gen + 32 - ((data >> 59) & 31)) & 31) as i32;
            let value = d - 8 * age;
            if value < victim_value {
                victim_value = value;
                victim = i;
            }
        }
        let slot = &b.slots[victim];
        let old = slot.1.load(Ordering::Relaxed);
        let same = old != 0 && slot.0.load(Ordering::Relaxed) ^ old == key;
        let mut mv16 = mv.map(move_to_u16).unwrap_or(0);
        if same {
            let old_d = ((old >> 48) & 0xff) as i32 - TT_DEPTH_OFFSET;
            if mv16 == 0 {
                mv16 = old as u16;
            }
            // Keep a deeper entry of the same position unless this one is
            // exact or the old one is from an earlier search.
            let old_gen = (old >> 59) & 31;
            if bound != Some(Bound::Exact) && depth + 4 < old_d && old_gen == gen {
                return;
            }
        }
        let bound_bits: u64 = match bound {
            None => 0,
            Some(Bound::Upper) => 1,
            Some(Bound::Lower) => 2,
            Some(Bound::Exact) => 3,
        };
        let d = (depth + TT_DEPTH_OFFSET).clamp(1, 255) as u64;
        let data = mv16 as u64
            | ((score.clamp(-32_000, 32_000) as i16 as u16 as u64) << 16)
            | ((eval.clamp(-32_000, 32_000) as i16 as u16 as u64) << 32)
            | (d << 48)
            | (bound_bits << 56)
            | ((pv as u64) << 58)
            | (gen << 59);
        slot.0.store(key ^ data, Ordering::Relaxed);
        slot.1.store(data, Ordering::Relaxed);
    }

    fn hashfull(&self) -> usize {
        let n = self.buckets.len().min(250);
        let gen = self.gen5();
        let mut used = 0;
        for b in &self.buckets[..n] {
            for s in &b.slots {
                let d = s.1.load(Ordering::Relaxed);
                if d != 0 && (d >> 59) & 31 == gen {
                    used += 1;
                }
            }
        }
        used * 1000 / (n * 4).max(1)
    }
}

// ---------------------------------------------------------------------------
// Public parameter / result types (same shape as the baseline)
// ---------------------------------------------------------------------------

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
    /// `movetime` is a fixed per-move budget (UCI `go movetime`) to be used
    /// in full, rather than a clock allocation the search may finish early.
    pub fixed_movetime: bool,
}

#[derive(Default, Debug, Clone)]
pub struct SearchResult {
    pub bestmove: Option<String>,
    pub score_cp: i32,
    pub nodes: u64,
    pub depth: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum EvalMode {
    Material,
    Pst,
    Nnue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Abort;

type SResult = Result<i32, Abort>;

struct StopHelpersOnDrop(Arc<AtomicBool>);

impl Drop for StopHelpersOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Search stack and move list
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Frame {
    static_eval: i32,
    /// piece-square index (12 * 64) of the move made from this ply, or None
    /// for a null move / no move yet.
    cont: Option<usize>,
    mv: Option<Move>,
    excluded: Option<Move>,
    tt_pv: bool,
    in_check: bool,
    cutoff_cnt: i32,
    stat_score: i32,
    killers: [Option<Move>; 2],
    double_ext: i32,
    /// The move made from this ply was a capture or queen promotion.
    tactical: bool,
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            static_eval: 0,
            cont: None,
            mv: None,
            excluded: None,
            tt_pv: false,
            in_check: false,
            cutoff_cnt: 0,
            stat_score: 0,
            killers: [None, None],
            double_ext: 0,
            tactical: false,
        }
    }
}

const MAX_MOVES: usize = 256;

/// Fixed-capacity move list. Storage is left uninitialised: a node that cuts
/// off on its first move never pays to clear ~2 KB of stack.
struct MoveList {
    moves: [std::mem::MaybeUninit<Move>; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline]
    fn new() -> Self {
        Self {
            moves: [const { std::mem::MaybeUninit::uninit() }; MAX_MOVES],
            scores: [0; MAX_MOVES],
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, m: Move) {
        if self.len < MAX_MOVES {
            self.moves[self.len] = std::mem::MaybeUninit::new(m);
            self.len += 1;
        }
    }

    #[inline]
    fn get(&self, i: usize) -> Move {
        debug_assert!(i < self.len);
        // SAFETY: entries below `len` were written by `push`.
        unsafe { self.moves[i].assume_init() }
    }

    #[inline]
    fn contains(&self, m: Move) -> bool {
        (0..self.len).any(|i| self.get(i) == m)
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        self.moves.swap(a, b);
        self.scores.swap(a, b);
    }

    /// Move the best-scored entry in `from..to` to `from` and return it.
    #[inline]
    fn pick_range(&mut self, from: usize, to: usize) -> Option<(Move, i32)> {
        if from >= to {
            return None;
        }
        let mut best = from;
        for i in from + 1..to {
            if self.scores[i] > self.scores[best] {
                best = i;
            }
        }
        self.swap(from, best);
        Some((self.get(from), self.scores[from]))
    }

    #[inline]
    fn pick(&mut self, from: usize) -> Option<(Move, i32)> {
        let len = self.len;
        self.pick_range(from, len)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Tt,
    Generate,
    GoodCaptures,
    Killer1,
    Killer2,
    Counter,
    ScoreQuiets,
    Quiets,
    BadCaptures,
    Done,
}

/// Staged move generation for interior nodes: the TT move before anything
/// is generated, then winning captures, killers and the counter move, quiets
/// (scored only once reached), and losing captures last.
struct MovePicker {
    list: MoveList,
    stage: Stage,
    tt_move: Option<Move>,
    refutations: [Option<Move>; 3],
    cur: usize,
    n_caps: usize,
    bad_start: usize,
    quiet_cur: usize,
}

const STAGE_SCORE_QUIET_LIMIT: i32 = SCORE_COUNTER;

impl MovePicker {
    fn new(tt_move: Option<Move>, killers: [Option<Move>; 2], counter: Option<Move>) -> Self {
        Self {
            list: MoveList::new(),
            stage: Stage::Tt,
            tt_move,
            refutations: [killers[0], killers[1], counter],
            cur: 0,
            n_caps: 0,
            bad_start: 0,
            quiet_cur: 0,
        }
    }

    /// Next move and its ordering class (SCORE_* bands; quiets below
    /// `STAGE_SCORE_QUIET_LIMIT`). `skip_quiets` drops the remaining quiet
    /// stages but keeps the refutation moves.
    fn next(&mut self, s: &Searcher, board: &Board, sp: usize, skip_quiets: bool) -> Option<(Move, i32)> {
        loop {
            match self.stage {
                Stage::Tt => {
                    self.stage = Stage::Generate;
                    if let Some(m) = self.tt_move {
                        if board.is_legal(m) {
                            return Some((m, SCORE_TT));
                        }
                        self.tt_move = None;
                    }
                }
                Stage::Generate => {
                    gen_all(board, &mut self.list);
                    // Partition: tactical moves first, scored now; quiets after.
                    let mut n = 0;
                    let mut i = 0;
                    while i < self.list.len {
                        let m = self.list.get(i);
                        if Some(m) == self.tt_move {
                            self.list.len -= 1;
                            let last = self.list.len;
                            self.list.swap(i, last);
                            continue;
                        }
                        if is_tactical(board, m) {
                            let sc = s.capture_score(board, m);
                            self.list.scores[i] = if see_ge(board, m, -sc / 20) {
                                SCORE_GOOD_CAPTURE + sc
                            } else {
                                SCORE_BAD_CAPTURE + sc
                            };
                            self.list.swap(i, n);
                            n += 1;
                        }
                        i += 1;
                    }
                    self.n_caps = n;
                    self.cur = 0;
                    self.quiet_cur = n;
                    self.stage = Stage::GoodCaptures;
                }
                Stage::GoodCaptures => {
                    if let Some((m, sc)) = self.list.pick_range(self.cur, self.n_caps) {
                        if sc >= SCORE_GOOD_CAPTURE / 2 {
                            self.cur += 1;
                            return Some((m, sc));
                        }
                    }
                    self.bad_start = self.cur;
                    self.stage = Stage::Killer1;
                }
                Stage::Killer1 | Stage::Killer2 | Stage::Counter => {
                    let (k, band) = match self.stage {
                        Stage::Killer1 => (0, SCORE_KILLER1),
                        Stage::Killer2 => (1, SCORE_KILLER2),
                        _ => (2, SCORE_COUNTER),
                    };
                    self.stage = match self.stage {
                        Stage::Killer1 => Stage::Killer2,
                        Stage::Killer2 => Stage::Counter,
                        _ => Stage::ScoreQuiets,
                    };
                    if let Some(r) = self.refutations[k] {
                        if k == 2 && (self.refutations[0] == Some(r) || self.refutations[1] == Some(r)) {
                            continue;
                        }
                        if let Some(pos) = (self.quiet_cur..self.list.len).find(|&i| self.list.get(i) == r) {
                            self.list.swap(pos, self.quiet_cur);
                            self.quiet_cur += 1;
                            return Some((r, band));
                        }
                    }
                }
                Stage::ScoreQuiets => {
                    if skip_quiets {
                        self.stage = Stage::BadCaptures;
                        continue;
                    }
                    let pk = pawn_key(board);
                    let checks = check_squares(board);
                    for i in self.quiet_cur..self.list.len {
                        let m = self.list.get(i);
                        self.list.scores[i] = if m.promotion.is_some() {
                            -HIST_MAX * 20
                        } else {
                            let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
                            let bonus = if checks[piece as usize].has(m.to) && see_ge(board, m, -75) {
                                16_384
                            } else {
                                0
                            };
                            s.quiet_score_pk(board, m, sp, pk) + bonus
                        };
                    }
                    self.stage = Stage::Quiets;
                }
                Stage::Quiets => {
                    if !skip_quiets {
                        let len = self.list.len;
                        if let Some((m, sc)) = self.list.pick_range(self.quiet_cur, len) {
                            self.quiet_cur += 1;
                            return Some((m, sc.min(STAGE_SCORE_QUIET_LIMIT - 1)));
                        }
                    }
                    self.stage = Stage::BadCaptures;
                }
                Stage::BadCaptures => {
                    if let Some((m, sc)) = self.list.pick_range(self.bad_start, self.n_caps) {
                        self.bad_start += 1;
                        return Some((m, sc));
                    }
                    self.stage = Stage::Done;
                }
                Stage::Done => return None,
            }
        }
    }
}

fn gen_all(board: &Board, list: &mut MoveList) {
    stat(8);
    board.generate_moves(|pm| {
        for m in pm {
            list.push(m);
        }
        false
    });
}

fn gen_tactical(board: &Board, list: &mut MoveList) {
    stat(9);
    let stm = board.side_to_move();
    let enemy = board.colors(!stm);
    let ep_bb = board
        .en_passant()
        .map(|f| {
            let rank = if stm == Color::White {
                cozy_chess::Rank::Sixth
            } else {
                cozy_chess::Rank::Third
            };
            Square::new(f, rank).bitboard()
        })
        .unwrap_or(BitBoard::EMPTY);
    let promo_rank = if stm == Color::White {
        cozy_chess::Rank::Eighth.bitboard()
    } else {
        cozy_chess::Rank::First.bitboard()
    };
    board.generate_moves(|mut pm| {
        if pm.piece == Piece::Pawn {
            pm.to &= enemy | ep_bb | promo_rank;
        } else {
            pm.to &= enemy;
        }
        for m in pm {
            if m.promotion.is_none() || m.promotion == Some(Piece::Queen) {
                list.push(m);
            }
        }
        false
    });
}

const SCORE_TT: i32 = 2_000_000_000;
const SCORE_GOOD_CAPTURE: i32 = 1_000_000_000;
const SCORE_KILLER1: i32 = 900_000_000;
const SCORE_KILLER2: i32 = 899_000_000;
const SCORE_COUNTER: i32 = 898_000_000;
const SCORE_BAD_CAPTURE: i32 = -1_000_000_000;

// ---------------------------------------------------------------------------
// Searcher
// ---------------------------------------------------------------------------

pub struct Searcher {
    tt: Arc<PackedTt>,
    tt_mb: usize,
    pub(crate) nodes: u64,
    node_limit: u64,
    hard_deadline: Option<Instant>,
    threads: usize,
    smp_pool: Option<Arc<rayon::ThreadPool>>,
    smp_workers: Vec<Searcher>,
    helper_stop: Option<Arc<AtomicBool>>,
    helper_depth_offset: u32,
    is_helper: bool,
    external_stop: Option<Arc<AtomicBool>>,
    stopped: bool,
    use_lmr: bool,
    use_nullmove: bool,
    use_aspiration: bool,
    use_tt: bool,
    order_captures: bool,
    use_history: bool,
    use_killers: bool,
    deterministic: bool,
    use_nnue: bool,
    nnue: Option<crate::eval::nnue::Nnue>,
    nnue_quant: Option<QuantNetwork>,
    eval_blend_percent: u8,
    eval_mode: EvalMode,
    // History tables
    main_hist: Vec<i16>,       // [color][from*64+to]
    cont_hist: Vec<i16>,       // [prev piece-square 768][piece-square 768]
    capt_hist: Vec<i16>,       // [piece 12][to 64][victim 6]
    pawn_hist: Vec<i16>,       // [pawn key % PAWN_HIST_SIZE][piece-square 768]
    counter_moves: Vec<u16>,   // [prev piece-square 768]
    pawn_corr: Vec<i32>,       // [color][CORR_SIZE]
    nonpawn_corr: Vec<i32>,    // [color][side][CORR_SIZE]
    minor_corr: Vec<i32>,      // [color][CORR_SIZE]
    cont_corr: Vec<i32>,       // [prev2 768][prev1 768]
    stack: Vec<Frame>,
    // Repetition detection: hashes of the game and the current line.
    hash_hist: Vec<u64>,
    root_history: Vec<Board>,
    root_index: usize,
    null_floor: usize,
    // Root state
    root_depth: i32,
    sel_depth: u32,
    last_depth: u32,
    nmp_min_ply: i32,
    root_moves: Vec<RootMove>,
    lmr: Box<[[i32; 64]; 64]>,
}

#[derive(Clone, Copy)]
struct RootMove {
    mv: Move,
    score: i32,
    prev_score: i32,
    nodes: u64,
}

impl Default for Searcher {
    fn default() -> Self {
        Self {
            tt: Arc::new(PackedTt::with_mb(16)),
            tt_mb: 16,
            nodes: 0,
            node_limit: u64::MAX,
            hard_deadline: None,
            threads: 1,
            smp_pool: None,
            smp_workers: Vec::new(),
            helper_stop: None,
            helper_depth_offset: 0,
            is_helper: false,
            external_stop: None,
            stopped: false,
            use_lmr: true,
            use_nullmove: true,
            use_aspiration: true,
            use_tt: true,
            order_captures: true,
            use_history: true,
            use_killers: true,
            deterministic: false,
            use_nnue: false,
            nnue: None,
            nnue_quant: None,
            eval_blend_percent: 100,
            eval_mode: EvalMode::Pst,
            main_hist: vec![0; 2 * 4096],
            cont_hist: vec![0; 768 * 768],
            capt_hist: vec![0; 12 * 64 * 6],
            pawn_hist: vec![0; PAWN_HIST_SIZE * 768],
            counter_moves: vec![0; 768],
            pawn_corr: vec![0; 2 * CORR_SIZE],
            nonpawn_corr: vec![0; 2 * 2 * CORR_SIZE],
            minor_corr: vec![0; 2 * CORR_SIZE],
            cont_corr: vec![0; 768 * 768],
            stack: vec![Frame::default(); MAX_PLY + STACK_OFFSET + 8],
            hash_hist: Vec::with_capacity(1024),
            root_history: Vec::new(),
            root_index: 0,
            null_floor: 0,
            root_depth: 0,
            sel_depth: 0,
            last_depth: 0,
            nmp_min_ply: 0,
            root_moves: Vec::new(),
            lmr: lmr_table(),
        }
    }
}

#[inline]
fn hist_update(entry: &mut i16, bonus: i32) {
    let e = *entry as i32;
    let b = bonus.clamp(-HIST_MAX, HIST_MAX);
    let v = e + b - e * b.abs() / HIST_MAX;
    *entry = v.clamp(-HIST_MAX, HIST_MAX) as i16;
}

#[inline]
fn corr_update(entry: &mut i32, diff: i32, depth: i32) {
    let w = (depth + 1).min(16);
    let v = (*entry * (256 - w) + diff * CORR_GRAIN * w) / 256;
    *entry = v.clamp(-CORR_MAX, CORR_MAX);
}

impl Searcher {
    // ---------------- configuration API (baseline compatible) -------------

    pub fn continuation_history_entries_count(&self) -> usize {
        self.cont_hist.iter().filter(|&&x| x != 0).count()
    }

    pub fn set_position_history(&mut self, history: &[Board]) {
        self.root_history.clear();
        self.root_history.extend_from_slice(history);
    }

    pub fn clear_position_history(&mut self) {
        self.root_history.clear();
    }

    pub fn set_stop_flag(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.external_stop = flag;
    }

    pub fn clear_stop_flag(&mut self) {
        self.external_stop = None;
    }

    pub fn clear_history(&mut self) {
        self.main_hist.fill(0);
        self.cont_hist.fill(0);
        self.capt_hist.fill(0);
        self.pawn_hist.fill(0);
        self.counter_moves.fill(0);
        self.pawn_corr.fill(0);
        self.nonpawn_corr.fill(0);
        self.minor_corr.fill(0);
        self.cont_corr.fill(0);
        for f in self.stack.iter_mut() {
            *f = Frame::default();
        }
        self.tt.clear();
        for w in &mut self.smp_workers {
            w.clear_history();
        }
    }

    pub fn set_eval_mode(&mut self, mode: EvalMode) {
        self.eval_mode = mode;
        self.tt.clear();
    }
    pub fn set_threads(&mut self, t: usize) {
        self.threads = t.max(1);
        if self
            .smp_pool
            .as_ref()
            .is_some_and(|p| p.current_num_threads() != self.threads)
        {
            self.smp_pool = None;
        }
        self.smp_workers.truncate(self.threads.saturating_sub(1));
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
    pub fn get_threads(&self) -> usize {
        self.threads
    }
    pub fn last_depth(&self) -> u32 {
        self.last_depth
    }
    pub fn last_seldepth(&self) -> u32 {
        self.sel_depth
    }
    pub fn set_use_nnue(&mut self, on: bool) {
        if self.use_nnue != on {
            self.tt.clear();
        }
        self.use_nnue = on;
    }
    pub fn set_nnue_network(&mut self, nn: Option<crate::eval::nnue::Nnue>) {
        self.smp_workers.clear();
        self.nnue = nn;
    }
    pub fn set_nnue_quant_model(&mut self, model: QuantNnue) {
        self.smp_workers.clear();
        let mut qn = QuantNetwork::new(model);
        // Build accumulators only for positions that are evaluated.
        qn.set_lazy(true);
        self.nnue_quant = Some(qn);
        self.tt.clear();
    }
    pub fn clear_nnue_quant(&mut self) {
        self.smp_workers.clear();
        self.nnue_quant = None;
    }
    pub fn set_eval_blend_percent(&mut self, p: u8) {
        if self.eval_blend_percent != p.min(100) {
            self.tt.clear();
        }
        self.eval_blend_percent = p.min(100);
    }
    pub fn eval_blend_percent(&self) -> u8 {
        self.eval_blend_percent
    }
    pub fn set_tt_capacity_mb(&mut self, mb: usize) {
        self.tt_mb = mb.max(1);
        self.tt = Arc::new(PackedTt::with_mb(self.tt_mb));
    }
    pub fn tt_probe(&self, board: &Board) -> Option<(u32, Bound)> {
        self.tt
            .probe(board.hash())
            .and_then(|e| e.bound.map(|b| (e.depth.max(0) as u32, b)))
    }
    #[doc(hidden)]
    pub fn set_killer_for_test(&mut self, ply: usize, slot: usize, m: Move) {
        if slot < 2 && ply + STACK_OFFSET < self.stack.len() {
            self.stack[ply + STACK_OFFSET].killers[slot] = Some(m);
        }
    }

    /// Null-move depth reduction: grows with depth and with how far the
    /// static evaluation sits above beta.
    pub fn null_move_reduction(&self, depth: u32, eval: i32, beta: i32) -> u32 {
        let r = 4 + depth as i32 / 3 + ((eval - beta) / 200).clamp(0, 6);
        r.clamp(1, depth.max(1) as i32) as u32
    }

    pub fn see_gain_cp(&mut self, board: &Board, uci: &str) -> Option<i32> {
        let mut chosen = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{m}") == uci {
                    chosen = Some(m);
                }
            }
            chosen.is_some()
        });
        chosen.and_then(|m| crate::search::see::see_gain_cp(board, m))
    }

    pub fn debug_order_root(&self, board: &Board) -> Vec<Move> {
        let mut list = MoveList::new();
        gen_all(board, &mut list);
        let tt_move = self.tt.probe(board.hash()).and_then(|e| e.mv);
        self.score_moves(board, &mut list, tt_move, STACK_OFFSET);
        let mut out = Vec::with_capacity(list.len);
        for i in 0..list.len {
            if let Some((m, _)) = list.pick(i) {
                out.push(m);
            }
        }
        out
    }

    pub fn qsearch_eval_cp(&mut self, board: &Board) -> i32 {
        self.nodes = 0;
        self.sel_depth = 0;
        self.node_limit = u64::MAX;
        self.hard_deadline = None;
        self.stopped = false;
        self.prepare_root(board);
        self.qsearch(board, -INF, INF, 0, true)
            .unwrap_or_else(|_| self.raw_eval(board))
    }

    // ---------------- evaluation ------------------------------------------

    #[inline]
    fn raw_eval(&mut self, board: &Board) -> i32 {
        stat(2);
        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.ensure_computed();
            }
        }
        let v = match self.eval_mode {
            EvalMode::Material => material_eval_cp(board),
            EvalMode::Pst | EvalMode::Nnue => {
                let nn = if self.use_nnue {
                    if let Some(qn) = &self.nnue_quant {
                        Some(qn.eval_current())
                    } else {
                        self.nnue.as_ref().map(|nn| nn.evaluate(board))
                    }
                } else {
                    None
                };
                match nn {
                    Some(raw) => {
                        let stm_raw = if board.side_to_move() == Color::White {
                            raw
                        } else {
                            -raw
                        };
                        if self.eval_blend_percent == 100 {
                            stm_raw
                        } else {
                            let w = self.eval_blend_percent as i32;
                            (stm_raw * w + eval_cp(board) * (100 - w)) / 100
                        }
                    }
                    None => eval_cp(board),
                }
            }
        };
        v.clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    #[inline]
    fn correction(&self, board: &Board, ply: usize) -> i32 {
        stat(10);
        let stm = board.side_to_move() as usize;
        let p = self.pawn_corr[stm * CORR_SIZE + (pawn_key(board) as usize % CORR_SIZE)];
        let nw = self.nonpawn_corr
            [(stm * 2) * CORR_SIZE + (non_pawn_key(board, Color::White) as usize % CORR_SIZE)];
        let nb = self.nonpawn_corr
            [(stm * 2 + 1) * CORR_SIZE + (non_pawn_key(board, Color::Black) as usize % CORR_SIZE)];
        let mi = self.minor_corr[stm * CORR_SIZE + (minor_key(board) as usize % CORR_SIZE)];
        let cc = match (self.stack[ply - 1].cont, self.stack[ply - 2].cont) {
            (Some(a), Some(b)) => self.cont_corr[b * 768 + a],
            _ => 0,
        };
        (tp(T::CorrPawn) * p + nw + nb + mi + cc) / (2 * CORR_GRAIN)
    }

    fn update_correction(&mut self, board: &Board, ply: usize, diff: i32, depth: i32) {
        let stm = board.side_to_move() as usize;
        let pk = pawn_key(board) as usize % CORR_SIZE;
        corr_update(&mut self.pawn_corr[stm * CORR_SIZE + pk], diff, depth);
        let wk = non_pawn_key(board, Color::White) as usize % CORR_SIZE;
        corr_update(&mut self.nonpawn_corr[(stm * 2) * CORR_SIZE + wk], diff, depth);
        let bk = non_pawn_key(board, Color::Black) as usize % CORR_SIZE;
        corr_update(&mut self.nonpawn_corr[(stm * 2 + 1) * CORR_SIZE + bk], diff, depth);
        let mk = minor_key(board) as usize % CORR_SIZE;
        corr_update(&mut self.minor_corr[stm * CORR_SIZE + mk], diff, depth);
        if let (Some(a), Some(b)) = (self.stack[ply - 1].cont, self.stack[ply - 2].cont) {
            corr_update(&mut self.cont_corr[b * 768 + a], diff, depth);
        }
    }

    #[inline]
    fn adjust_eval(&self, board: &Board, raw: i32, ply: usize) -> i32 {
        let mut v = raw + self.correction(board, ply);
        // Drift toward a draw as the fifty-move counter runs.
        let hm = board.halfmove_clock() as i32;
        if hm > 16 {
            v = v * (216 - hm) / 200;
        }
        v.clamp(-MATE_IN_MAX + 1, MATE_IN_MAX - 1)
    }

    // ---------------- NNUE incremental helpers -----------------------------

    #[inline]
    fn nnue_apply(&mut self, board: &Board, m: Move, child: &Board) -> Option<ChangeSet> {
        stat(3);
        if !self.use_nnue {
            return None;
        }
        self.nnue_quant
            .as_mut()
            .map(|qn| qn.apply_move(board, m, child))
    }

    #[inline]
    fn nnue_apply_null(&mut self, after: &Board) -> Option<ChangeSet> {
        if !self.use_nnue {
            return None;
        }
        self.nnue_quant.as_mut().map(|qn| qn.apply_null_move(after))
    }

    #[inline]
    fn nnue_revert(&mut self, change: Option<ChangeSet>) {
        if let Some(c) = change {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.revert(c);
            }
        }
    }

    // ---------------- draws -----------------------------------------------

    /// Score of a drawn position `ply` plies below the root, for the side to
    /// move there.
    #[inline]
    fn draw_value(&self, ply: i32) -> i32 {
        let c = CONTEMPT.load(Ordering::Relaxed);
        if ply % 2 == 0 {
            c
        } else {
            -c
        }
    }

    #[inline]
    fn is_draw(&self, board: &Board) -> bool {
        let hm = board.halfmove_clock() as usize;
        if hm >= 100 {
            // Checkmate ends the game before a fifty-move claim.
            return board.checkers().is_empty() || has_legal_move(board);
        }
        if crate::search::draw::is_insufficient_material(board) {
            return true;
        }
        let n = self.hash_hist.len();
        if hm < 4 || n < 5 {
            return false;
        }
        let cur = n - 1;
        let key = self.hash_hist[cur];
        let lowest = cur.saturating_sub(hm).max(self.null_floor);
        let mut count = 0;
        let mut d = 4;
        while d <= cur - lowest {
            let i = cur - d;
            if self.hash_hist[i] == key {
                if i >= self.root_index {
                    return true;
                }
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
            d += 2;
        }
        false
    }

    // ---------------- abort / bookkeeping --------------------------------

    #[inline]
    fn check_abort(&mut self) -> Result<(), Abort> {
        if self.stopped {
            return Err(Abort);
        }
        if self.nodes >= self.node_limit {
            self.stopped = true;
            return Err(Abort);
        }
        if self.nodes & 1023 == 0 {
            if self
                .external_stop
                .as_ref()
                .is_some_and(|f| f.load(Ordering::Relaxed))
                || self
                    .helper_stop
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Relaxed))
                || self.hard_deadline.is_some_and(|d| Instant::now() >= d)
            {
                self.stopped = true;
                return Err(Abort);
            }
        }
        Ok(())
    }

    fn prepare_root(&mut self, board: &Board) {
        self.hash_hist.clear();
        for b in &self.root_history {
            self.hash_hist.push(b.hash());
        }
        if self.root_history.last() != Some(board) {
            self.hash_hist.push(board.hash());
        }
        self.root_index = self.hash_hist.len() - 1;
        self.null_floor = 0;
        self.nmp_min_ply = 0;
        for f in self.stack.iter_mut() {
            f.static_eval = 0;
            f.cont = None;
            f.mv = None;
            f.excluded = None;
            f.tt_pv = false;
            f.in_check = false;
            f.cutoff_cnt = 0;
            f.stat_score = 0;
            f.double_ext = 0;
        }
        if self.use_nnue {
            if let Some(qn) = self.nnue_quant.as_mut() {
                qn.refresh(board);
            }
        }
    }

    // ---------------- move ordering ---------------------------------------

    #[inline]
    fn quiet_score(&self, board: &Board, m: Move, sp: usize) -> i32 {
        self.quiet_score_pk(board, m, sp, pawn_key(board))
    }

    #[inline]
    fn quiet_score_pk(&self, board: &Board, m: Move, sp: usize, pk: u64) -> i32 {
        stat(11);
        let stm = board.side_to_move();
        let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
        let ps = pc_index(stm, piece) * 64 + m.to as usize;
        let mut s = 2 * self.main_hist[stm as usize * 4096 + m.from as usize * 64 + m.to as usize] as i32;
        s += self.pawn_hist[(pk as usize % PAWN_HIST_SIZE) * 768 + ps] as i32;
        for (k, w) in [(1usize, 2i32), (2, 1), (4, 1), (6, 1)] {
            if let Some(prev) = self.stack[sp - k].cont {
                s += w * self.cont_hist[prev * 768 + ps] as i32;
            }
        }
        s
    }

    #[inline]
    fn capture_score(&self, board: &Board, m: Move) -> i32 {
        let stm = board.side_to_move();
        let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
        let victim = captured_piece(board, m).unwrap_or(Piece::Pawn);
        let mut s = 7 * piece_value(victim)
            + self.capt_hist[(pc_index(stm, piece) * 64 + m.to as usize) * 6 + victim as usize] as i32 / 8;
        if m.promotion == Some(Piece::Queen) {
            s += 1000;
        }
        s
    }

    fn score_moves(&self, board: &Board, list: &mut MoveList, tt_move: Option<Move>, sp: usize) {
        let killers = self.stack[sp].killers;
        let counter = self.stack[sp - 1]
            .cont
            .and_then(|c| move_from_u16(self.counter_moves[c]));
        let checks = check_squares(board);
        for i in 0..list.len {
            stat(5);
            let m = list.get(i);
            list.scores[i] = if Some(m) == tt_move {
                SCORE_TT
            } else if is_tactical(board, m) {
                let s = self.capture_score(board, m);
                if see_ge(board, m, -s / 20) {
                    SCORE_GOOD_CAPTURE + s
                } else {
                    SCORE_BAD_CAPTURE + s
                }
            } else if Some(m) == killers[0] {
                SCORE_KILLER1
            } else if Some(m) == killers[1] {
                SCORE_KILLER2
            } else if Some(m) == counter {
                SCORE_COUNTER
            } else if m.promotion.is_some() {
                // Under-promotions last among quiets.
                -HIST_MAX * 20
            } else {
                let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
                let check_bonus = if checks[piece as usize].has(m.to) && see_ge(board, m, -75) {
                    16_384
                } else {
                    0
                };
                self.quiet_score(board, m, sp) + check_bonus
            };
        }
    }

    // ---------------- history updates -------------------------------------

    fn update_quiet_stats(&mut self, board: &Board, m: Move, sp: usize, bonus: i32) {
        let stm = board.side_to_move();
        let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
        let ps = pc_index(stm, piece) * 64 + m.to as usize;
        hist_update(
            &mut self.main_hist[stm as usize * 4096 + m.from as usize * 64 + m.to as usize],
            bonus,
        );
        let pk = pawn_key(board) as usize % PAWN_HIST_SIZE;
        hist_update(&mut self.pawn_hist[pk * 768 + ps], bonus);
        self.update_cont(sp, ps, bonus);
    }

    #[inline]
    fn update_cont(&mut self, sp: usize, ps: usize, bonus: i32) {
        for (k, w) in [(1usize, 1024i32), (2, 1024), (4, 512), (6, 512)] {
            if let Some(prev) = self.stack[sp - k].cont {
                hist_update(&mut self.cont_hist[prev * 768 + ps], bonus * w / 1024);
            }
        }
    }

    fn update_capture_stat(&mut self, board: &Board, m: Move, bonus: i32) {
        let stm = board.side_to_move();
        let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
        let victim = captured_piece(board, m).unwrap_or(Piece::Pawn);
        hist_update(
            &mut self.capt_hist[(pc_index(stm, piece) * 64 + m.to as usize) * 6 + victim as usize],
            bonus,
        );
    }

    // ---------------- quiescence ------------------------------------------

    fn qsearch(&mut self, board: &Board, mut alpha: i32, beta: i32, ply: i32, pv: bool) -> SResult {
        self.check_abort()?;
        self.nodes += 1;
        stat(1);
        let sp = ply as usize + STACK_OFFSET;
        if pv {
            self.sel_depth = self.sel_depth.max(ply as u32);
        }
        if ply > 0 && self.is_draw(board) {
            return Ok(self.draw_value(ply));
        }
        let in_check = !board.checkers().is_empty();
        if ply as usize >= MAX_PLY - 1 {
            return Ok(if in_check { DRAW_SCORE } else { self.raw_eval(board) });
        }

        let key = board.hash();
        let tte = if self.use_tt { self.tt.probe(key) } else { None };
        let halfmove = board.halfmove_clock();
        let rule50_safe = (halfmove as u32) < FIFTY_MOVE_CLAIM_PLIES - 10;
        let tt_score = tte.map(|e| score_from_tt(e.score, ply, halfmove));
        if let (Some(e), Some(ts)) = (tte, tt_score) {
            if !pv && rule50_safe {
                match e.bound {
                    Some(Bound::Exact) => return Ok(ts),
                    Some(Bound::Lower) if ts >= beta => return Ok(ts),
                    Some(Bound::Upper) if ts <= alpha => return Ok(ts),
                    _ => {}
                }
            }
        }
        let tt_pv = tte.is_some_and(|e| e.pv);

        let mut best;
        let raw;
        let futility_base;
        if in_check {
            raw = -INF;
            best = -INF;
            futility_base = -INF;
        } else {
            raw = match tte {
                Some(e) if e.eval > -32_000 && e.eval < 32_000 => e.eval,
                _ => self.raw_eval(board),
            };
            let static_eval = self.adjust_eval(board, raw, sp);
            best = static_eval;
            if let (Some(e), Some(ts)) = (tte, tt_score) {
                if !is_decisive(ts)
                    && match e.bound {
                        Some(Bound::Exact) => true,
                        Some(Bound::Lower) => ts > best,
                        Some(Bound::Upper) => ts < best,
                        None => false,
                    }
                {
                    best = ts;
                }
            }
            if best >= beta {
                if !is_decisive(best) {
                    best = (best + beta) / 2;
                }
                if tte.is_none() && self.use_tt {
                    self.tt.store(key, None, score_to_tt(best, ply), raw, -1, Some(Bound::Lower), false);
                }
                return Ok(best);
            }
            if best > alpha {
                alpha = best;
            }
            futility_base = static_eval + tp(T::QsFutility);
        }

        let mut list = MoveList::new();
        if in_check {
            gen_all(board, &mut list);
        } else {
            gen_tactical(board, &mut list);
        }
        if list.len == 0 && !in_check && !has_legal_move(board) {
            return Ok(self.draw_value(ply));
        }
        let tt_move = tte.and_then(|e| e.mv).filter(|m| list.contains(*m));
        for i in 0..list.len {
            let m = list.get(i);
            list.scores[i] = if Some(m) == tt_move {
                SCORE_TT
            } else if is_tactical(board, m) {
                SCORE_GOOD_CAPTURE + self.capture_score(board, m)
            } else {
                self.quiet_score(board, m, sp)
            };
        }

        let mut best_move = None;
        let mut move_count = 0;
        let mut idx = 0;
        while let Some((m, mscore)) = list.pick(idx) {
            idx += 1;
            move_count += 1;
            let mut child = board.clone();
            child.play_unchecked(m);
            let gives_check = !child.checkers().is_empty();
            if !is_decisive(best) && non_pawn_material(board, board.side_to_move()) {
                if !in_check && !gives_check && m.promotion.is_none() {
                    // Futility: even winning the victim outright cannot
                    // reach alpha.
                    let victim = captured_piece(board, m).map(piece_value).unwrap_or(0);
                    let fv = futility_base + victim;
                    if fv <= alpha {
                        best = best.max(fv);
                        continue;
                    }
                    if futility_base <= alpha && !see_ge(board, m, 1) {
                        best = best.max(futility_base);
                        continue;
                    }
                }
                if in_check && move_count > 2 && !is_capture(board, m) && !gives_check && mscore < 0 {
                    // Late quiet evasions with a bad history, once a defence
                    // has been found.
                    if best > -MATE_IN_MAX {
                        continue;
                    }
                }
                if !see_ge(board, m, -tp(T::QsSee)) {
                    continue;
                }
            }
            self.tt.prefetch(child.hash());
            let change = self.nnue_apply(board, m, &child);
            let stm = board.side_to_move();
            let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
            self.stack[sp].mv = Some(m);
            self.stack[sp].cont = Some(pc_index(stm, piece) * 64 + m.to as usize);
            self.hash_hist.push(child.hash());
            let r = self.qsearch(&child, -beta, -alpha, ply + 1, pv);
            self.hash_hist.pop();
            self.nnue_revert(change);
            let score = -r?;
            if score > best {
                best = score;
                if score > alpha {
                    best_move = Some(m);
                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
        }

        if in_check && best == -INF {
            return Ok(mated_in(ply));
        }
        if !is_decisive(best) && best >= beta {
            best = (best + beta) / 2;
        }
        if self.use_tt {
            let bound = if best >= beta { Bound::Lower } else { Bound::Upper };
            self.tt.store(
                key,
                best_move,
                score_to_tt(best, ply),
                if in_check { -32_001 } else { raw },
                0,
                Some(bound),
                tt_pv,
            );
        }
        Ok(best)
    }

    // ---------------- main search -----------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        board: &Board,
        mut depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: i32,
        cut_node: bool,
        pv: bool,
    ) -> SResult {
        let root = ply == 0;
        if depth <= 0 {
            return self.qsearch(board, alpha, beta, ply, pv);
        }
        depth = depth.min(MAX_PLY as i32 - 1);
        self.check_abort()?;
        self.nodes += 1;
        stat(0);
        let sp = ply as usize + STACK_OFFSET;
        if pv {
            self.sel_depth = self.sel_depth.max(ply as u32);
        }
        let in_check = !board.checkers().is_empty();
        self.stack[sp].in_check = in_check;
        let excluded = self.stack[sp].excluded;

        if !root {
            if self.is_draw(board) {
                return Ok(self.draw_value(ply));
            }
            if ply as usize >= MAX_PLY - 1 {
                return Ok(if in_check { DRAW_SCORE } else { self.raw_eval(board) });
            }
            // Mate distance pruning.
            alpha = alpha.max(mated_in(ply));
            beta = beta.min(MATE_SCORE - ply - 1);
            if alpha >= beta {
                return Ok(alpha);
            }
        }

        let key = board.hash();
        let halfmove = board.halfmove_clock();
        let tte = if self.use_tt && excluded.is_none() {
            self.tt.probe(key)
        } else {
            None
        };
        let tt_score = tte.map(|e| score_from_tt(e.score, ply, halfmove));
        let tt_move = if root {
            self.root_moves.first().map(|r| r.mv)
        } else {
            tte.and_then(|e| e.mv)
        };
        let tt_capture = tt_move.is_some_and(|m| is_tactical(board, m));
        let tt_pv = if excluded.is_some() {
            self.stack[sp].tt_pv
        } else {
            pv || tte.is_some_and(|e| e.pv)
        };
        self.stack[sp].tt_pv = tt_pv;
        self.stack[sp + 1].killers = [None, None];
        self.stack[sp + 2].cutoff_cnt = 0;
        self.stack[sp + 1].excluded = None;
        let rule50_safe = (halfmove as u32) + (depth as u32) < FIFTY_MOVE_CLAIM_PLIES;

        // TT cutoff.
        if !pv && excluded.is_none() && rule50_safe {
            if let (Some(e), Some(ts)) = (tte, tt_score) {
                if e.depth >= depth
                    && match e.bound {
                        Some(Bound::Exact) => true,
                        Some(Bound::Lower) => ts >= beta,
                        Some(Bound::Upper) => ts <= alpha,
                        None => false,
                    }
                {
                    if ts >= beta {
                        if let Some(m) = tt_move {
                            if !is_tactical(board, m) && board.is_legal(m) {
                                self.update_quiet_stats(board, m, sp, stat_bonus(depth));
                            }
                        }
                    }
                    return Ok(ts);
                }
            }
        }

        // Tablebase probe, right after a capture or pawn move.
        if !root && excluded.is_none() && halfmove == 0 {
            let tb_max = crate::search::syzygy::max_pieces();
            if tb_max > 0 && board.occupied().len() as usize <= tb_max {
                if let Some(w) = crate::search::syzygy::probe_wdl(board) {
                    let v = match w {
                        2 => TB_WIN - ply,
                        -2 => -TB_WIN + ply,
                        1 => self.draw_value(ply) + 1,
                        -1 => self.draw_value(ply) - 1,
                        _ => self.draw_value(ply),
                    };
                    if self.use_tt {
                        self.tt.store(key, None, v, -32_001, (depth + 6).min(MAX_PLY as i32 - 1), Some(Bound::Exact), tt_pv);
                    }
                    return Ok(v);
                }
            }
        }

        // Static evaluation.
        let raw;
        let static_eval;
        let mut eval;
        if in_check {
            raw = -32_001;
            static_eval = -INF;
            eval = -INF;
            self.stack[sp].static_eval = -INF;
        } else if excluded.is_some() {
            static_eval = self.stack[sp].static_eval;
            raw = static_eval;
            eval = static_eval;
        } else {
            raw = match tte {
                Some(e) if e.eval > -32_000 && e.eval < 32_000 => e.eval,
                _ => self.raw_eval(board),
            };
            static_eval = self.adjust_eval(board, raw, sp);
            eval = static_eval;
            self.stack[sp].static_eval = static_eval;
            if let (Some(e), Some(ts)) = (tte, tt_score) {
                if !is_decisive(ts)
                    && match e.bound {
                        Some(Bound::Exact) => true,
                        Some(Bound::Lower) => ts > eval,
                        Some(Bound::Upper) => ts < eval,
                        None => false,
                    }
                {
                    eval = ts;
                }
            } else if tte.is_none() && self.use_tt {
                self.tt.store(key, None, 0, raw, -1, None, tt_pv);
            }
        }

        let improving = !in_check && {
            let p2 = self.stack[sp - 2];
            if !p2.in_check && ply >= 2 {
                static_eval > p2.static_eval
            } else if ply >= 4 && !self.stack[sp - 4].in_check {
                static_eval > self.stack[sp - 4].static_eval
            } else {
                true
            }
        };
        let opp_worsening = !in_check
            && ply >= 1
            && !self.stack[sp - 1].in_check
            && static_eval > -self.stack[sp - 1].static_eval;

        let prune_ok = self.use_nullmove && !root && !in_check && excluded.is_none();

        if prune_ok && !pv {
            // Razoring.
            if eval < alpha - tp(T::RazorBase) - tp(T::RazorMul) * depth * depth {
                let v = self.qsearch(board, alpha - 1, alpha, ply, false)?;
                if v < alpha && !is_decisive(v) {
                    return Ok(v);
                }
            }

            // Reverse futility pruning.
            if !tt_pv && depth < tp(T::RfpDepth) && !is_decisive(eval) && eval >= beta && !is_decisive(beta) {
                let margin = (tp(T::RfpMul) - tp(T::RfpCut) * cut_node as i32 - tp(T::RfpWorsen) * opp_worsening as i32) * depth
                    - tp(T::RfpImproving) * improving as i32 * depth;
                if eval - margin.max(20) >= beta {
                    return Ok((eval + beta) / 2);
                }
            }

            // Null-move pruning.
            if cut_node
                && self.stack[sp - 1].mv.is_some()
                && eval >= beta
                && static_eval >= beta - tp(T::NmpStaticMul) * depth + tp(T::NmpStaticBase)
                && ply >= self.nmp_min_ply
                && non_pawn_material(board, board.side_to_move())
                && !is_decisive(beta)
            {
                let r = (tp(T::NmpBase) + depth / tp(T::NmpDepthDiv) + ((eval - beta) / tp(T::NmpEvalDiv)).min(6)).min(depth);
                if let Some(nb) = board.null_move() {
                    self.tt.prefetch(nb.hash());
                    let change = self.nnue_apply_null(&nb);
                    let floor_before = self.null_floor;
                    self.hash_hist.push(nb.hash());
                    self.null_floor = self.hash_hist.len() - 1;
                    self.stack[sp].mv = None;
                    self.stack[sp].cont = None;
                    let res = self.search(&nb, depth - r, -beta, -beta + 1, ply + 1, false, false);
                    self.hash_hist.pop();
                    self.null_floor = floor_before;
                    self.nnue_revert(change);
                    let null_value = -res?;
                    if null_value >= beta && !is_decisive(null_value) {
                        if self.nmp_min_ply != 0 || depth < 14 {
                            return Ok(null_value);
                        }
                        // Verification at high depth with null moves
                        // disabled for the first part of the subtree.
                        self.nmp_min_ply = ply + 3 * (depth - r) / 4;
                        let v = self.search(board, depth - r, beta - 1, beta, ply, false, false);
                        self.nmp_min_ply = 0;
                        if v? >= beta {
                            return Ok(null_value);
                        }
                    }
                }
            }
        }

        // Internal iterative reduction.
        if !root && (pv || cut_node) && depth >= tp(T::IirDepth) && tt_move.is_none() && excluded.is_none() {
            depth -= 1 + (pv && depth >= 8) as i32;
        }

        // ProbCut.
        if prune_ok && !pv && depth >= tp(T::ProbcutDepth) && !is_decisive(beta) {
            let pc_beta = beta + tp(T::ProbcutMargin) - tp(T::ProbcutImproving) * improving as i32;
            let tt_blocks = matches!((tte, tt_score), (Some(e), Some(ts)) if e.depth >= depth - 3 && e.bound.is_some() && ts < pc_beta);
            if !tt_blocks {
                let mut list = MoveList::new();
                gen_tactical(board, &mut list);
                for i in 0..list.len {
                    list.scores[i] = self.capture_score(board, list.get(i));
                }
                let mut idx = 0;
                let pc_depth = depth - 4;
                while let Some((m, _)) = list.pick(idx) {
                    idx += 1;
                    if Some(m) == excluded || !see_ge(board, m, pc_beta - static_eval) {
                        continue;
                    }
                    let mut child = board.clone();
                    child.play_unchecked(m);
                    self.tt.prefetch(child.hash());
                    let change = self.nnue_apply(board, m, &child);
                    let stm = board.side_to_move();
                    let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
                    self.stack[sp].mv = Some(m);
                    self.stack[sp].cont = Some(pc_index(stm, piece) * 64 + m.to as usize);
                    self.hash_hist.push(child.hash());
                    let mut r = self.qsearch(&child, -pc_beta, -pc_beta + 1, ply + 1, false).map(|v| -v);
                    if matches!(r, Ok(v) if v >= pc_beta) && pc_depth > 0 {
                        r = self
                            .search(&child, pc_depth, -pc_beta, -pc_beta + 1, ply + 1, !cut_node, false)
                            .map(|v| -v);
                    }
                    self.hash_hist.pop();
                    self.nnue_revert(change);
                    let v = r?;
                    if v >= pc_beta {
                        if self.use_tt {
                            self.tt.store(key, Some(m), score_to_tt(v, ply), raw, depth - 3, Some(Bound::Lower), tt_pv);
                        }
                        if !is_decisive(v) {
                            return Ok(v - (pc_beta - beta));
                        }
                        return Ok(v);
                    }
                }
            }
        }

        // ---------------- move loop ----------------
        let mut root_list = MoveList::new();
        let mut picker = if root {
            for rm in &self.root_moves {
                root_list.push(rm.mv);
            }
            None
        } else {
            let counter = self.stack[sp - 1]
                .cont
                .and_then(|c| move_from_u16(self.counter_moves[c]));
            Some(MovePicker::new(tt_move, self.stack[sp].killers, counter))
        };
        let tt_move = if root {
            tt_move
        } else {
            tt_move.filter(|&m| board.is_legal(m))
        };

        let stm = board.side_to_move();
        let mut best = -INF;
        let mut best_move: Option<Move> = None;
        let mut move_count: usize = 0;
        let mut quiets: [Move; 64] = [Move { from: Square::A1, to: Square::A1, promotion: None }; 64];
        let mut n_quiets = 0;
        let mut captures: [Move; 32] = [Move { from: Square::A1, to: Square::A1, promotion: None }; 32];
        let mut n_captures = 0;
        let mut skip_quiets = false;
        let lmp_limit = ((tp(T::LmpBase) + depth * depth) / (2 - improving as i32)) as usize;

        let mut root_idx = 0;
        loop {
            let next = match picker.as_mut() {
                Some(p) => p.next(self, board, sp, skip_quiets),
                None => {
                    if root_idx < root_list.len {
                        root_idx += 1;
                        Some((root_list.get(root_idx - 1), 0))
                    } else {
                        None
                    }
                }
            };
            let Some((m, _mscore)) = next else { break };
            if Some(m) == excluded {
                continue;
            }
            let capture = is_tactical(board, m);
            move_count += 1;
            let nodes_before = self.nodes;
            let mut child = board.clone();
            child.play_unchecked(m);
            let gives_check = !child.checkers().is_empty();
            let piece = board.piece_on(m.from).unwrap_or(Piece::Pawn);
            let ps = pc_index(stm, piece) * 64 + m.to as usize;

            let hist = if capture {
                let victim = captured_piece(board, m).unwrap_or(Piece::Pawn);
                self.capt_hist[ps * 6 + victim as usize] as i32
            } else {
                let mut h = self.main_hist[stm as usize * 4096 + m.from as usize * 64 + m.to as usize] as i32;
                for k in [1usize, 2] {
                    if let Some(prev) = self.stack[sp - k].cont {
                        h += self.cont_hist[prev * 768 + ps] as i32;
                    }
                }
                h
            };

            // Shallow-depth pruning.
            if !root && best > -MATE_IN_MAX && non_pawn_material(board, stm) && self.use_nullmove {
                if move_count >= lmp_limit {
                    skip_quiets = true;
                }
                let base_r = self.lmr[depth.min(63) as usize][move_count.min(63)] / 1024;
                let lmr_depth = (depth - 1 - base_r).max(0);
                if capture || gives_check {
                    if !gives_check && lmr_depth < 7 && !in_check {
                        let victim = captured_piece(board, m).map(piece_value).unwrap_or(0);
                        if static_eval + tp(T::CapFutBase) + tp(T::CapFutMul) * lmr_depth + victim + hist / 8 <= alpha {
                            continue;
                        }
                    }
                    if !gives_check && !see_ge(board, m, -(tp(T::CapSeeMul) * depth).max(0) - hist / 64) {
                        continue;
                    }
                } else {
                    if lmr_depth < 6 && hist < -tp(T::HistPruneMul) * depth {
                        continue;
                    }
                    let lmr_depth_h = (lmr_depth + hist / tp(T::HistLmrDiv)).max(0);
                    if !in_check && lmr_depth_h < 10 {
                        let fut = static_eval + tp(T::FutBase) + tp(T::FutMul) * lmr_depth_h;
                        if fut <= alpha {
                            if best <= fut && !is_decisive(best) {
                                best = fut;
                            }
                            continue;
                        }
                    }
                    if !see_ge(board, m, -tp(T::QuietSeeMul) * lmr_depth_h * lmr_depth_h) {
                        continue;
                    }
                }
            }

            // Extensions.
            let mut extension = 0;
            if !root
                && Some(m) == tt_move
                && excluded.is_none()
                && depth >= tp(T::SeDepth)
                && ply < 2 * self.root_depth
            {
                if let (Some(e), Some(ts)) = (tte, tt_score) {
                    if e.depth >= depth - 3 && matches!(e.bound, Some(Bound::Lower | Bound::Exact)) && !is_decisive(ts) {
                        let s_beta = ts - (tp(T::SeMargin) + tp(T::SeTtPvMargin) * (tt_pv && !pv) as i32) * depth / 64;
                        let s_depth = (depth - 1) / 2;
                        self.stack[sp].excluded = Some(m);
                        let v = self.search(board, s_depth, s_beta - 1, s_beta, ply, cut_node, false);
                        self.stack[sp].excluded = None;
                        let v = v?;
                        if v < s_beta {
                            extension = 1;
                            if !pv && v < s_beta - tp(T::SeDouble) && self.stack[sp].double_ext <= 8 {
                                extension = 2;
                            }
                        } else if v >= beta && !is_decisive(v) {
                            // Multi-cut.
                            return Ok(v);
                        } else if ts >= beta {
                            extension = -2;
                        } else if cut_node {
                            extension = -1;
                        }
                    }
                }
            }
            if extension == 0 && gives_check && ply < 2 * self.root_depth {
                extension = 1;
            }
            self.stack[sp + 1].double_ext = self.stack[sp].double_ext + (extension >= 2) as i32;

            self.tt.prefetch(child.hash());
            let change = self.nnue_apply(board, m, &child);
            self.stack[sp].mv = Some(m);
            self.stack[sp].cont = Some(ps);
            self.stack[sp].tactical = capture;
            self.hash_hist.push(child.hash());
            let new_depth = depth - 1 + extension;

            let searched: SResult = (|| {
                if depth >= 2 && move_count > 1 + root as usize && self.use_lmr {
                    let mut r = self.lmr[depth.min(63) as usize][move_count.min(63)];
                    if tt_pv {
                        r -= tp(T::LmrTtPv);
                    }
                    if cut_node {
                        r += tp(T::LmrCut);
                    }
                    if tt_capture && !capture {
                        r += tp(T::LmrTtCapture);
                    }
                    if !improving {
                        r += tp(T::LmrNotImproving);
                    }
                    if gives_check {
                        r -= tp(T::LmrCheck);
                    }
                    if self.stack[sp + 1].cutoff_cnt > 2 {
                        r += tp(T::LmrCutoffCnt);
                    }
                    if Some(m) == tt_move {
                        r -= tp(T::LmrTtMove);
                    }
                    if capture {
                        r -= tp(T::LmrCapture);
                    }
                    let stat = if capture { hist * 2 } else { hist };
                    self.stack[sp].stat_score = stat;
                    r -= stat * 1024 / tp(T::LmrStatDiv);
                    let r_plies = r / 1024;
                    let d = (new_depth - r_plies).clamp(1, new_depth + 1);
                    let mut v = -self.search(&child, d, -alpha - 1, -alpha, ply + 1, true, false)?;
                    if v > alpha && d < new_depth {
                        let deeper = v > best + tp(T::DeeperBase) + tp(T::DeeperMul) * new_depth;
                        let shallower = v < best + tp(T::Shallower);
                        let nd = new_depth + deeper as i32 - shallower as i32;
                        if nd > d {
                            v = -self.search(&child, nd, -alpha - 1, -alpha, ply + 1, !cut_node, false)?;
                        }
                        if !capture {
                            let b = if v > alpha { stat_bonus(nd) } else { -stat_malus(nd) };
                            self.update_cont(sp, ps, b);
                        }
                    }
                    if pv && v > alpha {
                        v = -self.search(&child, new_depth, -beta, -alpha, ply + 1, false, true)?;
                    }
                    Ok(v)
                } else {
                    let mut v = -INF;
                    if !pv || move_count > 1 {
                        v = -self.search(&child, new_depth, -alpha - 1, -alpha, ply + 1, !cut_node, false)?;
                    }
                    if pv && (move_count == 1 || v > alpha) {
                        v = -self.search(&child, new_depth, -beta, -alpha, ply + 1, false, true)?;
                    }
                    Ok(v)
                }
            })();
            self.hash_hist.pop();
            self.nnue_revert(change);
            let value = searched?;

            if root {
                if let Some(rm) = self.root_moves.iter_mut().find(|r| r.mv == m) {
                    rm.nodes += self.nodes - nodes_before;
                    if move_count == 1 || value > alpha {
                        rm.score = value;
                    } else {
                        rm.score = -INF;
                    }
                }
            }

            if value > best {
                best = value;
                if value > alpha {
                    best_move = Some(m);
                    if value >= beta {
                        self.stack[sp].cutoff_cnt += 1;
                        break;
                    }
                    alpha = value;
                    if depth > 2 && depth < 14 && !is_decisive(value) {
                        depth -= 1;
                    }
                }
            }
            if Some(m) != best_move {
                if capture {
                    if n_captures < 32 {
                        captures[n_captures] = m;
                        n_captures += 1;
                    }
                } else if n_quiets < 64 {
                    quiets[n_quiets] = m;
                    n_quiets += 1;
                }
            }
        }

        if move_count == 0 && !root {
            return Ok(if excluded.is_some() {
                alpha
            } else if in_check {
                mated_in(ply)
            } else {
                self.draw_value(ply)
            });
        }

        // History updates.
        if let Some(bm) = best_move {
            if best >= beta {
                let bonus = stat_bonus(depth + (best > beta + 100) as i32);
                let malus = stat_malus(depth);
                if !is_tactical(board, bm) {
                    self.update_quiet_stats(board, bm, sp, bonus);
                    for &q in &quiets[..n_quiets] {
                        self.update_quiet_stats(board, q, sp, -malus);
                    }
                    let k = &mut self.stack[sp].killers;
                    if k[0] != Some(bm) {
                        k[1] = k[0];
                        k[0] = Some(bm);
                    }
                    if let Some(prev) = self.stack[sp - 1].cont {
                        self.counter_moves[prev] = move_to_u16(bm);
                    }
                } else {
                    self.update_capture_stat(board, bm, bonus);
                }
                for &c in &captures[..n_captures] {
                    self.update_capture_stat(board, c, -malus);
                }
            }
        } else if !root && ply >= 1 {
            // Fail low: reward the opponent's previous quiet move.
            if let (Some(_), Some(prev_ps)) = (self.stack[sp - 1].mv, self.stack[sp - 1].cont) {
                let prev_sp = sp - 1;
                if !self.stack[prev_sp].in_check && !self.stack[prev_sp].tactical {
                    let b = stat_bonus(depth) / 2;
                    for (k, w) in [(1usize, 1024i32), (2, 1024), (4, 512)] {
                        if prev_sp >= k {
                            if let Some(pp) = self.stack[prev_sp - k].cont {
                                hist_update(&mut self.cont_hist[pp * 768 + prev_ps], b * w / 1024);
                            }
                        }
                    }
                }
            }
        }

        if excluded.is_none() {
            let bound = if best >= beta {
                Bound::Lower
            } else if pv && best_move.is_some() {
                Bound::Exact
            } else {
                Bound::Upper
            };
            // Correction history.
            if !in_check
                && best_move.map_or(true, |bm| !is_tactical(board, bm))
                && !(bound == Bound::Lower && best <= static_eval)
                && !(bound == Bound::Upper && best >= static_eval)
                && !is_decisive(best)
            {
                let diff = best - static_eval;
                self.update_correction(board, sp, diff, depth);
            }
            if self.use_tt {
                let stored_depth = if rule50_safe { depth } else { 0 };
                self.tt.store(key, best_move, score_to_tt(best, ply), raw, stored_depth, Some(bound), tt_pv);
            }
        }
        Ok(best)
    }

    // ---------------- iterative deepening ---------------------------------

    fn init_root_moves(&mut self, board: &Board) {
        let mut list = MoveList::new();
        gen_all(board, &mut list);
        let tt_move = self.tt.probe(board.hash()).and_then(|e| e.mv);
        self.score_moves(board, &mut list, tt_move, STACK_OFFSET);
        self.root_moves.clear();
        for i in 0..list.len {
            if let Some((m, _)) = list.pick(i) {
                self.root_moves.push(RootMove {
                    mv: m,
                    score: -INF,
                    prev_score: -INF,
                    nodes: 0,
                });
            }
        }
    }

    /// Iterative deepening. `soft_ms`, when given, is the time after which no
    /// new iteration starts (scaled by move stability); the hard deadline is
    /// enforced inside the tree.
    fn iterate(&mut self, board: &Board, max_depth: u32, soft: Option<Duration>, start: Instant) -> SearchResult {
        self.prepare_root(board);
        self.init_root_moves(board);
        self.stopped = false;
        self.sel_depth = 0;
        self.last_depth = 0;
        if self.root_moves.is_empty() {
            let score = if board.checkers().is_empty() { DRAW_SCORE } else { -MATE_SCORE };
            return SearchResult {
                bestmove: None,
                score_cp: score,
                nodes: self.nodes,
                depth: 0,
            };
        }
        let mut best = SearchResult {
            bestmove: Some(format!("{}", self.root_moves[0].mv)),
            score_cp: 0,
            nodes: 0,
            depth: 0,
        };
        if let Some(tb_move) = crate::search::syzygy::root_move(board) {
            if let Some(rm) = self.root_moves.iter().copied().find(|r| format!("{}", r.mv) == tb_move) {
                self.root_moves = vec![rm];
                best.bestmove = Some(tb_move);
            }
        }
        let root_key = board.hash();
        if board.halfmove_clock() >= 100
            || crate::search::draw::is_insufficient_material(board)
            || self.hash_hist.iter().filter(|&&h| h == root_key).count() >= 3
        {
            best.score_cp = DRAW_SCORE;
            return best;
        }
        let mut best_mv = self.root_moves[0].mv;
        let mut prev_score: i32 = 0;
        let mut stability = 0;
        let mut prev_iter_scores: Vec<i32> = Vec::new();
        let single = self.root_moves.len() == 1;
        let max_depth = max_depth.min(MAX_PLY as u32 - 8);
        let mut depth = 1 + self.helper_depth_offset;
        while depth <= max_depth {
            self.root_depth = depth as i32;
            for rm in self.root_moves.iter_mut() {
                rm.prev_score = rm.score;
                rm.nodes = 0;
            }
            let mut delta = tp(T::AspDelta) + prev_score.abs() / tp(T::AspDiv);
            let (mut alpha, mut beta) = if depth >= 4 && self.use_aspiration && !is_decisive(prev_score) {
                ((prev_score - delta).max(-INF), (prev_score + delta).min(INF))
            } else {
                (-INF, INF)
            };
            let mut fail_high_count = 0;
            let result = loop {
                let adj_depth = (depth as i32 - fail_high_count / 2).max(1);
                let r = self.search(board, adj_depth, alpha, beta, 0, false, true);
                // Stable sort: best score first, unsearched moves keep order.
                self.root_moves.sort_by(|a, b| b.score.cmp(&a.score));
                match r {
                    Err(_) => break None,
                    Ok(v) => {
                        if v <= alpha {
                            beta = (alpha + beta) / 2;
                            alpha = (v - delta).max(-INF);
                            fail_high_count = 0;
                        } else if v >= beta {
                            beta = (v + delta).min(INF);
                            fail_high_count += 1;
                        } else {
                            break Some(v);
                        }
                        delta += delta / 2 + 5;
                        if delta > 1000 {
                            alpha = -INF;
                            beta = INF;
                        }
                    }
                }
            };
            match result {
                Some(v) => {
                    let mv = self.root_moves[0].mv;
                    if mv == best_mv {
                        stability = (stability + 1).min(10);
                    } else {
                        stability = 0;
                    }
                    best_mv = mv;
                    prev_score = v;
                    best = SearchResult {
                        bestmove: Some(format!("{mv}")),
                        score_cp: v,
                        nodes: self.nodes,
                        depth,
                    };
                    self.last_depth = depth;
                    prev_iter_scores.push(v);
                }
                None => {
                    // Interrupted: keep a move that has already proved itself
                    // better than the previous best at this depth.
                    let top = self.root_moves[0];
                    if top.mv != best_mv && top.score > -INF && top.score > prev_score - 30 && depth > 1 {
                        best.bestmove = Some(format!("{}", top.mv));
                        best.score_cp = top.score;
                    }
                    break;
                }
            }
            if single {
                break;
            }
            if let Some(soft) = soft {
                if !self.is_helper {
                    let elapsed = start.elapsed();
                    let total: u64 = self.root_moves.iter().map(|r| r.nodes).sum::<u64>().max(1);
                    let best_frac = self.root_moves[0].nodes as f64 / total as f64;
                    let stab_factor = [2.2, 1.6, 1.3, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.78, 0.75][stability as usize];
                    let node_factor = (1.6 - best_frac) * 1.35;
                    let n = prev_iter_scores.len();
                    let falling = if n >= 3 {
                        let drop = prev_iter_scores[n - 3] - prev_iter_scores[n - 1];
                        (1.0 + drop as f64 / 200.0).clamp(0.85, 1.5)
                    } else {
                        1.0
                    };
                    let limit = soft.as_secs_f64() * stab_factor * node_factor * falling;
                    if elapsed.as_secs_f64() >= limit {
                        break;
                    }
                }
            }
            depth += 1;
        }
        best.nodes = self.nodes;
        best.depth = self.last_depth;
        best
    }

    fn can_use_smp(&self, threads: usize) -> bool {
        threads > 1 && !self.deterministic && (!self.use_nnue || self.nnue_quant.is_some())
    }

    fn with_smp<R: Send>(&mut self, board: &Board, threads: usize, main: impl FnOnce(&mut Self) -> R + Send) -> R {
        if self.smp_pool.is_none() {
            match rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("piebot-smp-{i}"))
                .build()
            {
                Ok(pool) => self.smp_pool = Some(Arc::new(pool)),
                Err(_) => return main(self),
            }
        }
        let pool = Arc::clone(self.smp_pool.as_ref().unwrap());
        while self.smp_workers.len() < threads - 1 {
            let mut w = Self::default();
            w.nnue_quant = self.nnue_quant.as_ref().map(QuantNetwork::clone_for_search);
            w.nnue = None;
            self.smp_workers.push(w);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut workers = std::mem::take(&mut self.smp_workers);
        for (i, w) in workers.iter_mut().enumerate() {
            w.tt = Arc::clone(&self.tt);
            w.root_history.clone_from(&self.root_history);
            w.external_stop.clone_from(&self.external_stop);
            w.helper_stop = Some(Arc::clone(&stop));
            w.helper_depth_offset = (i % 2) as u32;
            w.is_helper = true;
            w.use_nnue = self.use_nnue;
            w.eval_mode = self.eval_mode;
            w.eval_blend_percent = self.eval_blend_percent;
            w.use_lmr = self.use_lmr;
            w.use_nullmove = self.use_nullmove;
            w.use_aspiration = self.use_aspiration;
            w.use_tt = self.use_tt;
            w.node_limit = u64::MAX;
            w.hard_deadline = self.hard_deadline;
            w.nodes = 0;
            w.lmr = self.lmr.clone();
        }
        let result = pool.scope(|scope| {
            let _guard = StopHelpersOnDrop(stop);
            for w in &mut workers {
                scope.spawn(move |_| {
                    w.iterate(board, MAX_PLY as u32, None, Instant::now());
                });
            }
            main(self)
        });
        for w in &mut workers {
            self.nodes = self.nodes.saturating_add(w.nodes);
            w.helper_stop = None;
        }
        self.smp_workers = workers;
        result
    }

    fn run(&mut self, board: &Board, depth: u32, movetime: Option<Duration>, soft: Option<Duration>, threads: usize) -> SearchResult {
        self.nodes = 0;
        self.lmr = lmr_table();
        for w in &mut self.smp_workers {
            w.lmr = self.lmr.clone();
        }
        self.tt.new_search();
        let start = Instant::now();
        self.hard_deadline = movetime.map(|d| start + d);
        let max_depth = if depth == 0 { MAX_PLY as u32 } else { depth };
        if self.can_use_smp(threads) && self.node_limit == u64::MAX {
            let mut r = self.with_smp(board, threads, |main| main.iterate(board, max_depth, soft, start));
            r.nodes = self.nodes;
            r
        } else {
            self.iterate(board, max_depth, soft, start)
        }
    }

    pub fn search_with_params(&mut self, board: &Board, params: SearchParams) -> SearchResult {
        let threads = params.threads.max(1);
        self.set_threads(threads);
        self.use_tt = params.use_tt;
        self.use_lmr = params.use_lmr;
        self.use_nullmove = params.use_nullmove;
        self.use_aspiration = params.use_aspiration;
        self.order_captures = params.order_captures;
        self.use_history = params.use_history;
        self.use_killers = params.use_killers;
        self.deterministic = params.deterministic;
        self.node_limit = params.max_nodes.unwrap_or(u64::MAX);
        if !params.use_tt {
            self.tt = Arc::new(PackedTt::with_mb(1));
        }
        // The front end's allocation is the hard limit; aim to finish well
        // inside it and let move stability stretch or shrink the target.
        let soft = if params.fixed_movetime {
            None
        } else {
            params.movetime.map(|d| d.mul_f64(tp(T::TmSoft) as f64 / 100.0))
        };
        let r = self.run(board, params.depth, params.movetime, soft, threads);
        self.node_limit = u64::MAX;
        r
    }

    pub fn search_movetime(&mut self, board: &Board, millis: u64, depth: u32) -> (Option<String>, i32, u64) {
        self.node_limit = u64::MAX;
        let t = Duration::from_millis(millis);
        let threads = self.threads;
        let r = self.run(board, depth, Some(t), Some(t.mul_f64(0.55)), threads);
        (r.bestmove, r.score_cp, r.nodes)
    }

    pub fn search_movetime_lazy_smp(&mut self, board: &Board, millis: u64, depth: u32) -> (Option<String>, i32, u64) {
        self.search_movetime(board, millis, depth)
    }

    pub fn search_depth(&mut self, board: &Board, depth: u32) -> SearchResult {
        self.node_limit = u64::MAX;
        let threads = self.threads;
        self.run(board, depth.max(1), None, None, threads)
    }

    pub fn hashfull(&self) -> usize {
        self.tt.hashfull()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mv(board: &Board, uci: &str) -> Move {
        let mut found = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{m}") == uci {
                    found = Some(m);
                }
            }
            false
        });
        found.expect("legal move")
    }

    #[test]
    fn see_threshold_matches_simple_exchanges() {
        // Pawn takes an undefended knight.
        let b = Board::from_fen("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", false).unwrap();
        let m = mv(&b, "e4d5");
        assert!(see_ge(&b, m, 0));
        assert!(see_ge(&b, m, 320));
        assert!(!see_ge(&b, m, 321));
        // Queen takes a pawn defended by a pawn: loses the queen.
        let b = Board::from_fen("4k3/2p5/3p4/8/8/8/3Q4/4K3 w - - 0 1", false).unwrap();
        let m = mv(&b, "d2d6");
        assert!(!see_ge(&b, m, 0));
        assert!(see_ge(&b, m, 100 - 950));
    }

    #[test]
    fn see_handles_xray_recaptures() {
        // Rook takes a pawn defended by a rook, with our second rook behind.
        let b = Board::from_fen("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", false).unwrap();
        let m = mv(&b, "d2d5");
        assert!(see_ge(&b, m, 100));
        assert!(!see_ge(&b, m, 101));
    }

    #[test]
    fn tt_round_trips_scores_moves_and_bounds() {
        let tt = PackedTt::with_mb(1);
        let b = Board::default();
        let m = mv(&b, "e2e4");
        tt.store(b.hash(), Some(m), -1234, 56, 7, Some(Bound::Lower), true);
        let e = tt.probe(b.hash()).unwrap();
        assert_eq!(e.mv, Some(m));
        assert_eq!(e.score, -1234);
        assert_eq!(e.eval, 56);
        assert_eq!(e.depth, 7);
        assert_eq!(e.bound, Some(Bound::Lower));
        assert!(e.pv);
        assert!(tt.probe(b.hash() ^ 1).is_none());
    }

    #[test]
    fn mate_scores_are_ply_relative_in_the_tt() {
        let stored = score_to_tt(MATE_SCORE - 7, 3);
        assert_eq!(score_from_tt(stored, 8, 0), MATE_SCORE - 12);
        let stored = score_to_tt(-MATE_SCORE + 7, 3);
        assert_eq!(score_from_tt(stored, 8, 0), -MATE_SCORE + 12);
        assert_eq!(score_from_tt(score_to_tt(173, 4), 29, 0), 173);
    }

    #[test]
    fn finds_mate_in_one() {
        let b = Board::from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", false).unwrap();
        let mut s = Searcher::default();
        let r = s.search_depth(&b, 4);
        assert_eq!(r.bestmove.as_deref(), Some("a1a8"));
        assert_eq!(r.score_cp, MATE_SCORE - 1);
    }

    #[test]
    fn in_tree_repetition_is_a_draw_but_not_across_a_null_move() {
        let mut s = Searcher::default();
        let b = Board::from_fen("4k3/8/8/8/8/8/7Q/4K3 w - - 8 5", false).unwrap();
        s.prepare_root(&b);
        let root = b.hash();
        s.hash_hist.extend_from_slice(&[11, 12, 13, root]);
        assert!(s.is_draw(&b));
        s.null_floor = s.hash_hist.len() - 2;
        assert!(!s.is_draw(&b));
    }

    #[test]
    fn game_history_before_the_root_needs_a_threefold() {
        let mut s = Searcher::default();
        let b = Board::from_fen("4k3/8/8/8/8/8/7Q/4K3 w - - 20 15", false).unwrap();
        let k = b.hash();
        // One occurrence before the root (index 0), the root at index 4,
        // then the line returns to k: only two occurrences, both before the
        // root's own subtree, so no draw yet.
        s.hash_hist = vec![k, 1, 2, 3, 99, 5, 6, 7, k];
        s.root_index = 4;
        assert!(!s.is_draw(&b));
        // Two earlier occurrences make it a threefold.
        s.hash_hist = vec![k, 1, 2, 3, k, 4, 5, 6, 99, 7, 8, 9, k];
        s.root_index = 8;
        assert!(s.is_draw(&b));
    }
}

#[cfg(test)]
mod contempt_tests {
    use super::*;

    #[test]
    fn contempt_favours_the_root_side_at_even_plies() {
        let s = Searcher::default();
        CONTEMPT.store(30, Ordering::Relaxed);
        assert_eq!(s.draw_value(0), 30);
        assert_eq!(s.draw_value(1), -30);
        assert_eq!(s.draw_value(4), 30);
        CONTEMPT.store(0, Ordering::Relaxed);
        assert_eq!(s.draw_value(3), 0);
    }
}
