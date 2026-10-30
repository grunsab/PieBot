//! Shared pieces of the bullet trainer for PieBot arch-v3 (PIENNQ03) nets:
//! the king-bucket layouts, the binpack filter and score mapping, a binpack
//! loader that interleaves many files, a reference feature map (mirrors
//! `training/nnue/train_v3.py::piece_features`) and the PIENNQ03 exporter.

use std::{
    fs::File,
    io::{self, BufReader, Write},
    path::Path,
    sync::mpsc,
    thread,
};

use bullet::game::{
    formats::bulletformat::ChessBoard,
    inputs::{ChessBucketsMirrored, SparseInputType},
};
use bullet_trainer::reader::DataReader;
use sfbinpack::{
    ChunkReader, TrainingDataEntry,
    chess::{color::Color, piece::Piece, piecetype::PieceType, r#move::MoveType},
    read_chunk_into,
};

pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const EVAL_SCALE: i32 = 400;
pub const OUTPUT_BUCKETS: usize = 8;
pub const MAGIC: &[u8; 8] = b"PIENNQ03";
pub const FILE_VERSION_SINGLE: u32 = 1;
/// Legacy i8 head with the material-experts header (train_v3.FILE_VERSION_EXPERTS).
pub const FILE_VERSION_EXPERTS: u32 = 3;
/// Linear head with i16 output weights at `QB_PRECISE` and i64 biases (see
/// train_v3.FILE_VERSION_PRECISE); the default export.
pub const FILE_VERSION_PRECISE: u32 = 4;
pub const QB_PRECISE: i32 = 4096;
/// Pairwise hidden stack (train_v3.FILE_VERSION_PAIRWISE): per perspective
/// x[i] = clamp(a[i]) * clamp(a[i + h/2]) for i < h/2, then L1 (i8 at
/// `L1_QB`, crelu) -> L2 (f32, crelu) -> 1 (f32), each with 8 output buckets.
pub const FILE_VERSION_PAIRWISE: u32 = 5;
/// Version 5 plus threat inputs: u32 n_threat_features, then i8
/// threat_w[n][hidden] (x QA) after the hidden stack.
pub const FILE_VERSION_THREATS: u32 = 6;
pub use piebot::eval::nnue::threats::{MAX_ACTIVE as MAX_THREATS, N_THREAT_FEATURES};
pub const L1_QB: i32 = 64;

/// 16 buckets, indexed by rank * 4 + file for the mirrored half (files a-d).
/// Identical to `KING_BUCKET_LAYOUT` in train_v3.py.
#[rustfmt::skip]
pub const LAYOUT_16: [usize; 32] = [
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 8, 9, 9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

pub const fn layout_32() -> [usize; 32] {
    let mut l = [0; 32];
    let mut i = 0;
    while i < 32 {
        l[i] = i;
        i += 1;
    }
    l
}

pub fn layout(buckets: usize) -> [usize; 32] {
    match buckets {
        16 => LAYOUT_16,
        32 => layout_32(),
        _ => panic!("king buckets must be 16 or 32"),
    }
}

pub fn num_buckets(layout: &[usize; 32]) -> usize {
    layout.iter().max().unwrap() + 1
}

// ---------------------------------------------------------------------------
// Material experts
// ---------------------------------------------------------------------------

/// Default cuts for 4 experts: the lowest piece count of experts 0..2, so
/// expert 0 takes >= 26 pieces, 1 takes 17-25, 2 takes 11-16 and 3 the rest.
/// Close to equal-frequency on LCZero data (27/25/22/26%).
pub const DEFAULT_EXPERT_CUTS: [usize; 3] = [26, 17, 11];

/// Piece count (0..=32) -> expert, as train_v3.experts_table(n, cuts):
/// expert i is the first with pieces >= cuts[i], else n - 1.
pub fn experts_table(n: usize, cuts: &[usize]) -> [u8; 33] {
    let mut t = [0u8; 33];
    match n {
        1 => {}
        4 => {
            assert!(cuts.len() == 3 && cuts.windows(2).all(|w| w[0] > w[1]), "4 experts need 3 descending cuts");
            for (p, e) in t.iter_mut().enumerate() {
                *e = cuts.iter().position(|&c| p >= c).unwrap_or(3) as u8;
            }
        }
        _ => panic!("experts must be 1 or 4"),
    }
    t
}

/// The network's shape, written next to the checkpoints (`piebot_net.txt`)
/// so `export` can rebuild it without repeating the training options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetMeta {
    pub buckets: usize,
    pub hidden: usize,
    pub experts: usize,
    pub cuts: Vec<usize>,
    /// Pairwise hidden stack widths (l1, l2); None for the linear head.
    pub stack: Option<(usize, usize)>,
    /// Threat inputs (file version 6; needs the pairwise stack).
    pub threats: bool,
}

impl NetMeta {
    pub const FILE: &'static str = "piebot_net.txt";

    pub fn table(&self) -> [u8; 33] {
        experts_table(self.experts, &self.cuts)
    }

    pub fn to_text(&self) -> String {
        let cuts: Vec<String> = self.cuts.iter().map(|c| c.to_string()).collect();
        let mut t = format!("buckets={}\nhidden={}\nexperts={}\ncuts={}\n", self.buckets, self.hidden, self.experts, cuts.join(","));
        if let Some((l1, l2)) = self.stack {
            t += &format!("pairwise_l1={l1}\npairwise_l2={l2}\n");
        }
        if self.threats {
            t += "threats=1\n";
        }
        t
    }

    pub fn parse(text: &str) -> Option<Self> {
        let mut m = std::collections::HashMap::new();
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                m.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
        let num = |k: &str| m.get(k).and_then(|v| v.parse().ok());
        let cuts = m.get("cuts").map(|v| parse_cuts(v)).unwrap_or_default();
        let stack = match (num("pairwise_l1"), num("pairwise_l2")) {
            (Some(l1), Some(l2)) => Some((l1, l2)),
            _ => None,
        };
        let threats = m.get("threats").is_some_and(|v| v == "1");
        Some(Self { buckets: num("buckets")?, hidden: num("hidden")?, experts: num("experts")?, cuts, stack, threats })
    }
}

pub fn parse_cuts(s: &str) -> Vec<usize> {
    s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().expect("--expert-cuts: integers")).collect()
}

/// King-bucketed, mirrored inputs (bullet's `ChessBucketsMirrored`), with the
/// whole 768 x buckets block repeated per material expert: a position's
/// features live in the block of its expert, chosen by piece count. With one
/// expert this is exactly `ChessBucketsMirrored`.
#[derive(Clone, Copy, Debug)]
pub struct ChessBucketsMirroredExperts {
    inner: ChessBucketsMirrored,
    block: usize,
    experts: usize,
    table: [u8; 33],
}

impl ChessBucketsMirroredExperts {
    pub fn new(layout: [usize; 32], experts: usize, table: [u8; 33]) -> Self {
        Self { inner: ChessBucketsMirrored::new(layout), block: 768 * num_buckets(&layout), experts, table }
    }
}

impl SparseInputType for ChessBucketsMirroredExperts {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        self.block * self.experts
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let off = self.block * usize::from(self.table[pos.occ().count_ones() as usize]);
        self.inner.map_features(pos, |s, n| f(off + s, off + n));
    }

    fn shorthand(&self) -> String {
        format!("{}x{}e", self.inner.shorthand(), self.experts)
    }

    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt inputs, one block per material expert".to_string()
    }
}

/// Threat features (PIENNQ03 version 6) of a position for the side to move
/// and the other side, sorted: the engine's own map
/// (`piebot::eval::nnue::threats`), fed the board as bullet stores it (side
/// to move as White, ranks flipped when Black is to move). Threat features
/// only use relative colours and oriented squares, so this equals the
/// engine's features on the real board.
pub fn threat_features(pos: &ChessBoard) -> (Vec<u32>, Vec<u32>) {
    let mut pieces = [[0u64; 6]; 2];
    for (piece, square) in pos.into_iter() {
        pieces[usize::from(piece & 8 > 0)][usize::from(piece & 7)] |= 1u64 << square;
    }
    use piebot::eval::nnue::threats::features_of_pieces;
    (features_of_pieces(pieces, 0), features_of_pieces(pieces, 1))
}

// ---------------------------------------------------------------------------
// Scores
// ---------------------------------------------------------------------------

/// Stockfish's VALUE_NONE: minimised binpacks mark positions to skip with it.
pub const SKIP_SCORE: i16 = 32002;

/// The BT4 relabeller's mapping from LCZero Q to the binpack score
/// (xu-shawn/lc0 `QToCentipawns`): cp = 660.6 q / (1 - 0.9751875 q^10).
pub fn q_to_binpack_cp(q: f64) -> f64 {
    (660.6 * q / (1.0 - 0.9751875 * q.powi(10))).clamp(-32000.0, 32000.0)
}

/// Inverse of `q_to_binpack_cp` (monotonic on [0, 1)), by bisection.
pub fn binpack_cp_to_q(cp: f64) -> f64 {
    let s = cp.signum();
    let target = cp.abs().min(q_to_binpack_cp(0.999_999_9));
    let (mut lo, mut hi) = (0.0f64, 0.999_999_9f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if q_to_binpack_cp(mid) < target { lo = mid } else { hi = mid }
    }
    s * 0.5 * (lo + hi)
}

/// How a binpack score becomes the training score that bullet feeds through
/// `sigmoid(score / 400)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreMode {
    /// Undo the relabeller's mapping to recover Q, then use
    /// 400 * ln((1 + q) / (1 - q)), so that sigmoid(score / 400) = (1 + q) / 2:
    /// the same eval target `train_v3.py` uses on LCZero data.
    Q,
    /// Use the binpack score as is.
    Raw,
}

/// Lookup table from |binpack score| (0..=32000) to the training score.
#[derive(Clone)]
pub struct ScoreMap {
    table: Vec<i16>,
    mode: ScoreMode,
}

impl ScoreMap {
    pub fn new(mode: ScoreMode) -> Self {
        let table = (0..=32000)
            .map(|cp| {
                let q = binpack_cp_to_q(f64::from(cp)).min(0.999_999);
                let s = f64::from(EVAL_SCALE) * ((1.0 + q) / (1.0 - q)).ln();
                s.round().clamp(-32000.0, 32000.0) as i16
            })
            .collect();
        Self { table, mode }
    }

    pub fn map(&self, score: i16) -> i16 {
        match self.mode {
            ScoreMode::Raw => score,
            ScoreMode::Q => {
                let a = self.table[usize::from(score.unsigned_abs().min(32000))];
                if score < 0 { -a } else { a }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Filter {
    pub min_ply: u16,
    pub skip_check: bool,
    pub skip_capture: bool,
}

impl Default for Filter {
    fn default() -> Self {
        // Same as lc0_to_v3bin.py: ply >= 8, not in check, best move not a capture.
        Self { min_ply: 8, skip_check: true, skip_capture: true }
    }
}

pub fn is_capture(e: &TrainingDataEntry) -> bool {
    match e.mv.mtype() {
        MoveType::EnPassant => true,
        // Castling is encoded king-takes-rook; never a capture.
        MoveType::Castle => false,
        _ => e.pos.piece_at(e.mv.to()) != Piece::NONE,
    }
}

impl Filter {
    pub fn keep(&self, e: &TrainingDataEntry) -> bool {
        e.score != SKIP_SCORE
            && e.score.unsigned_abs() <= 32000
            && e.ply >= self.min_ply
            && !(self.skip_check && e.pos.is_checked(e.pos.side_to_move()))
            && !(self.skip_capture && is_capture(e))
    }
}

/// Binpack entry -> bulletformat board (side-to-move relative), with the
/// score mapped by `scores`. Mirrors bullet's own sfbinpack conversion.
pub fn to_board(e: &TrainingDataEntry, scores: &ScoreMap) -> ChessBoard {
    let mut bbs = [0u64; 8];
    let stm = usize::from(e.pos.side_to_move().ordinal());
    let pc_bb = |pt| e.pos.pieces_bb_color(Color::Black, pt).bits() | e.pos.pieces_bb_color(Color::White, pt).bits();
    bbs[0] = e.pos.pieces_bb(Color::White).bits();
    bbs[1] = e.pos.pieces_bb(Color::Black).bits();
    bbs[2] = pc_bb(PieceType::Pawn);
    bbs[3] = pc_bb(PieceType::Knight);
    bbs[4] = pc_bb(PieceType::Bishop);
    bbs[5] = pc_bb(PieceType::Rook);
    bbs[6] = pc_bb(PieceType::Queen);
    bbs[7] = pc_bb(PieceType::King);
    // Binpack score and result are side-to-move relative; from_raw wants white relative.
    let mut score = scores.map(e.score);
    let mut result = f32::from(1 + e.result) / 2.0;
    if stm > 0 {
        score = -score;
        result = 1.0 - result;
    }
    ChessBoard::from_raw(bbs, stm, score, result).expect("malformed binpack entry")
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// Streams many binpacks at once: each chunk (about 1 MB of whole games) is
/// read from a file picked at random in proportion to its size, so every
/// shuffle buffer mixes all datasets. A file that ends or is truncated
/// mid-chunk is reopened from the start.
#[derive(Clone)]
pub struct InterleavedBinpackLoader {
    pub files: Vec<String>,
    pub buffer_positions: usize,
    pub threads: usize,
    pub filter: Filter,
    pub scores: ScoreMap,
    pub seed: u64,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

impl DataReader<ChessBoard> for InterleavedBinpackLoader {
    fn read_chunks<F: FnMut(&[ChessBoard]) -> bool>(&self, _skip: usize, mut f: F) {
        let files = self.files.clone();
        let sizes: Vec<u64> = files.iter().map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)).collect();
        let total: u64 = sizes.iter().sum();
        assert!(total > 0, "no binpack data");
        let threads = self.threads.max(1);
        let group = threads * 4;
        let seed = self.seed;

        let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<Vec<u8>>>(4);
        thread::spawn(move || {
            let mut rng = Rng(seed);
            let mut readers: Vec<Option<BufReader<File>>> = files.iter().map(|_| None).collect();
            let mut batch = Vec::with_capacity(group);
            loop {
                let mut pick = rng.below(total);
                let mut i = 0;
                while pick >= sizes[i] {
                    pick -= sizes[i];
                    i += 1;
                }
                let mut chunk = Vec::new();
                let mut ok = false;
                for _attempt in 0..2 {
                    let r = readers[i].get_or_insert_with(|| BufReader::new(File::open(&files[i]).unwrap()));
                    if let Ok(true) = read_chunk_into(r, &mut chunk) {
                        ok = true;
                        break;
                    }
                    // End of file, or a truncated last chunk: start the file again.
                    readers[i] = None;
                }
                if !ok {
                    continue;
                }
                batch.push(chunk);
                if batch.len() == group && chunk_tx.send(std::mem::take(&mut batch)).is_err() {
                    return;
                }
            }
        });

        let (conv_tx, conv_rx) = mpsc::sync_channel::<Vec<ChessBoard>>(4 * threads);
        let filter = self.filter;
        let scores = self.scores.clone();
        thread::spawn(move || {
            while let Ok(chunks) = chunk_rx.recv() {
                let per = chunks.len().div_ceil(threads);
                let stop = thread::scope(|s| {
                    let handles: Vec<_> = chunks
                        .chunks(per)
                        .map(|grp| {
                            let tx = conv_tx.clone();
                            let scores = &scores;
                            s.spawn(move || {
                                let mut out = Vec::new();
                                for chunk in grp {
                                    let mut r = ChunkReader::default();
                                    while r.has_next(chunk) {
                                        let e = r.next(chunk);
                                        if filter.keep(&e) {
                                            out.push(to_board(&e, scores));
                                        }
                                    }
                                }
                                tx.send(out).is_err()
                            })
                        })
                        .collect();
                    handles.into_iter().any(|h| h.join().unwrap())
                });
                if stop {
                    return;
                }
            }
        });

        let buffer_positions = self.buffer_positions;
        let mut rng = Rng(seed ^ 0xA5A5_5A5A);
        let mut buf = Vec::with_capacity(buffer_positions);
        while let Ok(conv) = conv_rx.recv() {
            for b in conv {
                buf.push(b);
                if buf.len() == buffer_positions {
                    for i in (1..buf.len()).rev() {
                        let j = rng.below(i as u64 + 1) as usize;
                        buf.swap(i, j);
                    }
                    if f(&buf) {
                        return;
                    }
                    buf.clear();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reference features (train_v3.piece_features), for parity checks
// ---------------------------------------------------------------------------

/// Feature indices for (white, black) perspectives from (square, code)
/// pairs, code = color * 6 + piece_type (white 0..5, black 6..11, P N B R Q K).
pub fn reference_features(layout: &[usize; 32], pieces: &[(usize, usize)]) -> [Vec<usize>; 2] {
    reference_features_experts(layout, &[0; 33], pieces)
}

/// As `reference_features`, with train_v3's expert offset:
/// bucket index = expert * buckets + king bucket.
pub fn reference_features_experts(layout: &[usize; 32], table: &[u8; 33], pieces: &[(usize, usize)]) -> [Vec<usize>; 2] {
    let expert = usize::from(table[pieces.len()]);
    let nb = num_buckets(layout);
    let king = |code| pieces.iter().find(|&&(_, c)| c == code).unwrap().0;
    let (kw, kb) = (king(5), king(11));
    let mut out = [Vec::new(), Vec::new()];
    for persp in 0..2 {
        let mut k = if persp == 0 { kw } else { kb ^ 56 };
        let mirror = (k & 7) >= 4;
        if mirror {
            k ^= 7;
        }
        let bucket = expert * nb + layout[(k >> 3) * 4 + (k & 7)];
        for &(sq, code) in pieces {
            let (color, ptype) = (code / 6, code % 6);
            let mut s = if persp == 0 { sq } else { sq ^ 56 };
            if mirror {
                s ^= 7;
            }
            let rel = if persp == 0 { color } else { 1 - color };
            out[persp].push(bucket * 768 + (rel * 6 + ptype) * 64 + s);
        }
    }
    out
}

/// (square, code) pairs from a binpack position.
pub fn entry_pieces(e: &TrainingDataEntry) -> Vec<(usize, usize)> {
    let types = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King];
    let mut v = Vec::new();
    for (ci, color) in [Color::White, Color::Black].into_iter().enumerate() {
        for (ti, &pt) in types.iter().enumerate() {
            let mut bb = e.pos.pieces_bb_color(color, pt).bits();
            while bb != 0 {
                v.push((bb.trailing_zeros() as usize, ci * 6 + ti));
                bb &= bb - 1;
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// numpy's np.round: round half to even.
fn round_even(x: f32) -> f32 {
    x.round_ties_even()
}

/// Float weights of a linear arch-v3 net, in train_v3's layout:
/// w1 [features][hidden], b1 [hidden], w2 [8][2 * hidden] (stm half first), b2 [8].
pub struct FloatNet {
    pub buckets: usize,
    pub layout: [usize; 32],
    pub hidden: usize,
    /// Material experts: w1 holds `experts` blocks of buckets * 768 rows.
    pub experts: usize,
    pub expert_table: [u8; 33],
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    /// Pairwise hidden stack; w2/b2 (the unused linear head) are then zero.
    pub stack: Option<PairwiseStack>,
    /// Threat input weights, feature-major `[N_THREAT_FEATURES][hidden]`.
    pub threat_w: Option<Vec<f32>>,
}

/// Float weights of the pairwise hidden stack, in train_v3's layout:
/// h1_w [8 * l1][hidden], h1_b [8 * l1], h2_w [8 * l2][l1], h2_b [8 * l2],
/// h3_w [8][l2], h3_b [8].
pub struct PairwiseStack {
    pub l1: usize,
    pub l2: usize,
    pub h1_w: Vec<f32>,
    pub h1_b: Vec<f32>,
    pub h2_w: Vec<f32>,
    pub h2_b: Vec<f32>,
    pub h3_w: Vec<f32>,
    pub h3_b: Vec<f32>,
}

impl FloatNet {
    /// Read the float `quantised.bin` our save format writes:
    /// l0w (factoriser merged), l0b, l1w (transposed), l1b, then bullet's padding.
    pub fn from_bullet_floats(path: &Path, buckets: usize, hidden: usize) -> io::Result<Self> {
        Self::from_bullet_floats_experts(path, buckets, hidden, 1, [0; 33])
    }

    /// As `from_bullet_floats` for a net with material experts (the saved
    /// l0w already has the shared base and factoriser folded in).
    pub fn from_bullet_floats_experts(path: &Path, buckets: usize, hidden: usize, experts: usize, expert_table: [u8; 33]) -> io::Result<Self> {
        let raw = std::fs::read(path)?;
        let nf = experts * buckets * 768;
        let sizes = [nf * hidden, hidden, OUTPUT_BUCKETS * 2 * hidden, OUTPUT_BUCKETS];
        let need: usize = sizes.iter().sum::<usize>() * 4;
        if raw.len() < need || raw.len() - need >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {} bytes, expected {} (+ <64 padding) for {experts} experts x {buckets} buckets x {hidden}", path.display(), raw.len(), need),
            ));
        }
        let mut off = 0;
        let mut take = |n: usize| {
            let v: Vec<f32> = raw[off..off + 4 * n].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            off += 4 * n;
            v
        };
        let (w1, b1, w2, b2) = (take(sizes[0]), take(sizes[1]), take(sizes[2]), take(sizes[3]));
        Ok(Self { buckets, layout: layout(buckets), hidden, experts, expert_table, w1, b1, w2, b2, stack: None, threat_w: None })
    }

    /// Read a checkpoint's float `quantised.bin` for any shape in `meta`.
    pub fn from_bullet_floats_meta(path: &Path, meta: &NetMeta) -> io::Result<Self> {
        match meta.stack {
            None => Self::from_bullet_floats_experts(path, meta.buckets, meta.hidden, meta.experts, meta.table()),
            Some((l1, l2)) => {
                if meta.experts != 1 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "a pairwise stack takes one expert"));
                }
                Self::from_bullet_floats_pairwise_threats(path, meta.buckets, meta.hidden, l1, l2, meta.threats)
            }
        }
    }

    /// Largest accumulator magnitude any position could reach once quantised,
    /// and the lane where it occurs: per expert/bucket block and lane, the
    /// bias plus the 32 largest same-signed feature weights (a position has
    /// at most 32 pieces). The engine's i16 accumulators wrap silently, so a
    /// result above 32767 means some position could evaluate wrongly.
    pub fn worst_case_accumulator(&self) -> (i64, usize) {
        let q = |x: f32| round_even(x * QA as f32).clamp(-32768.0, 32767.0) as i64;
        let h = self.hidden;
        let mut best = (0i64, 0usize);
        let mut col = Vec::with_capacity(768);
        for block in 0..self.experts * self.buckets {
            for lane in 0..h {
                col.clear();
                col.extend((0..768).map(|f| q(self.w1[(block * 768 + f) * h + lane])));
                col.sort_unstable();
                let bias = q(self.b1[lane]);
                let neg: i64 = col.iter().take(32).filter(|&&v| v < 0).sum();
                let pos: i64 = col.iter().rev().take(32).filter(|&&v| v > 0).sum();
                for v in [(bias + pos).abs(), (bias + neg).abs()] {
                    if v > best.0 {
                        best = (v, lane);
                    }
                }
            }
        }
        best
    }

    /// The pairwise save format: l0w (factoriser merged), l0b, then l1w, l1b,
    /// l2w, l2b, l3w, l3b with each weight transposed to [out][in].
    pub fn from_bullet_floats_pairwise(path: &Path, buckets: usize, hidden: usize, l1: usize, l2: usize) -> io::Result<Self> {
        Self::from_bullet_floats_pairwise_threats(path, buckets, hidden, l1, l2, false)
    }

    /// As `from_bullet_floats_pairwise`, with the threat weights `l0t`
    /// (feature-major) saved after the stack when `threats`.
    pub fn from_bullet_floats_pairwise_threats(
        path: &Path,
        buckets: usize,
        hidden: usize,
        l1: usize,
        l2: usize,
        threats: bool,
    ) -> io::Result<Self> {
        let raw = std::fs::read(path)?;
        let nb = OUTPUT_BUCKETS;
        let nf = buckets * 768;
        let nt = if threats { N_THREAT_FEATURES * hidden } else { 0 };
        let sizes = [nf * hidden, hidden, nb * l1 * hidden, nb * l1, nb * l2 * l1, nb * l2, nb * l2, nb, nt];
        let need: usize = sizes.iter().sum::<usize>() * 4;
        if raw.len() < need || raw.len() - need >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {} bytes, expected {} (+ <64 padding) for a pairwise {buckets} x {hidden} -> {l1} -> {l2} net", path.display(), raw.len(), need),
            ));
        }
        let mut off = 0;
        let mut take = |n: usize| {
            let v: Vec<f32> = raw[off..off + 4 * n].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            off += 4 * n;
            v
        };
        let (w1, b1) = (take(sizes[0]), take(sizes[1]));
        let stack = PairwiseStack {
            l1,
            l2,
            h1_w: take(sizes[2]),
            h1_b: take(sizes[3]),
            h2_w: take(sizes[4]),
            h2_b: take(sizes[5]),
            h3_w: take(sizes[6]),
            h3_b: take(sizes[7]),
        };
        let threat_w = threats.then(|| take(sizes[8]));
        Ok(Self {
            buckets,
            layout: layout(buckets),
            hidden,
            experts: 1,
            expert_table: [0; 33],
            w1,
            b1,
            w2: vec![0.0; nb * 2 * hidden],
            b2: vec![0.0; nb],
            stack: Some(stack),
            threat_w,
        })
    }

    /// Write PIENNQ03 file version 4 (precise head, the default) or 1 (i8
    /// head), byte for byte as train_v3.export does.
    pub fn write_piennq03(&self, path: &Path) -> io::Result<()> {
        self.write_piennq03_version(path, FILE_VERSION_PRECISE)
    }

    pub fn write_piennq03_version(&self, path: &Path, version: u32) -> io::Result<()> {
        let pairwise = version == FILE_VERSION_PAIRWISE || version == FILE_VERSION_THREATS;
        if self.stack.is_some() != pairwise {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "a pairwise stack is written as version 5 or 6, and only it"));
        }
        if self.threat_w.is_some() != (version == FILE_VERSION_THREATS) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "threat weights are written as version 6, and only they"));
        }
        if let Some(tw) = &self.threat_w {
            if tw.len() != N_THREAT_FEATURES * self.hidden {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "threat weights have the wrong size"));
            }
        }
        if pairwise && self.experts != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "experts with a hidden stack are not supported"));
        }
        if version != FILE_VERSION_SINGLE && version != FILE_VERSION_PRECISE && !pairwise {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("cannot write PIENNQ03 version {version}")));
        }
        let precise = version == FILE_VERSION_PRECISE;
        // As train_v3.export: a legacy i8 head with experts is version 3.
        let version = if !precise && self.experts > 1 { FILE_VERSION_EXPERTS } else { version };
        let qb_i = if precise { QB_PRECISE } else { QB };
        let (qa, qb) = (QA as f32, qb_i as f32);
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(self.buckets as u32).to_le_bytes());
        out.extend(self.layout.iter().map(|&b| b as u8));
        if precise || self.experts > 1 {
            out.extend_from_slice(&(self.experts as u32).to_le_bytes());
            out.extend_from_slice(&self.expert_table);
        }
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&(OUTPUT_BUCKETS as u32).to_le_bytes());
        for v in [QA, qb_i, EVAL_SCALE] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &w in &self.w1 {
            out.extend_from_slice(&(round_even(w * qa).clamp(-32768.0, 32767.0) as i16).to_le_bytes());
        }
        for &b in &self.b1 {
            out.extend_from_slice(&(round_even(b * qa).clamp(-32768.0, 32767.0) as i16).to_le_bytes());
        }
        if precise {
            for &w in &self.w2 {
                out.extend_from_slice(&(round_even(w * qb).clamp(-8192.0, 8191.0) as i16).to_le_bytes());
            }
            for &b in &self.b2 {
                // train_v3 widens ob to float64 before scaling.
                let v = (((b as f64 * QA as f64) * QA as f64) * qb_i as f64).round_ties_even();
                out.extend_from_slice(&(v as i64).to_le_bytes());
            }
        } else {
            for &w in &self.w2 {
                out.push(round_even(w * qb).clamp(-128.0, 127.0) as i8 as u8);
            }
            for &b in &self.b2 {
                // numpy computes ob * QA * QA * QB in float32, left to right.
                let v = (((b * qa) * qa) * qb).round_ties_even();
                out.extend_from_slice(&(v as i32).to_le_bytes());
            }
        }
        if let Some(st) = &self.stack {
            // As train_v3.export: u32 l1, u32 l2, i8 h1_w (x L1_QB), then the
            // f32 h1_b, h2_w, h2_b, h3_w, h3_b.
            out.extend_from_slice(&(st.l1 as u32).to_le_bytes());
            out.extend_from_slice(&(st.l2 as u32).to_le_bytes());
            for &w in &st.h1_w {
                out.push(round_even(w * L1_QB as f32).clamp(-128.0, 127.0) as i8 as u8);
            }
            for part in [&st.h1_b, &st.h2_w, &st.h2_b, &st.h3_w, &st.h3_b] {
                for &v in part.iter() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        if let Some(tw) = &self.threat_w {
            // i8 at QA, as threats.write_threat_net expects; training clips
            // these weights to +-127 / QA, so the clamp only catches rounding.
            out.extend_from_slice(&(N_THREAT_FEATURES as u32).to_le_bytes());
            out.extend(tw.iter().map(|&w| round_even(w * qa).clamp(-128.0, 127.0) as i8 as u8));
        }
        let tmp = path.with_extension("tmp");
        File::create(&tmp)?.write_all(&out)?;
        std::fs::rename(tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairwise_net(hidden: usize, threats: bool) -> FloatNet {
        let nb = OUTPUT_BUCKETS;
        let (l1, l2) = (16, 32);
        let f = |n: usize, k: f32| (0..n).map(|i| ((i as f32 * 0.37 + k).sin()) * 0.4).collect::<Vec<f32>>();
        FloatNet {
            buckets: 16, layout: layout(16), hidden, experts: 1, expert_table: [0; 33],
            w1: f(16 * 768 * hidden, 1.0), b1: f(hidden, 2.0),
            w2: vec![0.0; nb * 2 * hidden], b2: vec![0.0; nb],
            stack: Some(PairwiseStack {
                l1, l2, h1_w: f(nb * l1 * hidden, 3.0), h1_b: f(nb * l1, 4.0), h2_w: f(nb * l2 * l1, 5.0),
                h2_b: f(nb * l2, 6.0), h3_w: f(nb * l2, 7.0), h3_b: f(nb, 8.0),
            }),
            threat_w: threats.then(|| f(N_THREAT_FEATURES * hidden, 9.0).into_iter().map(|w| w * 2.0).collect()),
        }
    }

    #[test]
    fn net_meta_round_trips_the_threats_flag() {
        let m = NetMeta { buckets: 16, hidden: 64, experts: 1, cuts: vec![], stack: Some((16, 32)), threats: true };
        let back = NetMeta::parse(&m.to_text()).unwrap();
        assert!(back.threats && back.stack == Some((16, 32)) && back.hidden == 64);
        let plain = NetMeta { threats: false, ..m };
        assert!(!NetMeta::parse(&plain.to_text()).unwrap().threats);
    }

    #[test]
    fn version_6_is_version_5_plus_the_i8_threat_block() {
        let dir = std::env::temp_dir().join(format!("piebot_v6_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let with = pairwise_net(16, true);
        with.write_piennq03_version(&dir.join("v6.nnue"), FILE_VERSION_THREATS).unwrap();
        let mut without = pairwise_net(16, false);
        without.write_piennq03_version(&dir.join("v5.nnue"), FILE_VERSION_PAIRWISE).unwrap();
        let (v6, v5) = (std::fs::read(dir.join("v6.nnue")).unwrap(), std::fs::read(dir.join("v5.nnue")).unwrap());
        assert_eq!(&v6[..8], &v5[..8]);
        assert_eq!(u32::from_le_bytes(v6[8..12].try_into().unwrap()), 6);
        assert_eq!(&v6[12..v5.len()], &v5[12..]);
        let tail = &v6[v5.len()..];
        assert_eq!(u32::from_le_bytes(tail[..4].try_into().unwrap()) as usize, N_THREAT_FEATURES);
        let tw = with.threat_w.as_ref().unwrap();
        assert_eq!(tail.len() - 4, tw.len());
        for (i, (&b, &w)) in tail[4..].iter().zip(tw).enumerate().step_by(997) {
            assert_eq!(b as i8, round_even(w * QA as f32).clamp(-128.0, 127.0) as i8, "weight {i}");
        }
        // A plain net can't be written as version 6, nor a threats net as 5.
        assert!(without.write_piennq03_version(&dir.join("x.nnue"), FILE_VERSION_THREATS).is_err());
        without.threat_w = Some(vec![0.0; N_THREAT_FEATURES * 16]);
        assert!(without.write_piennq03_version(&dir.join("x.nnue"), FILE_VERSION_PAIRWISE).is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_threats_checkpoint_carries_the_threat_rows_after_the_stack() {
        let dir = std::env::temp_dir().join(format!("piebot_tq_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let net = pairwise_net(16, true);
        let st = net.stack.as_ref().unwrap();
        let mut raw = Vec::new();
        for part in [&net.w1, &net.b1, &st.h1_w, &st.h1_b, &st.h2_w, &st.h2_b, &st.h3_w, &st.h3_b, net.threat_w.as_ref().unwrap()] {
            for &v in part.iter() {
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(dir.join("q.bin"), &raw).unwrap();
        let m = NetMeta { buckets: 16, hidden: 16, experts: 1, cuts: vec![], stack: Some((16, 32)), threats: true };
        let back = FloatNet::from_bullet_floats_meta(&dir.join("q.bin"), &m).unwrap();
        assert_eq!(back.threat_w.as_ref().unwrap(), net.threat_w.as_ref().unwrap());
        assert_eq!(back.stack.as_ref().unwrap().h3_b, st.h3_b);
        let plain = NetMeta { threats: false, ..m };
        assert!(FloatNet::from_bullet_floats_meta(&dir.join("q.bin"), &plain).is_err(), "extra floats must not be ignored");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn threat_features_match_the_engine_from_both_sides() {
        use piebot::eval::nnue::threats;
        for fen in [
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
            "1Q6/7k/2q5/8/3BN3/2R1b3/5n2/r3K3 w - - 0 40",
            "6k1/5ppp/8/3Nn3/8/2B5/5PPP/2R3K1 b - - 0 30",
        ] {
            let pos: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();
            let board = cozy_chess::Board::from_fen(fen, false).unwrap();
            let (stm_side, ntm_side) = if board.side_to_move() == cozy_chess::Color::White { (0, 1) } else { (1, 0) };
            let (stm, ntm) = threat_features(&pos);
            assert_eq!(stm, threats::features(&board, stm_side), "{fen} stm");
            assert_eq!(ntm, threats::features(&board, ntm_side), "{fen} ntm");
            assert!(!stm.is_empty());
        }
    }

    #[test]
    fn worst_case_accumulator_bounds_32_pieces_and_the_bias() {
        let hidden = 8;
        let nf = 16 * 768;
        let mut net = FloatNet {
            buckets: 16, layout: layout(16), hidden, experts: 1, expert_table: [0; 33],
            w1: vec![0.0; nf * hidden], b1: vec![0.0; hidden],
            w2: vec![0.0; OUTPUT_BUCKETS * 2 * hidden], b2: vec![0.0; OUTPUT_BUCKETS], stack: None, threat_w: None,
        };
        // Lane 3 of bucket 5: 40 features at +1000 (quantised), only 32 can be active.
        for f in 0..40 {
            net.w1[(5 * 768 + f) * hidden + 3] = 1000.0 / QA as f32;
        }
        // Lane 6 of bucket 0: 32 features at -900.
        for f in 100..132 {
            net.w1[f * hidden + 6] = -900.0 / QA as f32;
        }
        assert_eq!(net.worst_case_accumulator(), (32_000, 3));
        net.b1[6] = -4000.0 / QA as f32;
        assert_eq!(net.worst_case_accumulator(), (32 * 900 + 4000, 6));
    }

    #[test]
    fn q_mapping_round_trips() {
        for &q in &[-0.999, -0.7, -0.2, 0.0, 0.05, 0.5, 0.9, 0.99] {
            let cp = q_to_binpack_cp(q);
            assert!((binpack_cp_to_q(cp) - q).abs() < 1e-9, "{q}");
        }
    }

    #[test]
    fn q_score_targets_match_train_v3() {
        // sigmoid(score / 400) must equal (1 + q) / 2, train_v3's eval target.
        let m = ScoreMap::new(ScoreMode::Q);
        for &q in &[-0.9, -0.3, 0.0, 0.25, 0.8] {
            let cp = q_to_binpack_cp(q).round() as i16;
            let s = f64::from(m.map(cp));
            let p = 1.0 / (1.0 + (-s / 400.0).exp());
            let want = (1.0 + binpack_cp_to_q(f64::from(cp))) / 2.0;
            assert!((p - want).abs() < 2e-3, "q {q}: {p} vs {want}");
        }
        assert_eq!(m.map(-500), -m.map(500));
    }

    #[test]
    fn expert_tables_match_train_v3() {
        let t = experts_table(4, &DEFAULT_EXPERT_CUTS);
        assert_eq!([t[32], t[26], t[25], t[17], t[16], t[11], t[10], t[2]], [0, 0, 1, 1, 2, 2, 3, 3]);
        // train_v3's own default split.
        let t = experts_table(4, &[25, 17, 9]);
        assert_eq!([t[25], t[24], t[17], t[16], t[9], t[8]], [0, 1, 1, 2, 2, 3]);
        assert_eq!(experts_table(1, &[]), [0; 33]);
    }

    #[test]
    fn net_meta_round_trips() {
        let m = NetMeta { buckets: 16, hidden: 1024, experts: 4, cuts: vec![26, 17, 11], stack: None, threats: false };
        assert_eq!(NetMeta::parse(&m.to_text()), Some(m));
        let m = NetMeta { buckets: 16, hidden: 1024, experts: 1, cuts: vec![], stack: Some((16, 32)), threats: false };
        assert_eq!(NetMeta::parse(&m.to_text()), Some(m));
    }

    #[test]
    fn layouts() {
        assert_eq!(num_buckets(&LAYOUT_16), 16);
        assert_eq!(num_buckets(&layout_32()), 32);
    }

    #[test]
    fn rounding_is_half_even_like_numpy() {
        assert_eq!(round_even(2.5), 2.0);
        assert_eq!(round_even(-2.5), -2.0);
        assert_eq!(round_even(3.5), 4.0);
    }
}
