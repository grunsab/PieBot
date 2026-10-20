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

use bullet::game::formats::bulletformat::ChessBoard;
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
/// Linear head with i16 output weights at `QB_PRECISE` and i64 biases (see
/// train_v3.FILE_VERSION_PRECISE); the default export.
pub const FILE_VERSION_PRECISE: u32 = 4;
pub const QB_PRECISE: i32 = 4096;

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
    let king = |code| pieces.iter().find(|&&(_, c)| c == code).unwrap().0;
    let (kw, kb) = (king(5), king(11));
    let mut out = [Vec::new(), Vec::new()];
    for persp in 0..2 {
        let mut k = if persp == 0 { kw } else { kb ^ 56 };
        let mirror = (k & 7) >= 4;
        if mirror {
            k ^= 7;
        }
        let bucket = layout[(k >> 3) * 4 + (k & 7)];
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
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl FloatNet {
    /// Read the float `quantised.bin` our save format writes:
    /// l0w (factoriser merged), l0b, l1w (transposed), l1b, then bullet's padding.
    pub fn from_bullet_floats(path: &Path, buckets: usize, hidden: usize) -> io::Result<Self> {
        let raw = std::fs::read(path)?;
        let nf = buckets * 768;
        let sizes = [nf * hidden, hidden, OUTPUT_BUCKETS * 2 * hidden, OUTPUT_BUCKETS];
        let need: usize = sizes.iter().sum::<usize>() * 4;
        if raw.len() < need || raw.len() - need >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {} bytes, expected {} (+ <64 padding) for {buckets} buckets x {hidden}", path.display(), raw.len(), need),
            ));
        }
        let mut off = 0;
        let mut take = |n: usize| {
            let v: Vec<f32> = raw[off..off + 4 * n].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            off += 4 * n;
            v
        };
        let (w1, b1, w2, b2) = (take(sizes[0]), take(sizes[1]), take(sizes[2]), take(sizes[3]));
        Ok(Self { buckets, layout: layout(buckets), hidden, w1, b1, w2, b2 })
    }

    /// Write PIENNQ03 file version 4 (precise head, the default) or 1 (i8
    /// head), byte for byte as train_v3.export does.
    pub fn write_piennq03(&self, path: &Path) -> io::Result<()> {
        self.write_piennq03_version(path, FILE_VERSION_PRECISE)
    }

    pub fn write_piennq03_version(&self, path: &Path, version: u32) -> io::Result<()> {
        if version != FILE_VERSION_SINGLE && version != FILE_VERSION_PRECISE {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("cannot write PIENNQ03 version {version}")));
        }
        let precise = version == FILE_VERSION_PRECISE;
        let qb_i = if precise { QB_PRECISE } else { QB };
        let (qa, qb) = (QA as f32, qb_i as f32);
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(self.buckets as u32).to_le_bytes());
        out.extend(self.layout.iter().map(|&b| b as u8));
        if precise {
            // One expert: n_experts = 1 and an all-zero piece-count table.
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&[0u8; 33]);
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
        let tmp = path.with_extension("tmp");
        File::create(&tmp)?.write_all(&out)?;
        std::fs::rename(tmp, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
