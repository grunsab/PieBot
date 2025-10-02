use cozy_chess::{Board, Color, Move, Piece};
pub mod accumulator;
pub mod features;
pub mod loader;
pub mod network;
pub mod quant;
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub struct NnueMeta {
    pub version: u32,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

pub struct Nnue {
    pub meta: NnueMeta,
    // Placeholder weights; real implementation will store quantized layers
    w1: Vec<f32>, // hidden_dim x input_dim
    b1: Vec<f32>, // hidden_dim
    w2: Vec<f32>, // output_dim x hidden_dim (output_dim=1)
    b2: Vec<f32>, // output_dim
}

impl Nnue {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Format:
        // magic: 8 bytes b"PIENNUE1"
        // u32 version (LE)
        // u32 input_dim, u32 hidden_dim, u32 output_dim (LE)
        // f32 w1[hidden_dim * input_dim]
        // f32 b1[hidden_dim]
        // f32 w2[output_dim * hidden_dim]
        // f32 b2[output_dim]
        let f = File::open(&path)
            .with_context(|| format!("open nnue file: {}", path.as_ref().display()))?;
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).context("read magic")?;
        if &magic != b"PIENNUE1" {
            bail!("bad NNUE magic");
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4).context("read version")?;
        let version = u32::from_le_bytes(buf4);
        r.read_exact(&mut buf4).context("read input_dim")?;
        let input_dim = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).context("read hidden_dim")?;
        let hidden_dim = u32::from_le_bytes(buf4) as usize;
        r.read_exact(&mut buf4).context("read output_dim")?;
        let output_dim = u32::from_le_bytes(buf4) as usize;
        let mut read_f32s = |n: usize| -> Result<Vec<f32>> {
            // Lenient reader: if file ends early, pad with zeros.
            use std::io::Read as _;
            let mut buf = vec![0u8; n * 4];
            let mut off = 0usize;
            while off < buf.len() {
                match r.read(&mut buf[off..]) {
                    Ok(0) => break, // EOF
                    Ok(k) => off += k,
                    Err(e) => return Err(e).with_context(|| format!("read {} f32s", n)),
                }
            }
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let mut b = [0u8; 4];
                b.copy_from_slice(&buf[i * 4..i * 4 + 4]);
                out.push(f32::from_le_bytes(b));
            }
            Ok(out)
        };
        let w1 = read_f32s(hidden_dim * input_dim)?;
        let b1 = read_f32s(hidden_dim)?;
        let w2 = read_f32s(output_dim * hidden_dim)?;
        let b2 = read_f32s(output_dim)?;
        Ok(Self {
            meta: NnueMeta {
                version,
                input_dim,
                hidden_dim,
                output_dim,
            },
            w1,
            b1,
            w2,
            b2,
        })
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let x = self.features(board);
        let n = self.meta.input_dim;
        let h = self.meta.hidden_dim;
        let mut y1 = vec![0f32; h];
        for j in 0..h {
            let mut sum = self.b1[j];
            let row = &self.w1[j * n..(j + 1) * n];
            for i in 0..n {
                sum += row[i] * x[i];
            }
            y1[j] = if sum > 0.0 { sum } else { 0.0 };
        }
        let mut out = 0f32;
        if self.meta.output_dim > 0 {
            let row = &self.w2[0..h];
            let mut sum = self.b2[0];
            for j in 0..h {
                sum += row[j] * y1[j];
            }
            out = sum;
        }
        out.round() as i32
    }

    pub fn refresh_accumulator(&mut self, _board: &Board) {
        // Stub: full recompute of accumulator
    }

    pub fn update_on_move(&mut self, _mv: Move) {
        // Stub: incremental update of accumulator
    }

    fn features(&self, board: &Board) -> Vec<f32> {
        let n = self.meta.input_dim;
        if n == 12 {
            let kinds = [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ];
            let mut out = vec![0f32; 12];
            for (i, p) in kinds.iter().enumerate() {
                out[i] = (board.pieces(*p) & board.colors(Color::White))
                    .into_iter()
                    .count() as f32;
                out[6 + i] = (board.pieces(*p) & board.colors(Color::Black))
                    .into_iter()
                    .count() as f32;
            }
            return out;
        }
        vec![0.0; n]
    }
}
