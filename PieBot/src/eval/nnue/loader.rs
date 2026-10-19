use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct QuantMeta {
    pub version: u32,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone)]
pub struct QuantNnue {
    pub meta: QuantMeta,
    pub w1_scale: f32,
    pub w2_scale: f32,
    pub w1: Vec<i8>,  // hidden x input
    pub b1: Vec<i16>, // hidden
    pub w2: Vec<i8>,  // output x hidden (output=1)
    pub b2: Vec<i16>, // output
    /// Present when the file is a PIENNQ02 dual-perspective model; the legacy
    /// fields above are empty in that case and `meta` mirrors the v2 dims.
    pub v2: Option<std::sync::Arc<QuantNnueV2>>,
    /// Present when the file is a PIENNQ03 king-bucketed model.
    pub v3: Option<std::sync::Arc<QuantNnueV3>>,
}

/// Arch-v3 model (`training/nnue/train_v3.py`): king-bucketed, horizontally
/// mirrored inputs (kings included), shared feature transformer, SCReLU,
/// output heads chosen by piece count.
#[derive(Debug, Clone)]
pub struct QuantNnueV3 {
    pub n_king_buckets: usize,
    /// Bucket for each oriented, mirrored king square: index `rank * 4 + file`.
    pub king_bucket_layout: [u8; 32],
    pub hidden_dim: usize,
    pub n_output_buckets: usize,
    pub qa: i32,
    pub qb: i32,
    pub scale: i32,
    /// Feature-major: row `idx` is `w1[idx * hidden .. (idx + 1) * hidden]`.
    pub w1: Vec<i16>,
    pub b1: Vec<i16>,
    /// Per output bucket, `2 * hidden` weights, side-to-move half first.
    pub w2: Vec<i8>,
    pub b2: Vec<i32>,
    /// File version 2: a hidden stack replaces the linear head.
    pub hidden_stack: Option<HiddenStackV3>,
}

/// Per output bucket: `2h -> l1` in int8 (inputs `(clamp(acc)^2) >> 8`,
/// weights x64), then `l1 -> l2 -> 1` in f32 with clamp(0, 1) activations.
#[derive(Debug, Clone)]
pub struct HiddenStackV3 {
    pub l1: usize,
    pub l2: usize,
    /// `[bucket][l1][2 * hidden]`
    pub h1_w: Vec<i8>,
    pub h1_b: Vec<f32>,
    /// `[bucket][l2][l1]`
    pub h2_w: Vec<f32>,
    pub h2_b: Vec<f32>,
    /// `[bucket][l2]`
    pub h3_w: Vec<f32>,
    pub h3_b: Vec<f32>,
}

impl QuantNnueV3 {
    pub fn n_features(&self) -> usize {
        self.n_king_buckets * 768
    }
}

/// Arch-v2 model: dual-perspective shared feature transformer, SCReLU head,
/// standard integer quantization (QA/QB/SCALE per chessprogramming.org/NNUE).
#[derive(Debug, Clone)]
pub struct QuantNnueV2 {
    pub per_perspective_input_dim: usize,
    pub hidden_dim: usize,
    pub qa: i32,
    pub qb: i32,
    pub scale: i32,
    /// Feature-major on disk and in memory: `w1[idx * hidden .. (idx+1) * hidden]`
    /// is the accumulator row for feature `idx`.
    pub w1: Vec<i16>,
    pub b1: Vec<i16>,
    /// Output weights over the concatenated accumulators, side-to-move half
    /// first: length `2 * hidden_dim`.
    pub w2: Vec<i8>,
    pub b2: i32,
}

const Q_MAGIC: &[u8; 8] = b"PIENNQ01"; // Pie NNUE Quant v1
const Q_MAGIC_V2: &[u8; 8] = b"PIENNQ02"; // Pie NNUE Quant v2 (dual perspective)
const Q_MAGIC_V3: &[u8; 8] = b"PIENNQ03"; // Pie NNUE Quant v3 (king buckets, output buckets)

impl QuantNnue {
    pub fn load_quantized<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Simple quant format for bootstrapping:
        // magic: 8 bytes b"PIENNQ01"
        // u32 version
        // u32 input_dim, u32 hidden_dim, u32 output_dim
        // f32 w1_scale, f32 w2_scale
        // i8  w1[hidden*input]
        // i16 b1[hidden]
        // i8  w2[output*hidden]
        // i16 b2[output]
        let f = File::open(&path)
            .with_context(|| format!("open quant nnue file: {}", path.as_ref().display()))?;
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).context("read magic")?;
        if &magic == Q_MAGIC_V2 {
            return Self::load_quantized_v2(&mut r);
        }
        if &magic == Q_MAGIC_V3 {
            return Self::load_quantized_v3(&mut r);
        }
        if &magic != Q_MAGIC {
            bail!("bad quant NNUE magic");
        }
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4).context("read version")?;
        let version = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("read input_dim")?;
        let input_dim = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4).context("read hidden_dim")?;
        let hidden_dim = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4).context("read output_dim")?;
        let output_dim = u32::from_le_bytes(b4) as usize;
        let mut b4f = [0u8; 4];
        r.read_exact(&mut b4f).context("read w1_scale")?;
        let w1_scale = f32::from_le_bytes(b4f);
        r.read_exact(&mut b4f).context("read w2_scale")?;
        let w2_scale = f32::from_le_bytes(b4f);

        if input_dim == 0 || hidden_dim == 0 {
            bail!(
                "invalid quant dims: input_dim={} hidden_dim={}",
                input_dim,
                hidden_dim
            );
        }
        if output_dim != 1 {
            bail!("unsupported quant output_dim {}; expected 1", output_dim);
        }
        if !w1_scale.is_finite() || w1_scale <= 0.0 {
            bail!("invalid w1_scale {}", w1_scale);
        }
        if !w2_scale.is_finite() || w2_scale <= 0.0 {
            bail!("invalid w2_scale {}", w2_scale);
        }

        let w1_len = hidden_dim
            .checked_mul(input_dim)
            .context("quant dimension overflow: hidden_dim * input_dim")?;
        let w2_len = output_dim
            .checked_mul(hidden_dim)
            .context("quant dimension overflow: output_dim * hidden_dim")?;

        let mut w1_bytes = vec![0u8; w1_len];
        r.read_exact(&mut w1_bytes)
            .context("read quant w1 payload")?;
        let mut b1_bytes = vec![0u8; hidden_dim * 2];
        r.read_exact(&mut b1_bytes)
            .context("read quant b1 payload")?;
        let mut w2_bytes = vec![0u8; w2_len];
        r.read_exact(&mut w2_bytes)
            .context("read quant w2 payload")?;
        let mut b2_bytes = vec![0u8; output_dim * 2];
        r.read_exact(&mut b2_bytes)
            .context("read quant b2 payload")?;

        let w1 = w1_bytes.into_iter().map(|b| b as i8).collect();
        let mut b1 = Vec::with_capacity(hidden_dim);
        for chunk in b1_bytes.chunks_exact(2) {
            b1.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        let w2 = w2_bytes.into_iter().map(|b| b as i8).collect();
        let mut b2 = Vec::with_capacity(output_dim);
        for chunk in b2_bytes.chunks_exact(2) {
            b2.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }

        Ok(Self {
            meta: QuantMeta {
                version,
                input_dim,
                hidden_dim,
                output_dim,
            },
            w1_scale,
            w2_scale,
            w1,
            b1,
            w2,
            b2,
            v2: None,
            v3: None,
        })
    }

    fn load_quantized_v3<R: Read>(r: &mut R) -> Result<Self> {
        // PIENNQ03 layout after the 8-byte magic (all little-endian):
        // u32 version, u32 n_king_buckets, u8 layout[32],
        // u32 hidden_dim, u32 n_output_buckets, i32 qa, i32 qb, i32 scale,
        // i16 w1[n_king_buckets * 768 * hidden] (feature-major), i16 b1[hidden],
        // i8 w2[n_output_buckets * 2 * hidden], i32 b2[n_output_buckets]
        fn u32le<R: Read>(r: &mut R, what: &str) -> Result<u32> {
            let mut b = [0u8; 4];
            r.read_exact(&mut b).with_context(|| format!("read v3 {what}"))?;
            Ok(u32::from_le_bytes(b))
        }
        let version = u32le(r, "version")?;
        let n_king_buckets = u32le(r, "n_king_buckets")? as usize;
        let mut king_bucket_layout = [0u8; 32];
        r.read_exact(&mut king_bucket_layout).context("read v3 bucket layout")?;
        let hidden_dim = u32le(r, "hidden_dim")? as usize;
        let n_output_buckets = u32le(r, "n_output_buckets")? as usize;
        let qa = u32le(r, "qa")? as i32;
        let qb = u32le(r, "qb")? as i32;
        let scale = u32le(r, "scale")? as i32;
        if version != 1 && version != 2 {
            bail!("unsupported PIENNQ03 version {version}");
        }
        if n_king_buckets == 0 || n_king_buckets > 32 || hidden_dim == 0 || hidden_dim > 8192 {
            bail!("invalid v3 dims: buckets={n_king_buckets} hidden={hidden_dim}");
        }
        if king_bucket_layout.iter().any(|&b| b as usize >= n_king_buckets) {
            bail!("v3 bucket layout names a bucket >= {n_king_buckets}");
        }
        if n_output_buckets == 0 || n_output_buckets > 16 {
            bail!("invalid v3 output buckets {n_output_buckets}");
        }
        if qa <= 0 || qa > (i16::MAX as i32) / 128 || qb <= 0 || scale <= 0 {
            bail!("invalid v3 quant scales: qa={qa} qb={qb} scale={scale}");
        }
        let n_features = n_king_buckets * 768;
        let read_i16 = |r: &mut R, n: usize, what: &str| -> Result<Vec<i16>> {
            let mut b = vec![0u8; n * 2];
            r.read_exact(&mut b).with_context(|| format!("read v3 {what}"))?;
            Ok(b.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
        };
        let w1 = read_i16(r, n_features * hidden_dim, "w1")?;
        let b1 = read_i16(r, hidden_dim, "b1")?;
        let mut w2b = vec![0u8; n_output_buckets * 2 * hidden_dim];
        r.read_exact(&mut w2b).context("read v3 w2")?;
        let w2: Vec<i8> = w2b.into_iter().map(|b| b as i8).collect();
        let mut b2b = vec![0u8; n_output_buckets * 4];
        r.read_exact(&mut b2b).context("read v3 b2")?;
        let b2: Vec<i32> = b2b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let hidden_stack = if version == 2 {
            let l1 = u32le(r, "l1")? as usize;
            let l2 = u32le(r, "l2")? as usize;
            if l1 == 0 || l1 > 256 || l2 == 0 || l2 > 256 {
                bail!("invalid v3 hidden stack {l1}x{l2}");
            }
            let nb = n_output_buckets;
            let mut hw = vec![0u8; nb * l1 * 2 * hidden_dim];
            r.read_exact(&mut hw).context("read v3 h1_w")?;
            let read_f32 = |r: &mut R, n: usize, what: &str| -> Result<Vec<f32>> {
                let mut b = vec![0u8; n * 4];
                r.read_exact(&mut b).with_context(|| format!("read v3 {what}"))?;
                Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            };
            Some(HiddenStackV3 {
                l1,
                l2,
                h1_w: hw.into_iter().map(|b| b as i8).collect(),
                h1_b: read_f32(r, nb * l1, "h1_b")?,
                h2_w: read_f32(r, nb * l2 * l1, "h2_w")?,
                h2_b: read_f32(r, nb * l2, "h2_b")?,
                h3_w: read_f32(r, nb * l2, "h3_w")?,
                h3_b: read_f32(r, nb, "h3_b")?,
            })
        } else {
            None
        };
        let mut extra = [0u8; 1];
        if r.read(&mut extra).context("check v3 trailing bytes")? != 0 {
            bail!("trailing bytes after PIENNQ03 payload");
        }
        Ok(Self {
            meta: QuantMeta {
                version,
                input_dim: n_features,
                hidden_dim,
                output_dim: 1,
            },
            w1_scale: 1.0,
            w2_scale: 1.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            v2: None,
            v3: Some(std::sync::Arc::new(QuantNnueV3 {
                n_king_buckets,
                king_bucket_layout,
                hidden_dim,
                n_output_buckets,
                qa,
                qb,
                scale,
                w1,
                b1,
                w2,
                b2,
                hidden_stack,
            })),
        })
    }

    fn load_quantized_v2<R: Read>(r: &mut R) -> Result<Self> {
        // PIENNQ02 layout after the 8-byte magic:
        // u32 version
        // u32 per_perspective_input_dim, u32 hidden_dim, u32 output_dim (=1)
        // i32 qa, i32 qb, i32 scale
        // i16 w1[input * hidden]   (feature-major)
        // i16 b1[hidden]
        // i8  w2[2 * hidden]       (side-to-move half first)
        // i32 b2
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4).context("read v2 version")?;
        let version = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("read v2 input_dim")?;
        let input_dim = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4).context("read v2 hidden_dim")?;
        let hidden_dim = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4).context("read v2 output_dim")?;
        let output_dim = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4).context("read v2 qa")?;
        let qa = i32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("read v2 qb")?;
        let qb = i32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("read v2 scale")?;
        let scale = i32::from_le_bytes(b4);

        if input_dim == 0 || hidden_dim == 0 {
            bail!(
                "invalid v2 quant dims: input_dim={} hidden_dim={}",
                input_dim,
                hidden_dim
            );
        }
        if output_dim != 1 {
            bail!("unsupported v2 quant output_dim {}; expected 1", output_dim);
        }
        if qa <= 0 || qb <= 0 || scale <= 0 {
            bail!("invalid v2 quant scales: qa={qa} qb={qb} scale={scale}");
        }

        let w1_len = input_dim
            .checked_mul(hidden_dim)
            .context("v2 dimension overflow: input_dim * hidden_dim")?;

        let mut w1_bytes = vec![0u8; w1_len * 2];
        r.read_exact(&mut w1_bytes).context("read v2 w1 payload")?;
        let mut b1_bytes = vec![0u8; hidden_dim * 2];
        r.read_exact(&mut b1_bytes).context("read v2 b1 payload")?;
        let mut w2_bytes = vec![0u8; 2 * hidden_dim];
        r.read_exact(&mut w2_bytes).context("read v2 w2 payload")?;
        let mut b2_bytes = [0u8; 4];
        r.read_exact(&mut b2_bytes).context("read v2 b2 payload")?;

        let w1: Vec<i16> = w1_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let b1: Vec<i16> = b1_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        let w2: Vec<i8> = w2_bytes.into_iter().map(|b| b as i8).collect();
        let b2 = i32::from_le_bytes(b2_bytes);

        Ok(Self {
            meta: QuantMeta {
                version,
                input_dim,
                hidden_dim,
                output_dim,
            },
            w1_scale: 1.0,
            w2_scale: 1.0,
            w1: Vec::new(),
            b1: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            v3: None,
            v2: Some(std::sync::Arc::new(QuantNnueV2 {
                per_perspective_input_dim: input_dim,
                hidden_dim,
                qa,
                qb,
                scale,
                w1,
                b1,
                w2,
                b2,
            })),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::QuantNnue;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}.nnue", name, std::process::id(), nanos))
    }

    fn write_quant_header(
        f: &mut File,
        version: u32,
        input_dim: u32,
        hidden_dim: u32,
        output_dim: u32,
        w1_scale: f32,
        w2_scale: f32,
    ) {
        f.write_all(b"PIENNQ01").unwrap();
        f.write_all(&version.to_le_bytes()).unwrap();
        f.write_all(&input_dim.to_le_bytes()).unwrap();
        f.write_all(&hidden_dim.to_le_bytes()).unwrap();
        f.write_all(&output_dim.to_le_bytes()).unwrap();
        f.write_all(&w1_scale.to_le_bytes()).unwrap();
        f.write_all(&w2_scale.to_le_bytes()).unwrap();
    }

    #[test]
    fn quant_loader_rejects_truncated_payload() {
        let path = tmp_path("quant_truncated");
        let mut f = File::create(&path).unwrap();
        write_quant_header(&mut f, 1, 12, 4, 1, 1.0, 1.0);
        // Intentionally omit payload.
        drop(f);
        let loaded = QuantNnue::load_quantized(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_err(), "truncated quant model must fail to load");
    }

    #[test]
    fn quant_loader_rejects_non_positive_scales() {
        let path = tmp_path("quant_bad_scales");
        let mut f = File::create(&path).unwrap();
        write_quant_header(&mut f, 1, 1, 1, 1, 0.0, -1.0);
        // Minimal payload for input=1, hidden=1, output=1
        f.write_all(&[0u8]).unwrap(); // w1
        f.write_all(&0i16.to_le_bytes()).unwrap(); // b1
        f.write_all(&[0u8]).unwrap(); // w2
        f.write_all(&0i16.to_le_bytes()).unwrap(); // b2
        drop(f);
        let loaded = QuantNnue::load_quantized(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_err(), "non-positive scales must be rejected");
    }
}
