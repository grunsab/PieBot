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
}

const Q_MAGIC: &[u8; 8] = b"PIENNQ01"; // Pie NNUE Quant v1

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

        fn read_fill<T: Copy + Default>(
            r: &mut BufReader<File>,
            elem_size: usize,
            n: usize,
        ) -> Result<Vec<u8>> {
            use std::io::Read as _;
            let total = n * elem_size;
            let mut buf = vec![0u8; total];
            let mut off = 0usize;
            while off < total {
                match r.read(&mut buf[off..]) {
                    Ok(0) => break,
                    Ok(k) => off += k,
                    Err(e) => return Err(e).context("read weights"),
                }
            }
            Ok(buf)
        }

        let w1_bytes = read_fill::<i8>(&mut r, 1, hidden_dim * input_dim)?;
        let b1_bytes = read_fill::<i16>(&mut r, 2, hidden_dim)?;
        let w2_bytes = read_fill::<i8>(&mut r, 1, output_dim * hidden_dim)?;
        let b2_bytes = read_fill::<i16>(&mut r, 2, output_dim)?;

        let w1 = w1_bytes.into_iter().map(|b| b as i8).collect();
        let mut b1 = Vec::with_capacity(hidden_dim);
        for i in 0..hidden_dim {
            let lo = *b1_bytes.get(2 * i).unwrap_or(&0) as u8;
            let hi = *b1_bytes.get(2 * i + 1).unwrap_or(&0) as u8;
            let val = i16::from_le_bytes([lo, hi]);
            b1.push(val);
        }
        let w2 = w2_bytes.into_iter().map(|b| b as i8).collect();
        let mut b2 = Vec::with_capacity(output_dim);
        for i in 0..output_dim {
            let lo = *b2_bytes.get(2 * i).unwrap_or(&0) as u8;
            let hi = *b2_bytes.get(2 * i + 1).unwrap_or(&0) as u8;
            b2.push(i16::from_le_bytes([lo, hi]));
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
        })
    }
}
