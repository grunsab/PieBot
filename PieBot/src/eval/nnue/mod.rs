use cozy_chess::{Board, Color, Move, Piece};
pub mod accumulator;
pub mod features;
pub mod loader;
pub mod network;
pub mod quant;
use anyhow::{bail, Context, Result};
use std::collections::HashSet;
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
    w1: Vec<f32>, // hidden_dim x input_dim
    b1: Vec<f32>, // hidden_dim
    w2: Vec<f32>, // output_dim x hidden_dim (output_dim=1)
    b2: Vec<f32>, // output_dim
    // Cached incremental state for callers that use refresh/update APIs.
    acc: Vec<f32>,          // pre-ReLU hidden sums
    active: HashSet<usize>, // active HalfKP indices when input_dim == halfkp_dim()
    current_board: Option<Board>,
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
        if input_dim == 0 || hidden_dim == 0 {
            bail!(
                "invalid dense dims: input_dim={} hidden_dim={}",
                input_dim,
                hidden_dim
            );
        }
        if output_dim != 1 {
            bail!("unsupported dense output_dim {}; expected 1", output_dim);
        }

        let w1_len = hidden_dim
            .checked_mul(input_dim)
            .context("dense dimension overflow: hidden_dim * input_dim")?;
        let w2_len = output_dim
            .checked_mul(hidden_dim)
            .context("dense dimension overflow: output_dim * hidden_dim")?;

        let w1 = read_f32s_exact(&mut r, w1_len, "read dense w1 payload")?;
        let b1 = read_f32s_exact(&mut r, hidden_dim, "read dense b1 payload")?;
        let w2 = read_f32s_exact(&mut r, w2_len, "read dense w2 payload")?;
        let b2 = read_f32s_exact(&mut r, output_dim, "read dense b2 payload")?;
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
            acc: vec![0.0; hidden_dim],
            active: HashSet::new(),
            current_board: None,
        })
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        if self.current_board.as_ref() == Some(board) && self.acc.len() == self.meta.hidden_dim {
            return self.eval_from_acc();
        }
        if self.meta.input_dim == features::halfkp_dim() {
            return self.eval_halfkp_sparse(board);
        }
        let x = self.features(board);
        self.eval_dense_from_input(&x)
    }

    pub fn refresh_accumulator(&mut self, board: &Board) {
        self.recompute_acc(board);
        self.current_board = Some(board.clone());
        if self.meta.input_dim == features::halfkp_dim() {
            self.active = self.halfkp_active_set(board);
        } else {
            self.active.clear();
        }
    }

    pub fn update_on_move(&mut self, mv: Move) {
        let Some(mut next_board) = self.current_board.clone() else {
            return;
        };
        next_board.play_unchecked(mv);

        if self.meta.input_dim == features::halfkp_dim() && !self.active.is_empty() {
            self.apply_halfkp_delta(&next_board);
        } else {
            self.recompute_acc(&next_board);
            if self.meta.input_dim == features::halfkp_dim() {
                self.active = self.halfkp_active_set(&next_board);
            } else {
                self.active.clear();
            }
        }
        self.current_board = Some(next_board);
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
        if n == features::halfkp_dim() {
            let mut out = vec![0.0; n];
            for idx in features::HalfKpA.active_indices(board) {
                out[idx] = 1.0;
            }
            return out;
        }
        vec![0.0; n]
    }

    fn eval_dense_from_input(&self, x: &[f32]) -> i32 {
        let n = self.meta.input_dim;
        let h = self.meta.hidden_dim;
        let mut y1 = vec![0f32; h];
        for j in 0..h {
            let mut sum = self.b1[j];
            let row = &self.w1[j * n..(j + 1) * n];
            for i in 0..n {
                sum += row[i] * x[i];
            }
            y1[j] = sum.max(0.0);
        }
        self.eval_head(&y1)
    }

    fn eval_head(&self, y1_relu: &[f32]) -> i32 {
        let h = self.meta.hidden_dim;
        let row = &self.w2[0..h];
        let mut sum = self.b2[0];
        for j in 0..h {
            sum += row[j] * y1_relu[j];
        }
        sum.round() as i32
    }

    fn eval_from_acc(&self) -> i32 {
        let h = self.meta.hidden_dim;
        let mut y1 = vec![0.0; h];
        for j in 0..h {
            y1[j] = self.acc[j].max(0.0);
        }
        self.eval_head(&y1)
    }

    fn eval_halfkp_sparse(&self, board: &Board) -> i32 {
        let active = self.halfkp_active_set(board);
        let h = self.meta.hidden_dim;
        let n = self.meta.input_dim;
        let mut y1 = self.b1.clone();
        for &idx in &active {
            for j in 0..h {
                y1[j] += self.w1[j * n + idx];
            }
        }
        for j in 0..h {
            y1[j] = y1[j].max(0.0);
        }
        self.eval_head(&y1)
    }

    fn recompute_acc(&mut self, board: &Board) {
        let n = self.meta.input_dim;
        let h = self.meta.hidden_dim;
        if self.acc.len() != h {
            self.acc = vec![0.0; h];
        }
        if n == features::halfkp_dim() {
            self.acc.copy_from_slice(&self.b1);
            let active = self.halfkp_active_set(board);
            for &idx in &active {
                for j in 0..h {
                    self.acc[j] += self.w1[j * n + idx];
                }
            }
        } else {
            let x = self.features(board);
            for j in 0..h {
                let mut sum = self.b1[j];
                let row = &self.w1[j * n..(j + 1) * n];
                for i in 0..n {
                    sum += row[i] * x[i];
                }
                self.acc[j] = sum;
            }
        }
    }

    fn halfkp_active_set(&self, board: &Board) -> HashSet<usize> {
        features::HalfKpA
            .active_indices(board)
            .into_iter()
            .collect()
    }

    fn apply_halfkp_delta(&mut self, after: &Board) {
        let h = self.meta.hidden_dim;
        let n = self.meta.input_dim;
        let after_set = self.halfkp_active_set(after);
        let removed: Vec<usize> = self.active.difference(&after_set).copied().collect();
        let added: Vec<usize> = after_set.difference(&self.active).copied().collect();

        for idx in &removed {
            for j in 0..h {
                self.acc[j] -= self.w1[j * n + *idx];
            }
            self.active.remove(idx);
        }
        for idx in &added {
            for j in 0..h {
                self.acc[j] += self.w1[j * n + *idx];
            }
            self.active.insert(*idx);
        }
    }
}

fn read_f32s_exact(r: &mut BufReader<File>, n: usize, ctx: &'static str) -> Result<Vec<f32>> {
    let nbytes = n
        .checked_mul(4)
        .context("dense dimension overflow: f32 byte count")?;
    let mut bytes = vec![0u8; nbytes];
    r.read_exact(&mut bytes).context(ctx)?;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Nnue;
    use crate::eval::nnue::features::halfkp_dim;
    use cozy_chess::{Board, Move};
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

    fn write_header(f: &mut File, version: u32, input_dim: u32, hidden_dim: u32, output_dim: u32) {
        f.write_all(b"PIENNUE1").unwrap();
        f.write_all(&version.to_le_bytes()).unwrap();
        f.write_all(&input_dim.to_le_bytes()).unwrap();
        f.write_all(&hidden_dim.to_le_bytes()).unwrap();
        f.write_all(&output_dim.to_le_bytes()).unwrap();
    }

    #[test]
    fn dense_loader_rejects_truncated_payload() {
        let path = tmp_path("dense_truncated");
        let mut f = File::create(&path).unwrap();
        write_header(&mut f, 1, 12, 4, 1);
        // Intentionally omit payload.
        drop(f);
        let err = Nnue::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(err.is_err(), "truncated dense model must fail to load");
    }

    #[test]
    fn dense_halfkp_input_uses_active_features() {
        let input_dim = halfkp_dim();
        let hidden_dim = 1usize;
        let output_dim = 1usize;
        // White king e1 => 4, white pawn e2 => 12, pawn piece slot => 0
        let e2_idx = (((0usize * 64 + 4usize) * 5usize + 0usize) * 64usize) + 12usize;

        let path = tmp_path("dense_halfkp");
        let mut f = File::create(&path).unwrap();
        write_header(
            &mut f,
            1,
            input_dim as u32,
            hidden_dim as u32,
            output_dim as u32,
        );

        for i in 0..(input_dim * hidden_dim) {
            let w = if i == e2_idx { 10.0f32 } else { 0.0f32 };
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // b1[0]
        f.write_all(&1.0f32.to_le_bytes()).unwrap(); // w2[0]
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // b2[0]
        drop(f);

        let nn = Nnue::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        let start = Board::default();
        let no_e2 = Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
            false,
        )
        .unwrap();

        assert_eq!(nn.evaluate(&start), 10);
        assert_eq!(nn.evaluate(&no_e2), 0);
    }

    fn find_move_uci(board: &Board, uci: &str) -> Move {
        let mut found = None;
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == uci {
                    found = Some(m);
                    break;
                }
            }
            found.is_some()
        });
        found.unwrap()
    }

    #[test]
    fn refresh_and_update_on_move_maintain_incremental_state() {
        let input_dim = halfkp_dim();
        let hidden_dim = 1usize;
        let output_dim = 1usize;
        // White king e1 => 4. Pawn squares: e2 => 12, e4 => 28
        let e2_idx = (((0usize * 64 + 4usize) * 5usize + 0usize) * 64usize) + 12usize;
        let e4_idx = (((0usize * 64 + 4usize) * 5usize + 0usize) * 64usize) + 28usize;

        let path = tmp_path("dense_halfkp_incremental");
        let mut f = File::create(&path).unwrap();
        write_header(
            &mut f,
            1,
            input_dim as u32,
            hidden_dim as u32,
            output_dim as u32,
        );
        for i in 0..(input_dim * hidden_dim) {
            let w = if i == e2_idx {
                10.0f32
            } else if i == e4_idx {
                6.0f32
            } else {
                0.0f32
            };
            f.write_all(&w.to_le_bytes()).unwrap();
        }
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // b1[0]
        f.write_all(&1.0f32.to_le_bytes()).unwrap(); // w2[0]
        f.write_all(&0.0f32.to_le_bytes()).unwrap(); // b2[0]
        drop(f);

        let mut nn = Nnue::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let start = Board::default();
        nn.refresh_accumulator(&start);
        assert_eq!(nn.current_board.as_ref(), Some(&start));
        assert_eq!(nn.evaluate(&start), 10);

        let mv = find_move_uci(&start, "e2e4");
        nn.update_on_move(mv);
        let mut after = start.clone();
        after.play_unchecked(mv);
        assert_eq!(nn.current_board.as_ref(), Some(&after));
        assert_eq!(nn.evaluate(&after), 6);
    }
}
