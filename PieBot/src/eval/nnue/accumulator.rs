/// Int16 accumulator buffer for NNUE first layer outputs.
#[derive(Clone)]
pub struct Accumulator {
    pub buf: Vec<i16>,
}

impl Accumulator {
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            buf: vec![0; hidden_dim],
        }
    }
    pub fn clear(&mut self) {
        for v in &mut self.buf {
            *v = 0;
        }
    }
}
