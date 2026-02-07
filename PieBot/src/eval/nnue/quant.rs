/// Quantization utilities and scalar kernels.

#[inline]
pub fn dot_i8_i16(w_row: &[i8], x: &[i16]) -> i32 {
    // Scalar reference; SIMD-accelerated paths will replace this under feature flags.
    let mut acc: i32 = 0;
    for i in 0..w_row.len() {
        acc += (w_row[i] as i32) * (x[i] as i32);
    }
    acc
}
