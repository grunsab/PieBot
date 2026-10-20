//! First hidden layer of the arch-v3 hidden stack, `2h (u8) -> l1 (i32)`,
//! computed sparsely in the manner of Stockfish's sparse-input affine
//! transform: most SCReLU activations are zero, so the input is split into
//! 4-byte blocks, the non-zero blocks are listed, and only those are
//! multiplied in. Weights are stored block-major (for each 4-input block, the
//! `l1` outputs' 4 bytes are contiguous) so one block is one broadcast and one
//! `vpdpbusd` per 16 outputs.
//!
//! The result is exactly the dense integer product; `dense_ref` is the
//! reference the tests compare against.

/// Row-major `[bucket][l1][n]` to block-major `[bucket][n/4][l1][4]`.
pub fn block_major(w: &[i8], buckets: usize, l1: usize, n: usize) -> Vec<i8> {
    assert_eq!(n % 4, 0, "input width must be a multiple of 4");
    assert_eq!(w.len(), buckets * l1 * n);
    let blocks = n / 4;
    let mut out = vec![0i8; w.len()];
    for b in 0..buckets {
        for j in 0..l1 {
            let row = &w[(b * l1 + j) * n..(b * l1 + j + 1) * n];
            for blk in 0..blocks {
                let dst = ((b * blocks + blk) * l1 + j) * 4;
                out[dst..dst + 4].copy_from_slice(&row[blk * 4..blk * 4 + 4]);
            }
        }
    }
    out
}

/// SCReLU to u8, side to move first: `x[i] = (clamp(a, 0, qa))^2 >> 8`.
/// `qa` must be at most 255 so the result fits a byte.
pub fn activate(first: &[i16], second: &[i16], qa: i32, x: &mut [u8]) {
    let h = first.len();
    debug_assert!(qa <= 255 && second.len() == h && x.len() >= 2 * h);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
    {
        if h % 32 == 0 {
            unsafe {
                activate_avx512(first, qa, &mut x[..h]);
                activate_avx512(second, qa, &mut x[h..2 * h]);
            }
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if h % 32 == 0 && is_x86_feature_detected!("avx2") {
            unsafe {
                activate_avx2(first, qa, &mut x[..h]);
                activate_avx2(second, qa, &mut x[h..2 * h]);
            }
            return;
        }
    }
    activate_scalar(first, qa, &mut x[..h]);
    activate_scalar(second, qa, &mut x[h..2 * h]);
}

fn activate_scalar(acc: &[i16], qa: i32, x: &mut [u8]) {
    let q = qa as i16;
    for (d, &a) in x.iter_mut().zip(acc) {
        let v = a.clamp(0, q) as u16;
        *d = ((v * v) >> 8) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn activate_avx2(acc: &[i16], qa: i32, x: &mut [u8]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let q = _mm256_set1_epi16(qa as i16);
    let ap = acc.as_ptr();
    let xp = x.as_mut_ptr();
    for i in 0..acc.len() / 32 {
        let a0 = _mm256_loadu_si256(ap.add(i * 32) as *const __m256i);
        let a1 = _mm256_loadu_si256(ap.add(i * 32 + 16) as *const __m256i);
        let v0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), q);
        let v1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), q);
        let s0 = _mm256_srli_epi16(_mm256_mullo_epi16(v0, v0), 8);
        let s1 = _mm256_srli_epi16(_mm256_mullo_epi16(v1, v1), 8);
        // packus works per 128-bit lane; the permute restores input order.
        let p = _mm256_permute4x64_epi64(_mm256_packus_epi16(s0, s1), 0xD8);
        _mm256_storeu_si256(xp.add(i * 32) as *mut __m256i, p);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
#[inline]
unsafe fn activate_avx512(acc: &[i16], qa: i32, x: &mut [u8]) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let q = _mm512_set1_epi16(qa as i16);
    let ap = acc.as_ptr();
    let xp = x.as_mut_ptr();
    for i in 0..acc.len() / 32 {
        let a = _mm512_loadu_si512(ap.add(i * 32) as *const _);
        let v = _mm512_min_epi16(_mm512_max_epi16(a, zero), q);
        // v * v < 2^16, so the low half of the product is the exact square.
        let s = _mm512_srli_epi16(_mm512_mullo_epi16(v, v), 8);
        _mm256_storeu_si256(xp.add(i * 32) as *mut __m256i, _mm512_cvtepi16_epi8(s));
    }
}

/// Dense reference: `out[j] = sum_k x[k] * w[j][k]` over row-major weights.
pub fn dense_ref(x: &[u8], w_rows: &[i8], l1: usize, out: &mut [i32]) {
    let n = x.len();
    for j in 0..l1 {
        let row = &w_rows[j * n..(j + 1) * n];
        out[j] = x.iter().zip(row).map(|(&a, &b)| a as i32 * b as i32).sum();
    }
}

/// `out[..l1] = x . W` over one bucket's block-major weights (`[n/4][l1][4]`),
/// exactly. AVX-512 VNNI when compiled in (dense or block-sparse, chosen per
/// call by how many blocks are non-zero), else AVX2, else scalar.
pub fn product(x: &[u8], w_blk: &[i8], l1: usize, out: &mut [i32]) {
    debug_assert_eq!(x.len() % 4, 0);
    debug_assert_eq!(w_blk.len(), x.len() * l1);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512bw"))]
    {
        if l1 % 16 == 0 && x.len() % 64 == 0 && x.len() <= 8192 {
            unsafe { vnni(x, w_blk, l1, out) };
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if l1 % 16 == 0 && is_x86_feature_detected!("avx2") {
            unsafe { dense_avx2(x, w_blk, l1, out) };
            return;
        }
    }
    sparse_scalar(x, w_blk, l1, out);
}

/// The portable fallback of `product`, public so tests cover it everywhere.
pub fn product_scalar(x: &[u8], w_blk: &[i8], l1: usize, out: &mut [i32]) {
    sparse_scalar(x, w_blk, l1, out)
}

fn sparse_scalar(x: &[u8], w_blk: &[i8], l1: usize, out: &mut [i32]) {
    out[..l1].fill(0);
    for (blk, xb) in x.chunks_exact(4).enumerate() {
        if xb == [0, 0, 0, 0] {
            continue;
        }
        let w = &w_blk[blk * l1 * 4..(blk + 1) * l1 * 4];
        for (j, o) in out[..l1].iter_mut().enumerate() {
            let wj = &w[j * 4..j * 4 + 4];
            *o += xb[0] as i32 * wj[0] as i32
                + xb[1] as i32 * wj[1] as i32
                + xb[2] as i32 * wj[2] as i32
                + xb[3] as i32 * wj[3] as i32;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512bw"))]
#[inline]
unsafe fn vnni(x: &[u8], w_blk: &[i8], l1: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let blocks = x.len() / 4;
    let x32 = x.as_ptr() as *const i32;
    let mut masks = [0u16; 128];
    let mut cnt = 0usize;
    for (i, m) in masks.iter_mut().enumerate().take(blocks / 16) {
        let v = _mm512_loadu_si512(x32.add(i * 16) as *const _);
        *m = _mm512_test_epi32_mask(v, v);
        cnt += m.count_ones() as usize;
    }
    let wp = w_blk.as_ptr();
    let row = l1 * 4; // weight bytes per block
    if cnt * 4 > blocks * 3 {
        // Mostly non-zero: a straight pass is cheaper than the index list.
        for c in 0..l1 / 16 {
            let woff = c * 64;
            let mut a = [_mm512_setzero_si512(); 4];
            for i in (0..blocks).step_by(4) {
                for (u, acc) in a.iter_mut().enumerate() {
                    let b = i + u;
                    let w = _mm512_loadu_si512(wp.add(b * row + woff) as *const _);
                    *acc = _mm512_dpbusd_epi32(*acc, _mm512_set1_epi32(*x32.add(b)), w);
                }
            }
            let s = _mm512_add_epi32(_mm512_add_epi32(a[0], a[1]), _mm512_add_epi32(a[2], a[3]));
            _mm512_storeu_si512(out.as_mut_ptr().add(c * 16) as *mut _, s);
        }
        return;
    }
    // List the non-zero blocks. Compress in a register, then store: a
    // compressing store to memory is microcoded and very slow on Zen 4.
    let mut idx = [0u32; 2048 + 16];
    let mut n = 0usize;
    let step = _mm512_set1_epi32(16);
    let mut lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for &m in masks.iter().take(blocks / 16) {
        _mm512_storeu_si512(idx.as_mut_ptr().add(n) as *mut _, _mm512_maskz_compress_epi32(m, lanes));
        n += m.count_ones() as usize;
        lanes = _mm512_add_epi32(lanes, step);
    }
    let idx = &idx[..n];
    for c in 0..l1 / 16 {
        let woff = c * 64;
        let mut a = [_mm512_setzero_si512(); 4];
        let mut chunks = idx.chunks_exact(4);
        for q in &mut chunks {
            for (u, acc) in a.iter_mut().enumerate() {
                let b = q[u] as usize;
                let w = _mm512_loadu_si512(wp.add(b * row + woff) as *const _);
                *acc = _mm512_dpbusd_epi32(*acc, _mm512_set1_epi32(*x32.add(b)), w);
            }
        }
        for &b in chunks.remainder() {
            let b = b as usize;
            let w = _mm512_loadu_si512(wp.add(b * row + woff) as *const _);
            a[0] = _mm512_dpbusd_epi32(a[0], _mm512_set1_epi32(*x32.add(b)), w);
        }
        let s = _mm512_add_epi32(_mm512_add_epi32(a[0], a[1]), _mm512_add_epi32(a[2], a[3]));
        _mm512_storeu_si512(out.as_mut_ptr().add(c * 16) as *mut _, s);
    }
}

/// Exact AVX2 product: inputs and weights widened to i16, `vpmaddwd` per
/// input pair.
/// Each 16-weight register covers 4 outputs x 4 inputs of one block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dense_avx2(x: &[u8], w_blk: &[i8], l1: usize, out: &mut [i32]) {
    use std::arch::x86_64::*;
    let w8 = w_blk.as_ptr();
    let blocks = x.len() / 4;
    let x32 = x.as_ptr() as *const i32;
    let row = l1 * 4;
    for c in 0..l1 / 16 {
        let woff = c * 64;
        let mut a = [_mm256_setzero_si256(); 4];
        // No zero-block skip: the activations are dense, and the branch
        // mispredicts more than the multiply it would save.
        for b in 0..blocks {
            // The block's 4 bytes as i16, repeated: [x0 x1 x2 x3] x 4.
            let xv = _mm256_cvtepu8_epi16(_mm_set1_epi32(x32.add(b).read_unaligned()));
            let base = w8.add(b * row + woff);
            for (u, acc) in a.iter_mut().enumerate() {
                let w = _mm256_cvtepi8_epi16(_mm_loadu_si128(base.add(u * 16) as *const __m128i));
                *acc = _mm256_add_epi32(*acc, _mm256_madd_epi16(xv, w));
            }
        }
        // a[u] holds outputs 4u..4u+3 as pairs of partial sums.
        let lo = _mm256_permute4x64_epi64(_mm256_hadd_epi32(a[0], a[1]), 0xD8);
        let hi = _mm256_permute4x64_epi64(_mm256_hadd_epi32(a[2], a[3]), 0xD8);
        _mm256_storeu_si256(out.as_mut_ptr().add(c * 16) as *mut __m256i, lo);
        _mm256_storeu_si256(out.as_mut_ptr().add(c * 16 + 8) as *mut __m256i, hi);
    }
}
