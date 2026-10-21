//! Times the arch-v3 hidden head pieces on a trained net's real accumulators
//! (squared SCReLU for file version 2, the pairwise product for version 5).
use cozy_chess::Board;
use piebot::eval::nnue::loader::QuantNnue;
use piebot::eval::nnue::sparse_l1;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("net path");
    let m = QuantNnue::load_quantized(&path).unwrap();
    let v3 = m.v3.as_ref().unwrap();
    let hs = v3.hidden_stack.as_ref().unwrap();
    let h = v3.hidden_dim;
    let n = if hs.pairwise { h } else { 2 * h };
    let act = |a: &[i16], b: &[i16], x: &mut [u8]| {
        if hs.pairwise {
            sparse_l1::activate_pairwise(a, b, v3.qa, x)
        } else {
            sparse_l1::activate(a, b, v3.qa, x)
        }
    };
    let fens = [
        "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9",
        "8/5pk1/6p1/3R4/7P/6P1/r4PK1/8 w - - 0 40",
        "r4rk1/pp3ppp/2n1b3/q1pp4/3P4/P1P1PN2/2Q1BPPP/R4RK1 w - - 0 14",
    ];
    for fen in fens {
        let b: Board = fen.parse().unwrap();
        let acc = piebot::eval::nnue::network_v3::debug_accumulators(v3, &b);
        let mut x = vec![0u8; n];
        act(&acc[0], &acc[1], &mut x);
        let nzb = x.chunks(4).filter(|c| c.iter().any(|&v| v != 0)).count();
        let nzx = x.iter().filter(|&&v| v != 0).count();
        let l1 = hs.l1;
        let w = &hs.h1_w_blk[..l1 * n];
        let mut out = [0i32; 256];
        let iters = 200_000;
        let t = Instant::now();
        for _ in 0..iters {
            sparse_l1::product(std::hint::black_box(&x), w, l1, &mut out);
            std::hint::black_box(&out);
        }
        let ts = t.elapsed().as_nanos() as f64 / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            act(std::hint::black_box(&acc[0]), &acc[1], &mut x);
            std::hint::black_box(&x);
        }
        let ta = t.elapsed().as_nanos() as f64 / iters as f64;
        println!("nonzero bytes {nzx}/{n}, blocks {nzb}/{}: sparse {ts:.0} ns, activate {ta:.0} ns", n / 4);
    }
}
