use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};

fn make_random_quant_model(hidden_dim: usize) -> QuantNnue {
    let input_dim = halfkp_dim();
    let mut seed = 0x9e3779b97f4a7c15u64;
    let mut next_i8 = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = ((seed >> 32) as i32 % 7) - 3; // [-3,3]
        v as i8
    };
    let w1_len = hidden_dim * input_dim;
    let mut w1 = Vec::with_capacity(w1_len);
    for _ in 0..w1_len {
        w1.push(next_i8());
    }
    let b1 = vec![0i16; hidden_dim];
    let mut w2 = Vec::with_capacity(hidden_dim);
    for _ in 0..hidden_dim {
        w2.push(next_i8());
    }
    let b2 = vec![0i16; 1];
    QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1,
        b1,
        w2,
        b2,
    }
}

fn bench_blended_eval(c: &mut Criterion) {
    let b = Board::default();
    let model = make_random_quant_model(64);
    // Using full recompute for overhead quantification; incremental is used in search paths.
    let mut group = c.benchmark_group("blended_eval");
    for &blend in &[0u8, 25, 50, 75, 100] {
        group.bench_function(format!("blend_{}", blend), |ben| {
            ben.iter(|| {
                // Quant eval (full) + PST, then mix
                let nn = piebot::eval::nnue::network::QuantNetwork::new(model.clone());
                let nn_val = nn.eval_full(black_box(&b));
                let pst = piebot::search::eval::eval_cp(black_box(&b));
                let mixed = if blend >= 100 {
                    nn_val
                } else if blend == 0 {
                    pst
                } else {
                    ((nn_val as i64 * blend as i64 + pst as i64 * (100 - blend) as i64) / 100)
                        as i32
                };
                black_box(mixed)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_blended_eval);
criterion_main!(benches);
