use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::eval::nnue::network::QuantNetwork;

fn make_random_quant_model(hidden_dim: usize) -> QuantNnue {
    let input_dim = halfkp_dim();
    let mut seed = 0x1234_5678_9abc_def0u64;
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
    let mut b1 = vec![0i16; hidden_dim];
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

fn bench_nnue_quant_eval(c: &mut Criterion) {
    // modest hidden size for benchmark speed; adjust as needed
    let model = make_random_quant_model(64);
    let net = QuantNetwork::new(model);
    let b = Board::default();
    c.bench_function("nnue_quant_eval_full_startpos", |ben| {
        ben.iter(|| {
            // call full eval path
            let v = net.eval_full(black_box(&b));
            black_box(v)
        })
    });
}

criterion_group!(benches, bench_nnue_quant_eval);
criterion_main!(benches);
