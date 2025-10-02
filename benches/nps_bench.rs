use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::search::alphabeta::{SearchParams, Searcher};
use std::time::{Duration, Instant};

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

fn run_once_nodes(
    board: &Board,
    threads: usize,
    use_nnue: bool,
    blend: u8,
    quant: Option<QuantNnue>,
    movetime_ms: u64,
) -> (u64, Duration) {
    let mut s = Searcher::default();
    let mut p = SearchParams::default();
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = threads;
    p.movetime = Some(Duration::from_millis(movetime_ms));
    // Ensure iterative deepening runs; rely on deadline to stop
    p.depth = 99;
    if use_nnue {
        s.set_use_nnue(true);
        s.set_eval_blend_percent(blend);
        if let Some(q) = quant {
            s.set_nnue_quant_model(q);
        }
    }
    // Use a per-case thread pool to avoid global pool effects
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let t0 = Instant::now();
    let r = pool.install(|| s.search_with_params(board, p));
    let dt = t0.elapsed();
    (r.nodes, dt)
}

fn bench_nps(c: &mut Criterion) {
    let b = Board::default();
    let model = make_random_quant_model(64);
    let mut group = c.benchmark_group("nps");
    let cases = vec![
        ("pst_t1", 1usize, false, 0u8),
        ("pst_t4", 4usize, false, 0u8),
        ("nnue_t1", 1usize, true, 100u8),
        ("nnue_t4", 4usize, true, 100u8),
        ("blend50_t1", 1usize, true, 50u8),
        ("blend50_t4", 4usize, true, 50u8),
    ];
    for (name, threads, use_nnue, blend) in cases {
        let label = format!("nps_{}", name);
        group.bench_function(label, |ben| {
            ben.iter(|| {
                let (nodes, dt) = run_once_nodes(
                    black_box(&b),
                    threads,
                    use_nnue,
                    blend,
                    Some(model.clone()),
                    100,
                );
                let nps = if dt.as_secs_f64() > 0.0 {
                    nodes as f64 / dt.as_secs_f64()
                } else {
                    0.0
                };
                // print for quick visibility (Criterion reports time separately)
                println!(
                    "{}: nodes={}, elapsed={:.3} s, nps={:.1}",
                    name,
                    nodes,
                    dt.as_secs_f64(),
                    nps
                );
                black_box(nps)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_nps);
criterion_main!(benches);
