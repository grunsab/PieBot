use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_eval(c: &mut Criterion) {
    let b = Board::default();
    c.bench_function("eval_cp_startpos", |ben| {
        ben.iter(|| {
            let v = piebot::search::eval::eval_cp(black_box(&b));
            black_box(v)
        })
    });
}

criterion_group!(benches, bench_eval);
criterion_main!(benches);
