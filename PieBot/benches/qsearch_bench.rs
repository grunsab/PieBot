use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_qsearch(c: &mut Criterion) {
    let start = Board::default();
    c.bench_function("qsearch_startpos", |ben| {
        ben.iter(|| {
            let mut s = piebot::search::alphabeta::Searcher::default();
            let v = s.qsearch_eval_cp(black_box(&start));
            black_box(v)
        })
    });
}

criterion_group!(benches, bench_qsearch);
criterion_main!(benches);
