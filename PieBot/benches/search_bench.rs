use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_search(c: &mut Criterion) {
    let b = Board::default();
    c.bench_function("search_depth_4_startpos", |ben| {
        ben.iter(|| {
            let mut s = piebot::search::alphabeta::Searcher::default();
            let mut p = piebot::search::alphabeta::SearchParams::default();
            p.depth = 4;
            p.use_tt = true;
            p.order_captures = true;
            p.use_history = true;
            let r = s.search_with_params(black_box(&b), p);
            black_box(r.nodes)
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
