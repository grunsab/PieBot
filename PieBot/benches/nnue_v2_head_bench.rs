//! Throughput of the arch-v2 SCReLU head at the shipped hidden-1024 shape.
//!
//! This times `eval_current`, which for a v2 model is exactly the SCReLU dot
//! product over the two concatenated accumulators — the kernel that dominates
//! v2 node cost. Random weights are appropriate here in a way they are NOT for
//! search NPS: there is no tree, so there is no degenerate-eval bias. The
//! accumulator contents are what the kernel sees, and they are seeded to span
//! the clamp boundary (below 0, inside [0, QA], and above QA) the way trained
//! accumulators do.

use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_v2_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue, QuantNnueV2};
use piebot::eval::nnue::network::QuantNetwork;
use std::sync::Arc;

const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

fn build_v2_network(hidden_dim: usize) -> QuantNetwork {
    let per_perspective_input_dim = halfkp_v2_dim();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };

    // Feature rows small enough that summing ~32 active features lands the
    // accumulator in the same range a trained net produces, straddling QA.
    let w1_len = per_perspective_input_dim * hidden_dim;
    let mut w1 = Vec::with_capacity(w1_len);
    for _ in 0..w1_len {
        w1.push(((next() % 25) as i32 - 12) as i16);
    }
    let mut b1 = Vec::with_capacity(hidden_dim);
    for _ in 0..hidden_dim {
        b1.push(((next() % 61) as i32 - 30) as i16);
    }
    let mut w2 = Vec::with_capacity(2 * hidden_dim);
    for _ in 0..2 * hidden_dim {
        w2.push(((next() % 256) as i32 - 128) as i8);
    }

    let v2 = QuantNnueV2 {
        per_perspective_input_dim,
        hidden_dim,
        qa: QA,
        qb: QB,
        scale: SCALE,
        w1,
        b1,
        w2,
        b2: 0,
    };

    QuantNetwork::new(QuantNnue {
        meta: QuantMeta {
            version: 2,
            input_dim: per_perspective_input_dim,
            hidden_dim,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1: Vec::new(),
        b1: Vec::new(),
        w2: Vec::new(),
        b2: Vec::new(),
        v2: Some(Arc::new(v2)),
    })
}

fn bench_head(c: &mut Criterion) {
    let board = Board::default();
    for hidden_dim in [256usize, 1024] {
        let mut network = build_v2_network(hidden_dim);
        network.refresh(&board);
        // Guard against benchmarking a degenerate constant: if the head folded
        // to zero the timing would be meaningless.
        assert_ne!(network.eval_current(), 0, "hidden_dim={hidden_dim}");
        c.bench_function(&format!("v2_screlu_head_h{hidden_dim}"), |b| {
            b.iter(|| black_box(network.eval_current()))
        });
    }
}

/// What the search actually pays per node: fold the move into both
/// accumulators, evaluate, then undo. `apply_delta` touches `hidden_dim` lanes
/// per changed feature per perspective, so at h1024 a quiet move rewrites
/// several thousand i16 lanes — potentially dwarfing the head's 2048.
fn bench_apply_eval_revert(c: &mut Criterion) {
    let board = Board::default();
    let mv = "e2e4".parse().expect("legal move");
    let mut after = board.clone();
    after.play(mv);

    for hidden_dim in [256usize, 1024] {
        let mut network = build_v2_network(hidden_dim);
        network.refresh(&board);
        c.bench_function(&format!("v2_apply_eval_revert_h{hidden_dim}"), |b| {
            b.iter(|| {
                let change = network.apply_move(&board, mv, &after);
                let score = black_box(network.eval_current());
                network.revert(change);
                score
            })
        });
    }
}

/// Same work as `bench_apply_eval_revert`, but cycling through many distinct
/// moves so the feature rows are not L1-resident. w1 is ~84 MB at h1024 and
/// each feature row is 2 KB, so a real search streams scattered rows out of
/// DRAM. The gap between this and the single-move bench is the memory cost
/// that no amount of SIMD can remove.
fn bench_apply_cache_realistic(c: &mut Criterion) {
    let board: Board = "r2q1rk1/1b2bppp/p2ppn2/1p6/3NP3/1BN5/PPP2PPP/R2Q1RK1 w - - 0 12"
        .parse()
        .expect("valid fen");
    let mut moves = Vec::new();
    board.generate_moves(|packed| {
        for m in packed {
            moves.push(m);
        }
        false
    });
    assert!(moves.len() > 20, "need a wide midgame position");

    let children: Vec<Board> = moves
        .iter()
        .map(|&m| {
            let mut b = board.clone();
            b.play(m);
            b
        })
        .collect();

    for hidden_dim in [1024usize] {
        let mut network = build_v2_network(hidden_dim);
        network.refresh(&board);
        let mut i = 0usize;
        c.bench_function(&format!("v2_apply_eval_revert_varied_h{hidden_dim}"), |b| {
            b.iter(|| {
                i = (i + 1) % moves.len();
                let change = network.apply_move(&board, moves[i], &children[i]);
                let score = black_box(network.eval_current());
                network.revert(change);
                score
            })
        });
    }
}

criterion_group!(
    benches,
    bench_head,
    bench_apply_eval_revert,
    bench_apply_cache_realistic
);
criterion_main!(benches);
