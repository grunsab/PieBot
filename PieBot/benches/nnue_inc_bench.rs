use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_v2_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::eval::nnue::network::QuantNetwork;

fn make_random_quant_model(hidden_dim: usize) -> QuantNnue {
    let input_dim = halfkp_v2_dim();
    let mut seed = 0xfeedfacecafebeefu64;
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

fn find_move(board: &Board, uci: &str) -> cozy_chess::Move {
    let mut found = None;
    board.generate_moves(|moves| {
        for mv in moves {
            if mv.to_string() == uci {
                found = Some(mv);
                break;
            }
        }
        found.is_some()
    });
    found.unwrap_or_else(|| panic!("move {uci} must be legal in {board}"))
}

fn prepare_sequence() -> Vec<(Board, cozy_chess::Move, Board)> {
    let mut out = Vec::new();
    let mut b = Board::default();
    for uci in [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5", "c2c3", "d7d6", "b1d2",
        "c8g4",
    ] {
        let mv = find_move(&b, uci);
        let mut nb = b.clone();
        nb.play_unchecked(mv);
        out.push((b.clone(), mv, nb.clone()));
        b = nb;
    }
    out
}

fn bench_nnue_incremental(c: &mut Criterion) {
    let seq = prepare_sequence();
    let model = make_random_quant_model(64);
    let mut net = QuantNetwork::new(model);
    let (quiet_before, quiet_move, quiet_after) = &seq[0];
    net.refresh(quiet_before);

    c.bench_function("nnue_v2_eval_current", |ben| {
        ben.iter(|| black_box(net.eval_current()))
    });

    c.bench_function("nnue_v2_quiet_apply_revert", |ben| {
        ben.iter(|| {
            let change = net.apply_move(
                black_box(quiet_before),
                black_box(*quiet_move),
                black_box(quiet_after),
            );
            net.revert(change);
        })
    });

    c.bench_function("nnue_v2_quiet_apply_eval_revert", |ben| {
        ben.iter(|| {
            let change = net.apply_move(
                black_box(quiet_before),
                black_box(*quiet_move),
                black_box(quiet_after),
            );
            let value = net.eval_current();
            net.revert(change);
            black_box(value)
        })
    });

    c.bench_function("nnue_v2_refresh_startpos", |ben| {
        ben.iter(|| net.refresh(black_box(quiet_before)))
    });

    let mut changes = Vec::with_capacity(seq.len());
    c.bench_function("nnue_v2_opening_line_apply_eval_unwind", |ben| {
        ben.iter(|| {
            net.refresh(quiet_before);
            let mut checksum = 0;
            for (before, mv, after) in &seq {
                changes.push(net.apply_move(before, *mv, after));
                checksum ^= net.eval_current();
            }
            while let Some(change) = changes.pop() {
                net.revert(change);
            }
            black_box(checksum)
        })
    });
}

criterion_group!(benches, bench_nnue_incremental);
criterion_main!(benches);
