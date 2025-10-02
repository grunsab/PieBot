use cozy_chess::Board;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use piebot::eval::nnue::features::halfkp_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::eval::nnue::network::QuantNetwork;

fn make_random_quant_model(hidden_dim: usize) -> QuantNnue {
    let input_dim = halfkp_dim();
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

fn prepare_sequence(max_plies: usize) -> Vec<(Board, String, Board)> {
    let mut out = Vec::new();
    let mut b = Board::default();
    for _ in 0..max_plies {
        // Pick the first legal move (if any)
        let mut chosen: Option<String> = None;
        b.generate_moves(|ml| {
            for m in ml {
                chosen = Some(format!("{}", m));
                break;
            }
            chosen.is_some()
        });
        let Some(mstr) = chosen else { break };
        let mut nb = b.clone();
        let mut found = None;
        b.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == mstr {
                    found = Some(m);
                    break;
                }
            }
            found.is_some()
        });
        if let Some(mv) = found {
            nb.play(mv);
            out.push((b.clone(), mstr.clone(), nb.clone()));
            b = nb;
        } else {
            break;
        }
    }
    out
}

fn bench_nnue_incremental(c: &mut Criterion) {
    let seq = prepare_sequence(64);
    let model = make_random_quant_model(64);
    let mut net = QuantNetwork::new(model);
    if let Some((ref b0, _, _)) = seq.first() {
        net.refresh(b0);
    }

    c.bench_function("nnue_incremental_apply_revert", |ben| {
        ben.iter(|| {
            let mut acc = 0i32;
            for (before, mstr, after) in &seq {
                // find move again (string → move)
                let mut chosen = None;
                before.generate_moves(|ml| {
                    for m in ml {
                        if format!("{}", m) == *mstr {
                            chosen = Some(m);
                            break;
                        }
                    }
                    chosen.is_some()
                });
                let mv = chosen.expect("move should be legal");
                let change = net.apply_move(before, mv, after);
                acc ^= net.eval_current();
                net.revert(change);
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_nnue_incremental);
criterion_main!(benches);
