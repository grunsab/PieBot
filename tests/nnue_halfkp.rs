use cozy_chess::Board;
use std::fs::File;
use std::io::Write;

fn write_quant_file(path: &str, input_dim: u32, hidden_dim: u32) {
    let mut f = File::create(path).unwrap();
    // magic PIENNQ01
    f.write_all(b"PIENNQ01").unwrap();
    // version
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // dims
    f.write_all(&input_dim.to_le_bytes()).unwrap();
    f.write_all(&hidden_dim.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap(); // out=1
                                               // scales
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    // w1 random small values in [-4,4]
    let mut seed = 20240601u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as i32 % 9 - 4) as i8
    };
    for _ in 0..(input_dim as usize * hidden_dim as usize) {
        f.write_all(&[next() as u8]).unwrap();
    }
    // b1 zeros
    for _ in 0..hidden_dim {
        f.write_all(&0i16.to_le_bytes()).unwrap();
    }
    // w2 random small
    for _ in 0..hidden_dim {
        f.write_all(&[next() as u8]).unwrap();
    }
    // b2 zero
    f.write_all(&0i16.to_le_bytes()).unwrap();
}

#[test]
fn halfkp_active_count_startpos() {
    use piebot::eval::nnue::features::HalfKpA;
    let b = Board::default();
    let feats = HalfKpA;
    let act = feats.active_indices(&b);
    // Startpos has 32 pieces, minus 2 kings => 30 active non-king pieces
    assert_eq!(act.len(), 30, "expected 30 active features in startpos");
}

#[test]
fn halfkp_incremental_matches_full_over_sequence() {
    use cozy_chess::Move;
    use piebot::eval::nnue::features::halfkp_dim;
    use piebot::eval::nnue::loader::QuantNnue;
    use piebot::eval::nnue::network::QuantNetwork;

    let path = "target/halfkp_test.nnue";
    let input_dim = halfkp_dim() as u32;
    let hidden_dim = 8u32;
    write_quant_file(path, input_dim, hidden_dim);
    let model = QuantNnue::load_quantized(path).unwrap();
    let mut net = QuantNetwork::new(model);
    let mut b = Board::default();
    net.refresh(&b);
    // small deterministic sequence of legal moves
    let seq = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5",
    ]; // 8 plies
    for uci in &seq {
        // find matching move
        let mut chosen: Option<Move> = None;
        b.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == *uci {
                    chosen = Some(m);
                    break;
                }
            }
            chosen.is_some()
        });
        let m = chosen.expect("legal move in sequence");
        let mut after = b.clone();
        after.play_unchecked(m);
        let change = net.apply_move(&b, m, &after);
        let inc = net.eval_current();
        let full = net.eval_full(&after);
        assert_eq!(inc, full, "incremental vs full mismatch for move {}", uci);
        net.revert(change);
        b = after;
        net.refresh(&b);
    }
}
