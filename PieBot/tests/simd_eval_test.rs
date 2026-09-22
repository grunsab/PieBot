use cozy_chess::Board;
use piebot::eval::nnue::features::HALFKP_DP_PER_PERSPECTIVE_DIM;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue, QuantNnueV2};
use piebot::eval::nnue::network::QuantNetwork;
use std::sync::Arc;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}

fn make_test_v2_model(hidden: usize, seed: u64) -> QuantNnue {
    let input = HALFKP_DP_PER_PERSPECTIVE_DIM;
    let mut s = seed;
    let w1: Vec<i16> = (0..input * hidden)
        .map(|_| (lcg(&mut s) % 101) as i16 - 50)
        .collect();
    let b1: Vec<i16> = (0..hidden).map(|_| (lcg(&mut s) % 61) as i16 - 30).collect();
    let w2: Vec<i8> = (0..2 * hidden)
        .map(|_| ((lcg(&mut s) % 256) as i32 - 128) as i8)
        .collect();
    let v2 = QuantNnueV2 {
        per_perspective_input_dim: input,
        hidden_dim: hidden,
        qa: 255,
        qb: 64,
        scale: 400,
        w1,
        b1,
        w2,
        b2: 123,
    };
    QuantNnue {
        meta: QuantMeta {
            version: 2,
            input_dim: input,
            hidden_dim: hidden,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1: Vec::new(),
        b1: Vec::new(),
        w2: Vec::new(),
        b2: Vec::new(),
        v2: Some(Arc::new(v2)),
    }
}

#[test]
fn test_simd_incremental_eval_exact_h1024() {
    // Test on production hidden_dim = 1024
    let mut net = QuantNetwork::new(make_test_v2_model(1024, 0x1234_5678_9abc));
    let mut board = Board::default();
    net.refresh(&board);

    let mut undo = Vec::new();
    let mut seed = 0xbeef_cafe_0001u64;

    for ply in 0..40 {
        let mut moves = Vec::new();
        board.generate_moves(|batch| {
            moves.extend(batch);
            false
        });
        if moves.is_empty() {
            break;
        }
        let mv = moves[(lcg(&mut seed) as usize) % moves.len()];
        let before = board.clone();
        board.play_unchecked(mv);
        undo.push((board.clone(), net.apply_move(&before, mv, &board)));

        let incr_eval = net.eval_current();
        let full_eval = net.eval_full(&board);
        assert_eq!(
            incr_eval, full_eval,
            "ply {ply}: incr_eval={incr_eval} != full_eval={full_eval}"
        );
    }

    // Revert backwards and assert exact equality
    while let Some((b, change)) = undo.pop() {
        assert_eq!(net.eval_current(), net.eval_full(&b));
        net.revert(change);
    }
}
