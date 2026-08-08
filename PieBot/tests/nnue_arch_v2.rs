//! Arch-v2 NNUE parity tests: dual-perspective feature encoding must match
//! the Python trainer byte-for-byte, incremental accumulator updates must
//! match full recomputation exactly, and the SCReLU integer head must match
//! a reference implementation.

use cozy_chess::{Board, Color};
use piebot::eval::nnue::features::{dp_active_indices, HALFKP_DP_PER_PERSPECTIVE_DIM};
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue, QuantNnueV2};
use piebot::eval::nnue::network::QuantNetwork;
use std::sync::Arc;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed
}

fn random_v2_model(hidden: usize, seed: u64) -> QuantNnue {
    let input = HALFKP_DP_PER_PERSPECTIVE_DIM;
    let mut s = seed;
    let w1: Vec<i16> = (0..input * hidden)
        .map(|_| (lcg(&mut s) % 61) as i16 - 30)
        .collect();
    let b1: Vec<i16> = (0..hidden).map(|_| (lcg(&mut s) % 41) as i16 - 20).collect();
    let w2: Vec<i8> = (0..2 * hidden)
        .map(|_| (lcg(&mut s) % 121) as i8 - 60)
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
        b2: 137,
    };
    QuantNnue {
        meta: QuantMeta {
            version: 1,
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
fn dual_perspective_indices_match_python_fixture() {
    let raw = include_str!("data/halfkp_dp_fixture.json");
    let fixture: serde_json::Value = serde_json::from_str(raw).expect("valid fixture json");
    assert_eq!(
        fixture["per_perspective_dim"].as_u64().unwrap() as usize,
        HALFKP_DP_PER_PERSPECTIVE_DIM
    );
    let positions = fixture["positions"].as_array().unwrap();
    assert!(!positions.is_empty());
    for row in positions {
        let fen = row["fen"].as_str().unwrap();
        let board = Board::from_fen(fen, false).expect("valid fen");
        for (key, perspective) in [
            ("white_perspective", Color::White),
            ("black_perspective", Color::Black),
        ] {
            let mut rust: Vec<usize> = dp_active_indices(&board, perspective);
            rust.sort_unstable();
            let python: Vec<usize> = row[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(rust, python, "fen={fen} perspective={perspective:?}");
        }
        let stm_is_white = row["stm_is_white"].as_bool().unwrap();
        assert_eq!(board.side_to_move() == Color::White, stm_is_white, "fen={fen}");
    }
}

#[test]
fn incremental_updates_match_full_recompute_over_random_games() {
    let mut net = QuantNetwork::new(random_v2_model(8, 0x5eed_a11c_e5));
    let mut seed = 20260808u64;
    for game in 0..24 {
        let mut board = Board::default();
        net.refresh(&board);
        let mut undo = Vec::new();
        for ply in 0..60 {
            assert_eq!(
                net.eval_current(),
                net.eval_full(&board),
                "game={game} ply={ply} incremental accumulator diverged"
            );
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
            undo.push(net.apply_move(&before, mv, &board));
        }
        // Walking the change stack back down must restore state exactly.
        for change in undo.into_iter().rev() {
            net.revert(change);
        }
        net.refresh(&board);
    }
}

#[test]
fn screlu_head_matches_reference_formula() {
    let model = random_v2_model(8, 0xfeed_beef);
    let mut net = QuantNetwork::new(model.clone());
    let board =
        Board::from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 5", false)
            .unwrap();
    net.refresh(&board);

    // Reference: recompute accumulators and the SCReLU head from scratch in
    // plain i64 arithmetic, stm half first, then flip to white-POV.
    let v2 = model.v2.as_ref().unwrap();
    let h = v2.hidden_dim;
    let mut accs = Vec::new();
    for perspective in [Color::White, Color::Black] {
        let mut acc: Vec<i64> = v2.b1.iter().map(|&b| b as i64).collect();
        for idx in dp_active_indices(&board, perspective) {
            for j in 0..h {
                acc[j] += v2.w1[idx * h + j] as i64;
            }
        }
        accs.push(acc);
    }
    let stm_black_first = [&accs[1], &accs[0]];
    let qa = v2.qa as i64;
    let mut sum = 0i64;
    for (half, acc) in stm_black_first.into_iter().enumerate() {
        for j in 0..h {
            let v = acc[j].clamp(0, qa);
            sum += v * v * (v2.w2[half * h + j] as i64);
        }
    }
    let out_stm = (sum + v2.b2 as i64) * v2.scale as i64 / (qa * qa * v2.qb as i64);
    let expected_white_pov = -(out_stm as i32); // black to move

    assert_eq!(net.eval_current(), expected_white_pov);
    assert_eq!(net.eval_full(&board), expected_white_pov);
}
