//! Arch-v3 (PIENNQ03) parity: incremental accumulator updates, eager and
//! lazy, must equal full recomputation along random games (captures,
//! castling, en passant, promotions, king moves across buckets and mirrors),
//! and the engine must agree with the Python reference evaluator
//! (`training/nnue/train_v3.py::quant_eval_fen`) on its exported fixture.

use cozy_chess::Board;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue, QuantNnueV3};
use piebot::eval::nnue::network::QuantNetwork;
use std::sync::Arc;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 11
}

const LAYOUT: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 12, 12, 13, 13, 14, 14, 15, 15,
    14, 14, 15, 15,
];

fn random_v3(hidden: usize, seed: u64) -> QuantNnue {
    random_v3_experts(hidden, seed, 1)
}

fn expert_table(n: usize) -> [u8; 33] {
    let mut t = [0u8; 33];
    if n == 4 {
        for (p, e) in t.iter_mut().enumerate() {
            *e = if p >= 25 { 0 } else if p >= 17 { 1 } else if p >= 9 { 2 } else { 3 };
        }
    }
    t
}

fn random_v3_experts(hidden: usize, seed: u64, experts: usize) -> QuantNnue {
    let mut s = seed;
    let nf = experts * 16 * 768;
    let w1 = (0..nf * hidden).map(|_| (lcg(&mut s) % 61) as i16 - 30).collect();
    let b1 = (0..hidden).map(|_| (lcg(&mut s) % 41) as i16 - 20).collect();
    let w2 = (0..8 * 2 * hidden).map(|_| (lcg(&mut s) % 121) as i8 - 60).collect();
    let b2 = (0..8).map(|_| (lcg(&mut s) % 2001) as i64 - 1000).collect();
    QuantNnue {
        meta: QuantMeta { version: 1, input_dim: nf, hidden_dim: hidden, output_dim: 1 },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1: Vec::new(),
        b1: Vec::new(),
        w2: Vec::new(),
        b2: Vec::new(),
        v2: None,
        v3: Some(Arc::new(QuantNnueV3 {
            n_king_buckets: 16,
            king_bucket_layout: LAYOUT,
            n_experts: experts,
            expert_of_pieces: expert_table(experts),
            hidden_dim: hidden,
            n_output_buckets: 8,
            qa: 255,
            qb: 64,
            scale: 400,
            w1,
            b1,
            w2,
            w2_lo: Vec::new(),
            w2_shift: 0,
            b2,
            hidden_stack: None,
        })),
    }
}

fn play_random_games(lazy: bool) {
    play_random_games_with(lazy, 1);
}

fn play_random_games_with(lazy: bool, experts: usize) {
    let mut net = QuantNetwork::new(random_v3_experts(32, 0xabc0_ffee, experts));
    net.set_lazy(lazy);
    let mut seed = if lazy { 99 } else { 7 };
    for game in 0..30 {
        let mut board = Board::default();
        net.refresh(&board);
        let mut undo = Vec::new();
        let mut ancestors = Vec::new();
        for ply in 0..160 {
            let mut moves = Vec::new();
            board.generate_moves(|b| {
                moves.extend(b);
                false
            });
            if moves.is_empty() {
                break;
            }
            ancestors.push(board.clone());
            let mv = moves[(lcg(&mut seed) as usize) % moves.len()];
            let before = board.clone();
            board.play_unchecked(mv);
            undo.push(net.apply_move(&before, mv, &board));
            if !lazy || lcg(&mut seed) % 3 == 0 {
                net.ensure_computed();
                assert_eq!(net.eval_current(), net.eval_full(&board), "lazy={lazy} game={game} ply={ply}");
            }
        }
        while let Some(change) = undo.pop() {
            net.revert(change);
            let a = ancestors.pop().unwrap();
            net.ensure_computed();
            assert_eq!(net.eval_current(), net.eval_full(&a), "lazy={lazy} game={game} unwind");
        }
    }
}

#[test]
fn incremental_v3_matches_full_recompute_eager() {
    play_random_games(false);
}

#[test]
fn incremental_v3_matches_full_recompute_lazy() {
    play_random_games(true);
}

#[test]
fn null_move_flips_the_perspective_order_only() {
    let mut net = QuantNetwork::new(random_v3(16, 5));
    let board: Board = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3".parse().unwrap();
    net.refresh(&board);
    let null = board.null_move().unwrap();
    let ch = net.apply_null_move(&null);
    assert_eq!(net.eval_current(), net.eval_full(&null));
    net.revert(ch);
    assert_eq!(net.eval_current(), net.eval_full(&board));
}

#[test]
fn engine_matches_the_python_reference_fixture() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let model = QuantNnue::load_quantized(dir.join("v3_fixture.nnue")).expect("load PIENNQ03 fixture");
    let cases: Vec<(String, i32)> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("v3_fixture_evals.json")).unwrap()).unwrap();
    let net = QuantNetwork::new(model);
    assert!(cases.len() >= 50);
    for (fen, want_stm) in cases {
        let b = Board::from_fen(&fen, false).unwrap();
        let white = net.eval_full(&b);
        let stm = if b.side_to_move() == cozy_chess::Color::White { white } else { -white };
        assert_eq!(stm, want_stm, "{fen}");
    }
}

#[test]
fn hidden_stack_engine_matches_the_python_reference_fixture() {
    // The f32 layers can round differently from numpy by one centipawn.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let model = QuantNnue::load_quantized(dir.join("v3h_fixture.nnue")).expect("load PIENNQ03 v2 fixture");
    assert!(model.v3.as_ref().unwrap().hidden_stack.is_some());
    let cases: Vec<(String, i32)> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("v3h_fixture_evals.json")).unwrap()).unwrap();
    let net = QuantNetwork::new(model);
    let mut exact = 0;
    for (fen, want_stm) in &cases {
        let b = Board::from_fen(fen, false).unwrap();
        let white = net.eval_full(&b);
        let stm = if b.side_to_move() == cozy_chess::Color::White { white } else { -white };
        assert!((stm - want_stm).abs() <= 1, "{fen}: engine {stm} python {want_stm}");
        exact += (stm == *want_stm) as usize;
    }
    assert!(exact * 10 >= cases.len() * 9, "only {exact}/{} exact", cases.len());
}

#[test]
fn incremental_v3_with_material_experts_matches_full_recompute() {
    // Random games shed material quickly, so captures cross every expert
    // boundary and force rebuilds from the refresh cache.
    play_random_games_with(false, 4);
    play_random_games_with(true, 4);
}

#[test]
fn experts_engine_matches_the_python_reference_fixture() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let model = QuantNnue::load_quantized(dir.join("v3e_fixture.nnue")).expect("load experts fixture");
    assert_eq!(model.v3.as_ref().unwrap().n_experts, 4);
    let cases: Vec<(String, i32)> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("v3e_fixture_evals.json")).unwrap()).unwrap();
    let net = QuantNetwork::new(model);
    for (fen, want_stm) in cases {
        let b = Board::from_fen(&fen, false).unwrap();
        let white = net.eval_full(&b);
        let stm = if b.side_to_move() == cozy_chess::Color::White { white } else { -white };
        assert_eq!(stm, want_stm, "{fen}");
    }
}

#[test]
fn sparse_first_layer_equals_the_dense_product() {
    use piebot::eval::nnue::sparse_l1;
    let mut s = 0x5eed_u64;
    for &(l1, n) in &[(16usize, 2048usize), (32, 2048), (16, 512), (8, 2048), (12, 64)] {
        let w: Vec<i8> = (0..l1 * n).map(|_| (lcg(&mut s) % 256) as i32 as u8 as i8).collect();
        let blk = sparse_l1::block_major(&w, 1, l1, n);
        for zero_pct in [0u64, 50, 90, 99, 100] {
            let x: Vec<u8> =
                (0..n).map(|_| if lcg(&mut s) % 100 < zero_pct { 0 } else { (lcg(&mut s) % 255) as u8 }).collect();
            let (mut want, mut got) = ([0i32; 64], [0i32; 64]);
            sparse_l1::dense_ref(&x, &w, l1, &mut want);
            sparse_l1::product(&x, &blk, l1, &mut got);
            assert_eq!(&got[..l1], &want[..l1], "l1={l1} n={n} zero={zero_pct}%");
            let mut got2 = [0i32; 64];
            sparse_l1::product_scalar(&x, &blk, l1, &mut got2);
            assert_eq!(&got2[..l1], &want[..l1], "scalar l1={l1} n={n} zero={zero_pct}%");
        }
    }
}

#[test]
fn simd_activation_matches_the_scalar_formula() {
    use piebot::eval::nnue::sparse_l1;
    let mut s = 77u64;
    let h = 1024;
    let a: Vec<i16> = (0..2 * h).map(|_| (lcg(&mut s) % 1200) as i16 - 600).collect();
    let mut x = vec![0u8; 2 * h];
    sparse_l1::activate(&a[..h], &a[h..], 255, &mut x);
    for (i, &v) in a.iter().enumerate() {
        let c = (v as i32).clamp(0, 255);
        assert_eq!(x[i] as i32, (c * c) >> 8, "i={i}");
    }
}

#[test]
fn precise_head_engine_matches_the_python_reference_fixture() {
    // File version 4: i16 output weights (split hi/lo at load), 4 experts.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let model = QuantNnue::load_quantized(dir.join("v3p_fixture.nnue")).expect("load version-4 fixture");
    let v3 = model.v3.as_ref().unwrap();
    assert_eq!((v3.qb, v3.w2_shift, v3.n_experts), (4096, 6, 4));
    let cases: Vec<(String, i32)> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("v3p_fixture_evals.json")).unwrap()).unwrap();
    assert!(cases.len() >= 50);
    let net = QuantNetwork::new(model);
    for (fen, want_stm) in cases {
        let b = Board::from_fen(&fen, false).unwrap();
        let white = net.eval_full(&b);
        let stm = if b.side_to_move() == cozy_chess::Color::White { white } else { -white };
        assert_eq!(stm, want_stm, "{fen}");
    }
}

#[test]
fn split_head_kernel_equals_its_scalar_reference() {
    let mut s = 77u64;
    for &len in &[16usize, 32, 48, 1024, 1040, 2048] {
        for _ in 0..20 {
            let acc: Vec<i16> = (0..len).map(|_| (lcg(&mut s) % 700) as i16 - 200).collect();
            let hi: Vec<i8> = (0..len).map(|_| (lcg(&mut s) % 256) as i16 as i8).collect();
            let lo: Vec<i8> = (0..len).map(|_| (lcg(&mut s) % 64) as i8).collect();
            let (fast, slow) = piebot::eval::nnue::network_v3::split_dot_for_tests(&acc, &hi, &lo, 255);
            assert_eq!(fast, slow, "len {len}");
        }
        // Worst case: every activation saturated, extreme weights.
        let acc = vec![255i16; len];
        let hi = vec![-128i8; len];
        let lo = vec![63i8; len];
        let (fast, slow) = piebot::eval::nnue::network_v3::split_dot_for_tests(&acc, &hi, &lo, 255);
        assert_eq!(fast, slow, "saturated len {len}");
    }
}
