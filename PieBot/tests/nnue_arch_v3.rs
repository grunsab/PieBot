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
    let mut s = seed;
    let nf = 16 * 768;
    let w1 = (0..nf * hidden).map(|_| (lcg(&mut s) % 61) as i16 - 30).collect();
    let b1 = (0..hidden).map(|_| (lcg(&mut s) % 41) as i16 - 20).collect();
    let w2 = (0..8 * 2 * hidden).map(|_| (lcg(&mut s) % 121) as i8 - 60).collect();
    let b2 = (0..8).map(|_| (lcg(&mut s) % 2001) as i32 - 1000).collect();
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
            hidden_dim: hidden,
            n_output_buckets: 8,
            qa: 255,
            qb: 64,
            scale: 400,
            w1,
            b1,
            w2,
            b2,
        })),
    }
}

fn play_random_games(lazy: bool) {
    let mut net = QuantNetwork::new(random_v3(32, 0xabc0_ffee));
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
