//! Threat inputs (PIENNQ03 version 6): the engine must match
//! training/nnue/threats.py and train_v3.quant_eval_fen exactly.
//!
//! Fixture: tests/data/v3t_fixture.json (training/nnue/make_v3t_fixture.py).
//! The net is v3pw_fixture.nnue plus threat weights from `threat_weight`,
//! the same formula the fixture script uses, so no large file is committed.

use cozy_chess::Board;
use piebot::eval::nnue::loader::QuantNnue;
use piebot::eval::nnue::network::QuantNetwork;
use piebot::eval::nnue::threats;
use std::path::{Path, PathBuf};

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

#[derive(serde::Deserialize)]
struct Case {
    fen: String,
    white: Vec<u32>,
    black: Vec<u32>,
    eval: i32,
}

fn cases() -> Vec<Case> {
    serde_json::from_str(&std::fs::read_to_string(data().join("v3t_fixture.json")).unwrap()).unwrap()
}

fn threat_weight(f: i64, j: i64) -> i8 {
    (((31 * f + 17 * j) % 41) - 20) as i8
}

/// The version-6 fixture: v3pw_fixture.nnue with version 6 and the threat block appended.
fn threat_net_bytes(count: u32) -> Vec<u8> {
    let mut raw = std::fs::read(data().join("v3pw_fixture.nnue")).unwrap();
    assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), 5);
    raw[8..12].copy_from_slice(&6u32.to_le_bytes());
    let hidden = 64i64;
    raw.extend_from_slice(&count.to_le_bytes());
    for f in 0..threats::N_THREAT_FEATURES as i64 {
        for j in 0..hidden {
            raw.push(threat_weight(f, j) as u8);
        }
    }
    raw
}

fn load_bytes(raw: &[u8], name: &str) -> anyhow::Result<QuantNnue> {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("piebot_{}_{name}_{n}.nnue", std::process::id()));
    std::fs::write(&path, raw).unwrap();
    let r = QuantNnue::load_quantized(&path);
    let _ = std::fs::remove_file(&path);
    r
}

fn threat_fixture() -> QuantNnue {
    load_bytes(&threat_net_bytes(threats::N_THREAT_FEATURES as u32), "v3t").expect("load v6 fixture")
}

fn random_boards(games: usize, plies: usize, seed: u64, mut f: impl FnMut(&Board, &Board)) {
    let mut s = seed;
    for _ in 0..games {
        let mut board = Board::default();
        for _ in 0..plies {
            let mut moves = Vec::new();
            board.generate_moves(|b| {
                moves.extend(b);
                false
            });
            if moves.is_empty() {
                break;
            }
            let mv = moves[(lcg(&mut s) as usize) % moves.len()];
            let before = board.clone();
            board.play_unchecked(mv);
            f(&before, &board);
        }
    }
}

#[test]
fn feature_count_matches_the_python_reference() {
    assert_eq!(threats::N_THREAT_FEATURES, 59808);
}

#[test]
fn threat_features_match_the_python_reference() {
    for c in cases() {
        let b = Board::from_fen(&c.fen, false).unwrap();
        assert_eq!(threats::features(&b, 0), c.white, "{} white", c.fen);
        assert_eq!(threats::features(&b, 1), c.black, "{} black", c.fen);
    }
}

#[test]
fn incremental_changes_equal_the_difference_of_full_feature_sets() {
    let mut moves = 0;
    random_boards(60, 200, 0x7e57, |before, after| {
        for persp in 0..2 {
            if threats::mirrored(before, persp) != threats::mirrored(after, persp) {
                continue; // the engine rebuilds this perspective instead
            }
            let old = threats::features(before, persp);
            let new = threats::features(after, persp);
            let (mut added, mut removed) = (Vec::new(), Vec::new());
            threats::changed_features(before, after, persp, &mut added, &mut removed);
            added.sort_unstable();
            removed.sort_unstable();
            let want_removed: Vec<u32> = old.iter().copied().filter(|f| new.binary_search(f).is_err()).collect();
            let want_added: Vec<u32> = new.iter().copied().filter(|f| old.binary_search(f).is_err()).collect();
            assert_eq!(removed, want_removed, "removed: {} -> {} persp {persp}", before, after);
            assert_eq!(added, want_added, "added: {} -> {} persp {persp}", before, after);
            moves += 1;
        }
    });
    assert!(moves > 10_000, "only {moves} checked");
}

#[test]
fn threat_net_matches_the_python_reference_fixture() {
    // The f32 layers can round differently from numpy by one centipawn.
    let net = QuantNetwork::new(threat_fixture());
    let cases = cases();
    let mut exact = 0;
    for c in &cases {
        let b = Board::from_fen(&c.fen, false).unwrap();
        let white = net.eval_full(&b);
        let stm = if b.side_to_move() == cozy_chess::Color::White { white } else { -white };
        assert!((stm - c.eval).abs() <= 1, "{}: engine {stm} python {}", c.fen, c.eval);
        exact += (stm == c.eval) as usize;
    }
    assert!(exact * 10 >= cases.len() * 9, "only {exact}/{} exact", cases.len());
}

fn play_random_games(lazy: bool) {
    let mut net = QuantNetwork::new(threat_fixture());
    net.set_lazy(lazy);
    let mut seed = if lazy { 31 } else { 37 };
    for game in 0..30 {
        let mut board = Board::default();
        net.refresh(&board);
        let mut undo = Vec::new();
        let mut ancestors = Vec::new();
        for ply in 0..200 {
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
fn incremental_threat_net_matches_full_recompute_eager() {
    play_random_games(false);
}

#[test]
fn incremental_threat_net_matches_full_recompute_lazy() {
    play_random_games(true);
}

#[test]
fn threat_accumulators_include_every_active_threat_row() {
    // Version 6 minus version 5 accumulators = the sum of the active threat rows.
    let v6 = std::sync::Arc::clone(threat_fixture().v3.as_ref().unwrap());
    let v5 = std::sync::Arc::clone(
        QuantNnue::load_quantized(data().join("v3pw_fixture.nnue")).unwrap().v3.as_ref().unwrap(),
    );
    for c in cases().iter().take(40) {
        let b = Board::from_fen(&c.fen, false).unwrap();
        let a6 = piebot::eval::nnue::network_v3::debug_accumulators(&v6, &b);
        let a5 = piebot::eval::nnue::network_v3::debug_accumulators(&v5, &b);
        for (p, feats) in [&c.white, &c.black].into_iter().enumerate() {
            for j in 0..64 {
                let sum: i32 = feats.iter().map(|&f| threat_weight(f as i64, j as i64) as i32).sum();
                assert_eq!(a6[p][j] as i32 - a5[p][j] as i32, sum, "{} persp {p} lane {j}", c.fen);
            }
        }
    }
}

#[test]
fn loader_rejects_a_wrong_threat_feature_count() {
    let err = load_bytes(&threat_net_bytes(123), "bad").unwrap_err();
    assert!(format!("{err:#}").contains("threat"), "{err:#}");
}
