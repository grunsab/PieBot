use cozy_chess::{Board, Move};
use piebot::eval::nnue::features::{halfkp_dim, halfkp_v2_dim, HalfKpA, HalfKpSchema, HalfKpV2};
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::eval::nnue::network::QuantNetwork;
use piebot::eval::nnue::Nnue;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn tmp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{name}_{}_{}.nnue", std::process::id(), nanos))
}

fn find_move(board: &Board, uci: &str) -> Move {
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

fn write_dense_one_feature(path: &Path, input_dim: usize, feature_idx: usize) {
    let mut file = File::create(path).expect("create dense test network");
    file.write_all(b"PIENNUE1").unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&(input_dim as u32).to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    file.write_all(&1u32.to_le_bytes()).unwrap();
    for idx in 0..input_dim {
        let weight = if idx == feature_idx { 10.0f32 } else { 0.0 };
        file.write_all(&weight.to_le_bytes()).unwrap();
    }
    file.write_all(&0.0f32.to_le_bytes()).unwrap();
    file.write_all(&1.0f32.to_le_bytes()).unwrap();
    file.write_all(&0.0f32.to_le_bytes()).unwrap();
}

fn quant_one_feature(input_dim: usize, feature_idx: usize) -> QuantNnue {
    let mut w1 = vec![0; input_dim];
    w1[feature_idx] = 10;
    QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim: 1,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1,
        b1: vec![0],
        w2: vec![1],
        b2: vec![0],
        v2: None,
    }
}

#[test]
fn schemas_preserve_legacy_and_add_full_perspective_features() {
    let board = Board::default();
    let legacy = HalfKpA.active_indices(&board);
    let full = HalfKpV2.active_indices(&board);

    assert_eq!(halfkp_dim(), 40_960);
    assert_eq!(halfkp_v2_dim(), 81_920);
    assert_eq!(legacy.len(), 30);
    assert_eq!(full.len(), 60);
    assert_eq!(
        legacy.len(),
        legacy.iter().copied().collect::<HashSet<_>>().len()
    );
    assert_eq!(
        full.len(),
        full.iter().copied().collect::<HashSet<_>>().len()
    );
    assert!(legacy.iter().all(|&idx| idx < halfkp_dim()));
    assert!(full.iter().all(|&idx| idx < halfkp_v2_dim()));
    assert_eq!(
        HalfKpSchema::from_input_dim(halfkp_dim()),
        Some(HalfKpSchema::Legacy)
    );
    assert_eq!(
        HalfKpSchema::from_input_dim(halfkp_v2_dim()),
        Some(HalfKpSchema::FullPerspective)
    );
}

#[test]
fn full_schema_keys_an_opponent_piece_to_each_king() {
    let board = Board::from_fen("4k3/8/8/8/8/7r/4P3/4K3 w - - 0 1", false).unwrap();
    let active = HalfKpV2.active_indices(&board);

    // Two non-king pieces are represented once from each king's perspective.
    assert_eq!(active.len(), 4);
    // White king e1, black rook h3: perspective=white, king=4,
    // colored piece plane=black rook=8, square=23.
    let black_rook_from_white_king = ((4usize * 10 + 8) * 64) + 23;
    assert!(active.contains(&black_rook_from_white_king));
}

#[test]
fn dense_v2_selects_features_from_its_input_dimension() {
    let board = Board::from_fen("4k3/8/8/8/8/7r/4P3/4K3 w - - 0 1", false).unwrap();
    let feature_idx = ((4usize * 10 + 8) * 64) + 23;
    let path = tmp_path("dense_v2_schema");
    write_dense_one_feature(&path, halfkp_v2_dim(), feature_idx);

    let network = Nnue::load(&path).expect("load dense v2 network");
    std::fs::remove_file(path).ok();
    assert_eq!(network.evaluate(&board), 10);
}

#[test]
fn dense_v2_incremental_update_uses_full_schema() {
    let start = Board::default();
    // White perspective, white king e1, white pawn e2.
    let e2_feature_idx = ((4usize * 10) * 64) + 12;
    let path = tmp_path("dense_v2_incremental");
    write_dense_one_feature(&path, halfkp_v2_dim(), e2_feature_idx);

    let mut incremental = Nnue::load(&path).expect("load incremental dense v2 network");
    let full = Nnue::load(&path).expect("load full dense v2 network");
    std::fs::remove_file(path).ok();
    incremental.refresh_accumulator(&start);
    assert_eq!(incremental.evaluate(&start), 10);

    let mv = find_move(&start, "e2e4");
    let mut after = start.clone();
    after.play_unchecked(mv);
    incremental.update_on_move(mv);
    assert_eq!(incremental.evaluate(&after), 0);
    assert_eq!(incremental.evaluate(&after), full.evaluate(&after));
}

#[test]
fn quant_v2_selects_features_from_its_input_dimension() {
    let board = Board::from_fen("4k3/8/8/8/8/7r/4P3/4K3 w - - 0 1", false).unwrap();
    let feature_idx = ((4usize * 10 + 8) * 64) + 23;
    let mut network = QuantNetwork::new(quant_one_feature(halfkp_v2_dim(), feature_idx));

    network.refresh(&board);
    assert_eq!(network.eval_current(), 10);
    assert_eq!(network.eval_full(&board), 10);
}

#[test]
fn quant_v2_incremental_matches_full_and_revert() {
    let input_dim = halfkp_v2_dim();
    let mut w1 = vec![0i8; 4 * input_dim];
    for (idx, weight) in w1.iter_mut().enumerate() {
        *weight = ((idx.wrapping_mul(17).wrapping_add(3) % 15) as i8) - 7;
    }
    let model = QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim: 4,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1,
        b1: vec![3, -2, 7, 1],
        w2: vec![2, -1, 3, 1],
        b2: vec![5],
            v2: None,
        };
    let mut network = QuantNetwork::new(model);
    let mut board = Board::default();
    network.refresh(&board);

    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1h1"] {
        let before_eval = network.eval_current();
        let mv = find_move(&board, uci);
        let mut after = board.clone();
        after.play_unchecked(mv);
        let change = network.apply_move(&board, mv, &after);
        assert_eq!(
            network.eval_current(),
            network.eval_full(&after),
            "after {uci}"
        );
        network.revert(change);
        assert_eq!(network.eval_current(), before_eval, "revert {uci}");
        assert_eq!(
            network.eval_current(),
            network.eval_full(&board),
            "full revert {uci}"
        );
        board = after;
        network.refresh(&board);
    }
}
