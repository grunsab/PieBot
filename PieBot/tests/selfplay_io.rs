use piebot::selfplay::{
    flatten_game_to_records, generate_games, read_shard, write_jsonl_shards, write_shards,
    GameRecord, GameTermination, SelfPlayParams,
};
use std::fs::create_dir_all;

#[test]
fn write_and_read_shard() {
    let games = vec![GameRecord {
        run_id: "test-run".to_string(),
        game_id: "test-game".to_string(),
        start_fen: cozy_chess::Board::default().to_string(),
        moves: vec![
            "f2f3".to_string(),
            "e7e5".to_string(),
            "g2g4".to_string(),
            "d8h4".to_string(),
        ],
        move_target_best: vec![None; 4],
        move_value_cp: vec![None; 4],
        move_teacher_depth: vec![None; 4],
        move_policy_top: vec![Vec::new(); 4],
        result: -1,
        outcome_valid: true,
        termination: GameTermination::Checkmate,
    }];
    let outdir = std::path::Path::new("target/selfplay_test");
    create_dir_all(outdir).unwrap();
    let shards = write_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let recs = read_shard(&shards[0]).unwrap();
    assert!(!recs.is_empty());
}

#[test]
fn binary_records_omit_games_without_a_real_outcome() {
    let invalid = GameRecord {
        run_id: "test-run".to_string(),
        game_id: "test-invalid-game".to_string(),
        start_fen: cozy_chess::Board::default().to_string(),
        moves: vec!["e2e4".to_string()],
        move_target_best: vec![Some("e2e4".to_string())],
        move_value_cp: vec![Some(12.0)],
        move_teacher_depth: vec![Some(2)],
        move_policy_top: vec![Vec::new()],
        result: 0,
        outcome_valid: false,
        termination: GameTermination::MaxPlies,
    };

    assert!(
        flatten_game_to_records(&invalid).is_empty(),
        "the v1 binary format cannot encode outcome validity, so truncated games must be omitted"
    );
}

#[test]
fn write_jsonl_shard_contains_fen_result_best_move() {
    let params = SelfPlayParams {
        games: 2,
        max_plies: 8,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 2,
        movetime_ms: None,
        seed: 456,
        temperature_tau: 0.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        dirichlet_plies: 0,
        temperature_moves: 0,
        openings_path: None,
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    };
    let games = generate_games(&params).expect("selfplay games");
    let outdir = std::path::Path::new("target/selfplay_jsonl_test");
    create_dir_all(outdir).unwrap();
    let shards = write_jsonl_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let content = std::fs::read_to_string(&shards[0]).unwrap();
    let first = content.lines().next().expect("jsonl line");
    let v: serde_json::Value = serde_json::from_str(first).unwrap();
    assert!(v.get("fen").and_then(|x| x.as_str()).is_some());
    assert!(v.get("run_id").and_then(|x| x.as_str()).is_some());
    assert!(v.get("game_id").and_then(|x| x.as_str()).is_some());
    assert!(v.get("played_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("best_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("target_best_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("result").and_then(|x| x.as_i64()).is_some());
    assert!(v.get("result_q").and_then(|x| x.as_f64()).is_some());
    assert_eq!(
        v.get("outcome_valid").and_then(|x| x.as_bool()),
        Some(false)
    );
    assert_eq!(
        v.get("termination").and_then(|x| x.as_str()),
        Some("max_plies")
    );
    assert!(v.get("teacher_depth").is_none());
}

#[test]
fn write_jsonl_shard_contains_ply_value_and_policy_top_for_engine_games() {
    let params = SelfPlayParams {
        games: 1,
        max_plies: 4,
        threads: 1,
        parallel_games: 1,
        use_engine: true,
        depth: 1,
        movetime_ms: None,
        seed: 777,
        temperature_tau: 1.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        dirichlet_plies: 4,
        temperature_moves: 4,
        openings_path: None,
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    };
    let games = generate_games(&params).expect("selfplay games");
    let outdir = std::path::Path::new("target/selfplay_jsonl_value_test");
    create_dir_all(outdir).unwrap();
    let shards = write_jsonl_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let content = std::fs::read_to_string(&shards[0]).unwrap();
    let first = content.lines().next().expect("jsonl line");
    let v: serde_json::Value = serde_json::from_str(first).unwrap();
    assert!(v.get("ply").and_then(|x| x.as_u64()).is_some());
    assert!(v.get("value_cp").and_then(|x| x.as_f64()).is_some());
    assert_eq!(v.get("teacher_depth").and_then(|x| x.as_u64()), Some(1));
    assert!(v.get("run_id").and_then(|x| x.as_str()).is_some());
    assert!(v.get("game_id").and_then(|x| x.as_str()).is_some());
    assert!(v.get("target_best_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("played_move").and_then(|x| x.as_str()).is_some());
    let policy = v.get("policy_top").and_then(|x| x.as_array()).unwrap();
    assert!(!policy.is_empty());
}

#[test]
fn selfplay_preserves_opening_start_fen() {
    let opening_fen = "8/8/8/8/8/8/4K3/7k w - - 0 1";
    let openings_path = std::path::Path::new("target/selfplay_openings_test.txt");
    std::fs::write(openings_path, format!("{}\n", opening_fen)).unwrap();

    let params = SelfPlayParams {
        games: 1,
        max_plies: 2,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 2,
        movetime_ms: None,
        seed: 1,
        temperature_tau: 0.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        dirichlet_plies: 0,
        temperature_moves: 0,
        openings_path: Some(openings_path.to_path_buf()),
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    };
    let games = generate_games(&params).expect("selfplay games");
    assert_eq!(games.len(), 1);
    assert_eq!(games[0].start_fen, opening_fen);
}
