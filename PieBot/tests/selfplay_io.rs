use piebot::selfplay::{
    generate_games, read_shard, write_jsonl_shards, write_shards, SelfPlayParams,
};
use std::fs::create_dir_all;

#[test]
fn write_and_read_shard() {
    let params = SelfPlayParams {
        games: 3,
        max_plies: 8,
        threads: 1,
        use_engine: false,
        depth: 2,
        movetime_ms: None,
        seed: 123,
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
    let games = generate_games(&params);
    let outdir = std::path::Path::new("target/selfplay_test");
    create_dir_all(outdir).unwrap();
    let shards = write_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let recs = read_shard(&shards[0]).unwrap();
    assert!(!recs.is_empty());
}

#[test]
fn write_jsonl_shard_contains_fen_result_best_move() {
    let params = SelfPlayParams {
        games: 2,
        max_plies: 8,
        threads: 1,
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
    let games = generate_games(&params);
    let outdir = std::path::Path::new("target/selfplay_jsonl_test");
    create_dir_all(outdir).unwrap();
    let shards = write_jsonl_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let content = std::fs::read_to_string(&shards[0]).unwrap();
    let first = content.lines().next().expect("jsonl line");
    let v: serde_json::Value = serde_json::from_str(first).unwrap();
    assert!(v.get("fen").and_then(|x| x.as_str()).is_some());
    assert!(v.get("played_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("best_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("target_best_move").and_then(|x| x.as_str()).is_some());
    assert!(v.get("result").and_then(|x| x.as_i64()).is_some());
    assert!(v.get("result_q").and_then(|x| x.as_f64()).is_some());
}

#[test]
fn write_jsonl_shard_contains_ply_value_and_policy_top_for_engine_games() {
    let params = SelfPlayParams {
        games: 1,
        max_plies: 4,
        threads: 1,
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
    let games = generate_games(&params);
    let outdir = std::path::Path::new("target/selfplay_jsonl_value_test");
    create_dir_all(outdir).unwrap();
    let shards = write_jsonl_shards(&games, outdir, 10).unwrap();
    assert!(!shards.is_empty());
    let content = std::fs::read_to_string(&shards[0]).unwrap();
    let first = content.lines().next().expect("jsonl line");
    let v: serde_json::Value = serde_json::from_str(first).unwrap();
    assert!(v.get("ply").and_then(|x| x.as_u64()).is_some());
    assert!(v.get("value_cp").and_then(|x| x.as_f64()).is_some());
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
    let games = generate_games(&params);
    assert_eq!(games.len(), 1);
    assert_eq!(games[0].start_fen, opening_fen);
}
