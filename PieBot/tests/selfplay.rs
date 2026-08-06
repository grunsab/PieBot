use piebot::selfplay::{generate_games, SelfPlayParams};

#[test]
fn selfplay_generates_games_deterministically() {
    let params = SelfPlayParams {
        games: 2,
        max_plies: 16,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 2,
        movetime_ms: None,
        seed: 42,
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
    let g1 = generate_games(&params).expect("selfplay games");
    let g2 = generate_games(&params).expect("selfplay games");
    assert_eq!(g1.len(), 2);
    assert_eq!(g2.len(), 2);
    // Deterministic by seed
    assert_eq!(g1[0].moves, g2[0].moves);
    assert_eq!(g1[0].run_id, g2[0].run_id);
    assert_eq!(g1[0].game_id, g2[0].game_id);
    assert_eq!(g1[0].run_id, g1[1].run_id);
    assert_ne!(g1[0].game_id, g1[1].game_id);
}

#[test]
fn max_ply_cutoff_is_not_labeled_as_a_real_draw() {
    let params = SelfPlayParams {
        games: 1,
        max_plies: 1,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 1,
        movetime_ms: None,
        seed: 7,
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
    assert_eq!(1, games.len());
    assert_eq!("max_plies", games[0].termination.as_str());
    assert!(!games[0].outcome_valid);
}

#[test]
fn selfplay_noise_changes_moves_with_different_seeds() {
    // With engine + noise, different seeds produce different sequences
    let mut p = SelfPlayParams {
        games: 1,
        max_plies: 10,
        threads: 1,
        parallel_games: 1,
        use_engine: true,
        depth: 2,
        movetime_ms: None,
        seed: 1,
        temperature_tau: 1.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        dirichlet_plies: 8,
        temperature_moves: 10,
        openings_path: None,
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    };
    let g1 = generate_games(&p).expect("selfplay games");
    p.seed = 2;
    let g2 = generate_games(&p).expect("selfplay games");
    assert_ne!(
        g1[0].moves, g2[0].moves,
        "noise did not alter move sampling"
    );
}

#[test]
fn selfplay_parallel_random_matches_serial_by_seed() {
    let mut params = SelfPlayParams {
        games: 8,
        max_plies: 12,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 2,
        movetime_ms: None,
        seed: 99,
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
    let serial = generate_games(&params).expect("selfplay games");
    params.parallel_games = 4;
    let parallel = generate_games(&params).expect("selfplay games");
    assert_eq!(serial.len(), parallel.len());
    for i in 0..serial.len() {
        assert_eq!(serial[i].start_fen, parallel[i].start_fen);
        assert_eq!(serial[i].moves, parallel[i].moves);
        assert_eq!(serial[i].result, parallel[i].result);
        assert_eq!(serial[i].run_id, parallel[i].run_id);
        assert_eq!(serial[i].game_id, parallel[i].game_id);
    }
}

#[test]
fn selfplay_parallel_engine_matches_serial_by_seed() {
    let mut params = SelfPlayParams {
        games: 4,
        max_plies: 10,
        threads: 1,
        parallel_games: 1,
        use_engine: true,
        depth: 1,
        movetime_ms: None,
        seed: 1234,
        temperature_tau: 1.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
        dirichlet_plies: 8,
        temperature_moves: 10,
        openings_path: None,
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    };
    let serial = generate_games(&params).expect("selfplay games");
    params.parallel_games = 4;
    let parallel = generate_games(&params).expect("selfplay games");
    assert_eq!(serial.len(), parallel.len());
    for i in 0..serial.len() {
        assert_eq!(serial[i].start_fen, parallel[i].start_fen);
        assert_eq!(serial[i].moves, parallel[i].moves);
        assert_eq!(serial[i].result, parallel[i].result);
        assert_eq!(serial[i].run_id, parallel[i].run_id);
        assert_eq!(serial[i].game_id, parallel[i].game_id);
    }
}

fn openings_params(openings_path: std::path::PathBuf) -> SelfPlayParams {
    SelfPlayParams {
        games: 4,
        max_plies: 4,
        threads: 1,
        parallel_games: 1,
        use_engine: false,
        depth: 1,
        movetime_ms: None,
        seed: 11,
        temperature_tau: 0.0,
        temp_cp_scale: 200.0,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        dirichlet_plies: 0,
        temperature_moves: 0,
        openings_path: Some(openings_path),
        temperature_tau_final: 0.1,
        nnue_quant_model: None,
        nnue_blend_percent: 100,
    }
}

fn write_temp_openings(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("piebot_openings_{}_{}.fen", name, std::process::id()));
    std::fs::write(&path, contents).expect("write openings file");
    path
}

#[test]
fn missing_openings_file_is_an_error() {
    let params = openings_params(std::path::PathBuf::from(
        "/nonexistent/piebot_no_such_openings.fen",
    ));
    let err = generate_games(&params).err().expect("missing openings file must fail loudly");
    assert!(
        err.contains("openings"),
        "error should mention openings: {err}"
    );
}

#[test]
fn openings_file_without_valid_positions_is_an_error() {
    let path = write_temp_openings("empty", "# only a comment\n\n");
    let params = openings_params(path.clone());
    let err = generate_games(&params).err().expect("empty openings suite must fail loudly");
    std::fs::remove_file(&path).ok();
    assert!(
        err.contains("no valid"),
        "error should say no valid positions: {err}"
    );
}

#[test]
fn openings_file_with_invalid_line_is_an_error() {
    let path = write_temp_openings(
        "badline",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1\nthis is not a fen\n",
    );
    let params = openings_params(path.clone());
    let err = generate_games(&params).err().expect("invalid opening line must fail loudly");
    std::fs::remove_file(&path).ok();
    assert!(err.contains("line 2"), "error should cite line 2: {err}");
}

#[test]
fn games_start_from_openings_suite_positions() {
    let suite = [
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1",
    ];
    let path = write_temp_openings("valid", &suite.join("\n"));
    let params = openings_params(path.clone());
    let games = generate_games(&params).expect("valid openings suite");
    std::fs::remove_file(&path).ok();
    assert_eq!(4, games.len());
    for game in &games {
        assert!(
            suite.contains(&game.start_fen.as_str()),
            "start_fen {} not from suite",
            game.start_fen
        );
    }
}
