use piebot::selfplay::{
    generate_games, AdjudicationVerdict, Adjudicator, GameTermination, SelfPlayParams,
};

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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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
        resign_cp: 900.0,
        resign_plies: 8,
        no_resign_fraction: 0.15,
        draw_adj_cp: 10.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
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

// ---- Adjudicator unit tests (pure state machine, no engine) ----

#[test]
fn adjudicator_resign_fires_after_exact_consecutive_plies() {
    let mut adj = Adjudicator::new(900.0, 3, 0.0, 0, 0);
    assert_eq!(adj.observe(0, Some(950.0)), None);
    assert_eq!(adj.observe(1, Some(1200.0)), None);
    assert_eq!(
        adj.observe(2, Some(901.0)),
        Some(AdjudicationVerdict::ResignWhiteWins)
    );
}

#[test]
fn adjudicator_resign_streak_resets_on_sign_break() {
    let mut adj = Adjudicator::new(900.0, 3, 0.0, 0, 0);
    assert_eq!(adj.observe(0, Some(950.0)), None);
    assert_eq!(adj.observe(1, Some(950.0)), None);
    // dips back inside the window: streak must restart from zero
    assert_eq!(adj.observe(2, Some(100.0)), None);
    assert_eq!(adj.observe(3, Some(950.0)), None);
    assert_eq!(adj.observe(4, Some(950.0)), None);
    assert_eq!(
        adj.observe(5, Some(950.0)),
        Some(AdjudicationVerdict::ResignWhiteWins)
    );
}

#[test]
fn adjudicator_resign_streak_resets_on_missing_value() {
    let mut adj = Adjudicator::new(900.0, 2, 0.0, 0, 0);
    assert_eq!(adj.observe(0, Some(950.0)), None);
    assert_eq!(adj.observe(1, None), None);
    assert_eq!(adj.observe(2, Some(950.0)), None);
    assert_eq!(
        adj.observe(3, Some(950.0)),
        Some(AdjudicationVerdict::ResignWhiteWins)
    );
}

#[test]
fn adjudicator_black_wins_when_white_is_lost() {
    let mut adj = Adjudicator::new(900.0, 2, 0.0, 0, 0);
    assert_eq!(adj.observe(0, Some(-950.0)), None);
    assert_eq!(
        adj.observe(1, Some(-1100.0)),
        Some(AdjudicationVerdict::ResignBlackWins)
    );
}

#[test]
fn adjudicator_sign_flip_restarts_the_streak() {
    let mut adj = Adjudicator::new(900.0, 2, 0.0, 0, 0);
    assert_eq!(adj.observe(0, Some(950.0)), None);
    // flip to the other side: white streak must reset, black streak starts
    assert_eq!(adj.observe(1, Some(-950.0)), None);
    assert_eq!(adj.observe(2, Some(950.0)), None);
    assert_eq!(adj.observe(3, Some(-950.0)), None);
    assert_eq!(
        adj.observe(4, Some(-950.0)),
        Some(AdjudicationVerdict::ResignBlackWins)
    );
}

#[test]
fn adjudicator_draw_waits_for_min_ply() {
    let mut adj = Adjudicator::new(0.0, 0, 10.0, 3, 6);
    // streak completes well before min ply but must not fire early
    for ply in 0..6 {
        assert_eq!(adj.observe(ply, Some(4.0)), None, "fired at ply {}", ply);
    }
    assert_eq!(
        adj.observe(6, Some(4.0)),
        Some(AdjudicationVerdict::AdjudicatedDraw)
    );
}

#[test]
fn adjudicator_draw_streak_resets_when_out_of_band() {
    let mut adj = Adjudicator::new(0.0, 0, 10.0, 3, 0);
    assert_eq!(adj.observe(0, Some(5.0)), None);
    assert_eq!(adj.observe(1, Some(-8.0)), None);
    // |cp| above threshold: streak must restart from zero
    assert_eq!(adj.observe(2, Some(60.0)), None);
    assert_eq!(adj.observe(3, Some(0.0)), None);
    assert_eq!(adj.observe(4, Some(3.0)), None);
    assert_eq!(
        adj.observe(5, Some(-2.0)),
        Some(AdjudicationVerdict::AdjudicatedDraw)
    );
}

#[test]
fn adjudicator_zero_thresholds_disable_adjudication() {
    let mut adj = Adjudicator::new(0.0, 8, 0.0, 40, 0);
    for ply in 0..200 {
        assert_eq!(adj.observe(ply, Some(5000.0)), None);
    }
    let mut adj = Adjudicator::new(0.0, 8, 0.0, 40, 0);
    for ply in 0..200 {
        assert_eq!(adj.observe(ply, Some(0.0)), None);
    }
}

#[test]
fn adjudication_terminations_serialize_snake_case() {
    assert_eq!(GameTermination::Resigned.as_str(), "resigned");
    assert_eq!(GameTermination::AdjudicatedDraw.as_str(), "adjudicated_draw");
    assert_eq!(
        serde_json::to_string(&GameTermination::Resigned).unwrap(),
        "\"resigned\""
    );
    assert_eq!(
        serde_json::to_string(&GameTermination::AdjudicatedDraw).unwrap(),
        "\"adjudicated_draw\""
    );
}

// ---- Adjudication integration tests (engine games) ----

const LOPSIDED_FEN: &str = "8/8/8/8/3k4/8/3P4/3QK3 w - - 0 1";

fn adjudication_params(openings_path: std::path::PathBuf) -> SelfPlayParams {
    SelfPlayParams {
        games: 1,
        max_plies: 40,
        threads: 1,
        parallel_games: 1,
        use_engine: true,
        depth: 1,
        movetime_ms: None,
        seed: 31,
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
        resign_cp: 500.0,
        resign_plies: 2,
        no_resign_fraction: 0.0,
        draw_adj_cp: 0.0,
        draw_adj_plies: 40,
        draw_adj_min_ply: 80,
    }
}

#[test]
fn engine_game_resigns_lopsided_position_for_white() {
    let path = write_temp_openings("resign", LOPSIDED_FEN);
    let params = adjudication_params(path.clone());
    let games = generate_games(&params).expect("selfplay games");
    std::fs::remove_file(&path).ok();
    assert_eq!(1, games.len());
    let game = &games[0];
    assert_eq!(
        "resigned",
        game.termination.as_str(),
        "termination was {:?}",
        game.termination
    );
    assert!(game.outcome_valid, "resignations are real outcomes");
    assert_eq!(1, game.result, "white is a queen up and must win");
    assert!(
        game.moves.len() >= 2,
        "resign needs resign_plies consecutive observations"
    );
}

#[test]
fn no_resign_fraction_one_keeps_lopsided_game_alive() {
    let path = write_temp_openings("no_resign", LOPSIDED_FEN);
    let mut params = adjudication_params(path.clone());
    params.no_resign_fraction = 1.0;
    params.max_plies = 6;
    let games = generate_games(&params).expect("selfplay games");
    std::fs::remove_file(&path).ok();
    assert_eq!(1, games.len());
    assert_ne!(
        "resigned",
        games[0].termination.as_str(),
        "no-resign games must never resign"
    );
}

#[test]
fn adjudication_same_seed_reproduces_identical_games() {
    let path = write_temp_openings("adj_det", LOPSIDED_FEN);
    let mut params = adjudication_params(path.clone());
    params.games = 4;
    // exercise the per-game no-resign decision: it must be seed-stable
    params.no_resign_fraction = 0.5;
    let g1 = generate_games(&params).expect("selfplay games");
    let g2 = generate_games(&params).expect("selfplay games");
    std::fs::remove_file(&path).ok();
    assert_eq!(g1.len(), g2.len());
    for i in 0..g1.len() {
        assert_eq!(g1[i].moves, g2[i].moves);
        assert_eq!(g1[i].result, g2[i].result);
        assert_eq!(g1[i].outcome_valid, g2[i].outcome_valid);
        assert_eq!(g1[i].termination, g2[i].termination);
    }
}
