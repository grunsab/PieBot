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
    let g1 = generate_games(&params);
    let g2 = generate_games(&params);
    assert_eq!(g1.len(), 2);
    assert_eq!(g2.len(), 2);
    // Deterministic by seed
    assert_eq!(g1[0].moves, g2[0].moves);
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
    let g1 = generate_games(&p);
    p.seed = 2;
    let g2 = generate_games(&p);
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
    let serial = generate_games(&params);
    params.parallel_games = 4;
    let parallel = generate_games(&params);
    assert_eq!(serial.len(), parallel.len());
    for i in 0..serial.len() {
        assert_eq!(serial[i].start_fen, parallel[i].start_fen);
        assert_eq!(serial[i].moves, parallel[i].moves);
        assert_eq!(serial[i].result, parallel[i].result);
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
    let serial = generate_games(&params);
    params.parallel_games = 4;
    let parallel = generate_games(&params);
    assert_eq!(serial.len(), parallel.len());
    for i in 0..serial.len() {
        assert_eq!(serial[i].start_fen, parallel[i].start_fen);
        assert_eq!(serial[i].moves, parallel[i].moves);
        assert_eq!(serial[i].result, parallel[i].result);
    }
}
