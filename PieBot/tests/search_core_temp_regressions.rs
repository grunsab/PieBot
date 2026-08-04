use cozy_chess::Board;
use piebot::eval::nnue::features::halfkp_dim;
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::search::alphabeta_temp::{EvalMode, SearchParams, Searcher};
use piebot::search::eval::{eval_cp, MATE_SCORE};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn deterministic_model() -> QuantNnue {
    let input_dim = halfkp_dim();
    let hidden_dim = 8;
    let mut seed = 0x5EED_CAFE_F00D_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 32) % 15) as i8 - 7
    };
    QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1: (0..hidden_dim * input_dim).map(|_| next()).collect(),
        b1: vec![11, -7, 3, 19, -13, 5, 2, -17],
        w2: vec![3, -2, 5, 1, -4, 6, -1, 2],
        b2: vec![23],
    }
}

fn write_dense_equivalent(path: &Path, model: &QuantNnue) {
    let mut file = File::create(path).expect("create dense test network");
    file.write_all(b"PIENNUE1").unwrap();
    file.write_all(&model.meta.version.to_le_bytes()).unwrap();
    file.write_all(&(model.meta.input_dim as u32).to_le_bytes())
        .unwrap();
    file.write_all(&(model.meta.hidden_dim as u32).to_le_bytes())
        .unwrap();
    file.write_all(&(model.meta.output_dim as u32).to_le_bytes())
        .unwrap();
    for &weight in &model.w1 {
        file.write_all(&(weight as f32).to_le_bytes()).unwrap();
    }
    for &bias in &model.b1 {
        file.write_all(&(bias as f32).to_le_bytes()).unwrap();
    }
    for &weight in &model.w2 {
        file.write_all(&(weight as f32).to_le_bytes()).unwrap();
    }
    for &bias in &model.b2 {
        file.write_all(&(bias as f32).to_le_bytes()).unwrap();
    }
}

fn search_params(depth: u32) -> SearchParams {
    SearchParams {
        depth,
        use_tt: false,
        threads: 1,
        deterministic: true,
        ..SearchParams::default()
    }
}

fn configured_quant(model: QuantNnue) -> Searcher {
    let mut searcher = Searcher::default();
    searcher.set_eval_mode(EvalMode::Nnue);
    searcher.set_use_nnue(true);
    searcher.set_eval_blend_percent(100);
    searcher.set_nnue_quant_model(model);
    searcher
}

fn configured_dense(path: &Path) -> Searcher {
    let mut searcher = Searcher::default();
    searcher.set_eval_mode(EvalMode::Nnue);
    searcher.set_use_nnue(true);
    searcher.set_eval_blend_percent(100);
    searcher.set_nnue_network(Some(
        piebot::eval::nnue::Nnue::load(path).expect("load dense test network"),
    ));
    searcher
}

#[test]
fn interrupted_iteration_keeps_last_completed_depth_result() {
    let board = Board::default();

    let mut depth_one = Searcher::default();
    let expected = depth_one.search_with_params(&board, search_params(1));
    assert_eq!(depth_one.last_depth(), 1);

    let mut limited = Searcher::default();
    let mut params = search_params(5);
    params.max_nodes = Some(expected.nodes + 1);
    let actual = limited.search_with_params(&board, params);

    assert_eq!(
        limited.last_depth(),
        1,
        "partial depth must not be reported"
    );
    assert_eq!(actual.bestmove, expected.bestmove);
    assert_eq!(actual.score_cp, expected.score_cp);
    assert!(actual.nodes <= expected.nodes + 1);
}

#[test]
fn zero_movetime_reports_no_completed_iteration_but_returns_legal_fallback() {
    let board = Board::default();
    let mut searcher = Searcher::default();
    let mut params = search_params(8);
    params.movetime = Some(Duration::ZERO);

    let result = searcher.search_with_params(&board, params);

    assert_eq!(searcher.last_depth(), 0);
    let best = result
        .bestmove
        .expect("search must retain a legal fallback move");
    assert!(
        searcher
            .debug_order_root(&board)
            .iter()
            .any(|mv| format!("{mv}") == best),
        "fallback {best} must be legal"
    );
}

#[test]
fn recursive_quant_search_matches_full_recompute_for_special_move_paths() {
    let model = deterministic_model();
    let path = std::env::temp_dir().join(format!(
        "piebot_temp_dense_parity_{}.nnue",
        std::process::id()
    ));
    write_dense_equivalent(&path, &model);

    let cases = [
        ("quiet", Board::default(), 3),
        (
            "capture",
            Board::from_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", false).unwrap(),
            3,
        ),
        (
            "castle",
            Board::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", false).unwrap(),
            3,
        ),
        (
            "promotion",
            Board::from_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1", false).unwrap(),
            3,
        ),
        (
            "en-passant",
            Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap(),
            3,
        ),
    ];

    for (name, board, depth) in cases {
        let mut incremental = configured_quant(model.clone());
        let mut full = configured_dense(&path);
        let incremental_result = incremental.search_with_params(&board, search_params(depth));
        let full_result = full.search_with_params(&board, search_params(depth));
        assert_eq!(
            incremental_result.bestmove, full_result.bestmove,
            "best move mismatch on {name} path"
        );
        assert_eq!(
            incremental_result.score_cp, full_result.score_cp,
            "score mismatch on {name} path"
        );
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn recursive_quant_search_does_not_leak_accumulator_between_siblings_or_searches() {
    let model = deterministic_model();
    let path = std::env::temp_dir().join(format!(
        "piebot_temp_dense_siblings_{}.nnue",
        std::process::id()
    ));
    write_dense_equivalent(&path, &model);
    let first = Board::default();
    let second = Board::from_fen(
        "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 1",
        false,
    )
    .unwrap();

    let mut incremental = configured_quant(model);
    let _ = incremental.search_with_params(&first, search_params(3));
    let after_siblings = incremental.search_with_params(&second, search_params(3));
    let mut full = configured_dense(&path);
    let expected = full.search_with_params(&second, search_params(3));

    assert_eq!(after_siblings.bestmove, expected.bestmove);
    assert_eq!(after_siblings.score_cp, expected.score_cp);
    std::fs::remove_file(path).ok();
}

#[test]
fn qsearch_scores_checkmate_and_stalemate_as_terminal() {
    let checkmate = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", false).unwrap();
    let stalemate = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", false).unwrap();
    let mut searcher = Searcher::default();

    assert_eq!(searcher.qsearch_eval_cp(&checkmate), -MATE_SCORE);
    assert_eq!(searcher.qsearch_eval_cp(&stalemate), 0);
}

#[test]
fn checkmate_outranks_a_fifty_move_claim_in_qsearch() {
    let checkmate = Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", false).unwrap();
    let mut searcher = Searcher::default();

    assert_eq!(
        searcher.qsearch_eval_cp(&checkmate),
        -MATE_SCORE,
        "a terminal checkmate ends the game before a fifty-move draw can be claimed"
    );
}

#[test]
fn quiet_mate_on_halfmove_one_hundred_is_scored_as_mate() {
    let board = Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 99 1", false).unwrap();
    let mut searcher = Searcher::default();

    let result = searcher.search_with_params(&board, search_params(1));

    assert_eq!(result.bestmove.as_deref(), Some("g6g7"));
    assert!(
        result.score_cp >= MATE_SCORE - 2,
        "quiet checkmate must outrank the resulting halfmove-100 draw claim: {}",
        result.score_cp
    );
}

#[test]
fn qsearch_in_check_searches_quiet_legal_evasions_instead_of_stand_pat() {
    let board = Board::from_fen("k3r3/8/8/8/8/8/8/4K3 w - - 0 1", false).unwrap();
    assert!(!board.checkers().is_empty());
    let stand_pat = eval_cp(&board);
    let mut expected = -MATE_SCORE;
    board.generate_moves(|moves| {
        for mv in moves {
            let mut child = board.clone();
            child.play_unchecked(mv);
            expected = expected.max(-eval_cp(&child));
        }
        false
    });

    let actual = Searcher::default().qsearch_eval_cp(&board);

    assert_eq!(actual, expected);
    assert_ne!(actual, stand_pat, "standing pat while checked is illegal");
}

#[test]
fn qsearch_includes_en_passant_and_promotions() {
    let en_passant = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap();
    let promotion = Board::from_fen("7k/P7/8/8/8/8/8/K7 w - - 0 1", false).unwrap();
    let mut searcher = Searcher::default();

    assert!(
        searcher.qsearch_eval_cp(&en_passant) > eval_cp(&en_passant),
        "en passant capture must be searched"
    );
    assert!(
        searcher.qsearch_eval_cp(&promotion) >= eval_cp(&promotion) + 500,
        "quiet promotion must be searched"
    );
}

#[test]
fn qsearch_obeys_node_budget_and_does_not_complete_partial_depth() {
    let board = Board::from_fen(
        "r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 1",
        false,
    )
    .unwrap();
    let mut searcher = Searcher::default();
    let mut params = search_params(5);
    params.max_nodes = Some(1);

    let result = searcher.search_with_params(&board, params);

    assert_eq!(searcher.last_depth(), 0);
    assert!(result.bestmove.is_some());
    assert!(result.nodes <= 1, "node budget exceeded: {}", result.nodes);
}

#[test]
fn search_returns_draw_for_fifty_move_and_insufficient_material_positions() {
    let fifty_move = Board::from_fen("4k3/8/8/8/8/8/7Q/4K3 w - - 100 1", false).unwrap();
    let insufficient = Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", false).unwrap();

    for (name, board) in [("fifty-move", fifty_move), ("insufficient", insufficient)] {
        let mut searcher = Searcher::default();
        let result = searcher.search_with_params(&board, search_params(3));
        assert_eq!(result.score_cp, 0, "{name} draw must score zero");
        assert!(
            result.bestmove.is_some(),
            "{name} draw still needs a legal move"
        );
    }
}

#[test]
fn search_returns_draw_for_supplied_threefold_history() {
    let board = Board::from_fen("4k3/8/8/8/8/8/7Q/4K3 w - - 8 5", false).unwrap();
    let history = vec![board.clone(), board.clone(), board.clone()];
    let mut searcher = Searcher::default();
    searcher.set_position_history(&history);

    let result = searcher.search_with_params(&board, search_params(3));

    assert_eq!(result.score_cp, 0);
    assert!(result.bestmove.is_some());
}

#[test]
fn external_stop_returns_fallback_without_completing_a_partial_iteration() {
    let board = Board::default();
    let stop = Arc::new(AtomicBool::new(true));
    let mut searcher = Searcher::default();
    searcher.set_stop_flag(Some(stop.clone()));

    let stopped = searcher.search_with_params(&board, search_params(8));

    assert_eq!(searcher.last_depth(), 0);
    assert!(stopped.bestmove.is_some());
    assert_eq!(stopped.nodes, 0);

    stop.store(false, Ordering::Relaxed);
    searcher.clear_stop_flag();
    let resumed = searcher.search_with_params(&board, search_params(1));
    assert_eq!(searcher.last_depth(), 1);
    assert!(resumed.nodes > 0);
}

#[test]
fn completed_search_reports_completed_depth_and_real_selective_depth() {
    let board = Board::default();
    let mut searcher = Searcher::default();

    let result = searcher.search_with_params(&board, search_params(3));

    assert!(result.bestmove.is_some());
    assert_eq!(searcher.last_depth(), 3);
    assert!(
        searcher.last_seldepth() >= 3,
        "seldepth {} must include the nominal search ply",
        searcher.last_seldepth()
    );
}

#[test]
fn parallel_search_aggregates_worker_selective_depth() {
    let board = Board::default();
    let mut searcher = Searcher::default();
    searcher.set_threads(2);
    searcher.set_deterministic(false);

    let result = searcher.search_depth(&board, 4);

    assert!(result.bestmove.is_some());
    assert_eq!(searcher.last_depth(), 4);
    assert!(
        searcher.last_seldepth() >= 4,
        "parallel workers reported seldepth {}",
        searcher.last_seldepth()
    );
}

#[test]
fn dense_nnue_remains_full_recompute_correct_when_threads_are_requested() {
    let model = deterministic_model();
    let path = std::env::temp_dir().join(format!(
        "piebot_temp_dense_thread_safety_{}.nnue",
        std::process::id()
    ));
    write_dense_equivalent(&path, &model);
    let board = Board::default();

    let mut serial = configured_dense(&path);
    let mut serial_params = search_params(4);
    serial_params.deterministic = false;
    let expected = serial.search_with_params(&board, serial_params);

    let mut requested_parallel = configured_dense(&path);
    let mut parallel_params = serial_params;
    parallel_params.threads = 4;
    let actual = requested_parallel.search_with_params(&board, parallel_params);

    assert_eq!(actual.bestmove, expected.bestmove);
    assert_eq!(actual.score_cp, expected.score_cp);
    std::fs::remove_file(path).ok();
}
