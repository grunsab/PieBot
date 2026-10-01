use cozy_chess::{Board, GameStatus, Move, Piece};
use piebot::eval::nnue::loader::QuantNnue;
use piebot::search::alphabeta::{SearchParams, Searcher};
use serde_json::Value;
use std::path::Path;

fn fixture() -> Value {
    serde_json::from_str(include_str!("data/lichess_smp_blunders_20261001.json")).unwrap()
}

fn moves(board: &Board) -> Vec<Move> {
    let mut result = Vec::new();
    board.generate_moves(|batch| {
        result.extend(batch);
        false
    });
    result
}

fn permits_mate_in_one(board: &Board, mv: Move) -> bool {
    let mut child = board.clone();
    child.play(mv);
    moves(&child).into_iter().any(|reply| {
        let mut after = child.clone();
        after.play(reply);
        after.status() == GameStatus::Won
    })
}

#[test]
fn historical_blunders_have_legal_immediate_refutations() {
    for case in fixture()["cases"].as_array().unwrap() {
        let mut board = Board::from_fen(case["fen"].as_str().unwrap(), false).unwrap();
        let bad: Move = case["bad_move"].as_str().unwrap().parse().unwrap();
        let reply: Move = case["refutation_uci"].as_str().unwrap().parse().unwrap();
        assert!(moves(&board).contains(&bad), "{}", case["id"]);
        board.play(bad);
        assert!(moves(&board).contains(&reply), "{}", case["id"]);
        if case["refutation_is_mate"].as_bool().unwrap() {
            board.play(reply);
            assert_eq!(board.status(), GameStatus::Won);
        } else {
            assert!(
                matches!(board.piece_on(reply.to), Some(Piece::Queen | Piece::Bishop)),
                "refutation must capture the hanging piece: {}", case["id"]
            );
        }
    }
}

#[test]
fn real_nnue_parallel_search_avoids_recorded_blunders_and_mate_in_one() {
    let fixture = fixture();
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models")
        .join(fixture["model_file"].as_str().unwrap());
    let model = QuantNnue::load_quantized(&model_path).expect("checked-in regression model");
    let mut searcher = Searcher::default();
    searcher.set_nnue_quant_model(model);
    searcher.set_use_nnue(true);
    searcher.set_eval_blend_percent(75);
    for case in fixture["cases"].as_array().unwrap() {
        let history: Vec<Board> = case["history_fens"].as_array().unwrap().iter()
            .map(|fen| Board::from_fen(fen.as_str().unwrap(), false).unwrap())
            .collect();
        let board = history.last().unwrap();
        for _ in 0..2 {
            searcher.set_tt_capacity_mb(64);
            searcher.clear_history();
            searcher.set_position_history(&history);
            let result = searcher.search_with_params(board, SearchParams {
                depth: 8,
                threads: 2,
                deterministic: false,
                use_tt: true,
                order_captures: true,
                use_history: true,
                use_killers: true,
                use_nullmove: true,
                use_lmr: true,
                use_aspiration: true,
                aspiration_window_cp: 35,
                ..SearchParams::default()
            });
            assert_eq!(result.depth, 8, "{}", case["id"]);
            let selected = result.bestmove.expect("legal move");
            assert_ne!(selected, case["bad_move"].as_str().unwrap(), "{}", case["id"]);
            let mv = selected.parse().unwrap();
            assert!(moves(board).contains(&mv), "{}", case["id"]);
            assert!(!permits_mate_in_one(board, mv), "{} selected {selected}", case["id"]);
        }
    }
}
