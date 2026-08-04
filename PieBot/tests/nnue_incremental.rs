use cozy_chess::{Board, Move};
use piebot::eval::nnue::features::{halfkp_dim, halfkp_v2_dim, HalfKpA};
use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
use piebot::eval::nnue::network::QuantNetwork;
use piebot::search::alphabeta::{EvalMode, Searcher};
use std::collections::HashSet;

fn lcg_next(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*seed >> 32) as u32
}

fn lcg_i8(seed: &mut u64) -> i8 {
    (lcg_next(seed) % 15) as i8 - 7
}

fn lcg_i16(seed: &mut u64) -> i16 {
    (lcg_next(seed) % 41) as i16 - 20
}

fn make_quant_model_for(input_dim: usize, hidden_dim: usize, bias: i16) -> QuantNnue {
    let mut seed = 0xC0FFEE_u64;

    let w1 = (0..hidden_dim * input_dim)
        .map(|_| lcg_i8(&mut seed))
        .collect();
    let b1 = (0..hidden_dim).map(|_| lcg_i16(&mut seed)).collect();
    let w2 = (0..hidden_dim).map(|_| lcg_i8(&mut seed)).collect();

    QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1,
        b1,
        w2,
        b2: vec![bias],
    }
}

fn make_quant_model(hidden_dim: usize, bias: i16) -> QuantNnue {
    make_quant_model_for(halfkp_dim(), hidden_dim, bias)
}

fn make_bias_only_quant_model(hidden_dim: usize, bias: i16) -> QuantNnue {
    let input_dim = halfkp_dim();
    QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim,
            output_dim: 1,
        },
        w1_scale: 1.0,
        w2_scale: 1.0,
        w1: vec![0; hidden_dim * input_dim],
        b1: vec![0; hidden_dim],
        w2: vec![0; hidden_dim],
        b2: vec![bias],
    }
}

fn find_move_uci(board: &Board, uci: &str) -> Move {
    let mut found = None;
    board.generate_moves(|ml| {
        for m in ml {
            if format!("{}", m) == uci {
                found = Some(m);
                break;
            }
        }
        found.is_some()
    });
    found.unwrap_or_else(|| panic!("move {uci} not legal in {}", board))
}

fn assert_incremental_parity(start_fen: &str, seq: &[&str]) {
    let mut board = Board::from_fen(start_fen, false).expect("valid start FEN");
    let mut net = QuantNetwork::new(make_quant_model(8, 0));
    net.refresh(&board);

    assert_eq!(
        net.eval_current(),
        net.eval_full(&board),
        "initial incremental/full mismatch"
    );

    for uci in seq {
        let mv = find_move_uci(&board, uci);
        let before_eval = net.eval_current();
        let mut after = board.clone();
        after.play_unchecked(mv);

        let change = net.apply_move(&board, mv, &after);
        assert_eq!(
            net.eval_current(),
            net.eval_full(&after),
            "incremental/full mismatch after move {uci}"
        );

        net.revert(change);
        assert_eq!(
            net.eval_current(),
            before_eval,
            "revert did not restore eval for move {uci}"
        );
        assert_eq!(
            net.eval_current(),
            net.eval_full(&board),
            "revert mismatch with full recompute for move {uci}"
        );

        board = after;
        net.refresh(&board);
    }
}

fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::new();
    board.generate_moves(|list| {
        moves.extend(list);
        false
    });
    moves
}

fn assert_long_walk_and_unwind(input_dim: usize, seed: u64) {
    let mut board = Board::default();
    let mut net = QuantNetwork::new(make_quant_model_for(input_dim, 8, 0));
    net.refresh(&board);

    let mut rng = seed;
    let mut boards = vec![board.clone()];
    let mut changes = Vec::new();

    for ply in 0..192 {
        let moves = legal_moves(&board);
        if moves.is_empty() {
            break;
        }
        let mv = moves[(lcg_next(&mut rng) as usize) % moves.len()];
        let mut after = board.clone();
        after.play_unchecked(mv);
        let change = net.apply_move(&board, mv, &after);
        assert_eq!(
            net.eval_current(),
            net.eval_full(&after),
            "incremental/full mismatch at ply {ply} after {mv}"
        );
        changes.push(change);
        boards.push(after.clone());
        board = after;
    }

    assert!(changes.len() >= 64, "deterministic walk ended too early");
    for ply in (0..changes.len()).rev() {
        net.revert(changes.pop().unwrap());
        assert_eq!(
            net.eval_current(),
            net.eval_full(&boards[ply]),
            "incremental/full mismatch while unwinding ply {ply}"
        );
    }
}

fn assert_all_siblings_restore(start_fen: &str, input_dim: usize) {
    let board = Board::from_fen(start_fen, false).expect("valid sibling FEN");
    let mut net = QuantNetwork::new(make_quant_model_for(input_dim, 8, 0));
    net.refresh(&board);
    let root_eval = net.eval_current();

    for mv in legal_moves(&board) {
        let mut child = board.clone();
        child.play_unchecked(mv);
        let change = net.apply_move(&board, mv, &child);
        assert_eq!(
            net.eval_current(),
            net.eval_full(&child),
            "sibling incremental/full mismatch after {mv}"
        );
        net.revert(change);
        assert_eq!(
            net.eval_current(),
            root_eval,
            "sibling revert failed for {mv}"
        );
        assert_eq!(net.eval_current(), net.eval_full(&board));
    }
}

#[test]
fn halfkp_indices_are_unique_and_in_range() {
    let b = Board::default();
    let feats = HalfKpA;
    let act = feats.active_indices(&b);

    assert_eq!(
        act.len(),
        30,
        "start position should have 30 active non-king features"
    );
    assert!(
        act.iter().all(|&i| i < halfkp_dim()),
        "feature index out of HalfKP bounds"
    );
    let uniq: HashSet<usize> = act.into_iter().collect();
    assert_eq!(uniq.len(), 30, "active HalfKP indices should be unique");
}

#[test]
fn halfkp_king_move_changes_feature_indices() {
    let before = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", false).unwrap();
    let mv = find_move_uci(&before, "e1d1");
    let mut after = before.clone();
    after.play_unchecked(mv);

    let feats = HalfKpA;
    let bset: HashSet<usize> = feats.active_indices(&before).into_iter().collect();
    let aset: HashSet<usize> = feats.active_indices(&after).into_iter().collect();

    assert_eq!(
        bset.len(),
        1,
        "expected one non-king piece feature before move"
    );
    assert_eq!(
        aset.len(),
        1,
        "expected one non-king piece feature after move"
    );
    assert_ne!(
        bset, aset,
        "king move must re-key HalfKP features even with unchanged piece set"
    );
}

#[test]
fn incremental_parity_castling_and_king_moves() {
    // cozy-chess encodes castling with king source and rook destination squares.
    assert_incremental_parity("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", &["e1h1", "e8a8"]);
}

#[test]
fn incremental_parity_en_passant() {
    assert_incremental_parity("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", &["e5d6"]);
}

#[test]
fn incremental_parity_promotion() {
    assert_incremental_parity("7k/P7/8/8/8/8/8/K7 w - - 0 1", &["a7a8q"]);
}

#[test]
fn incremental_parity_long_nested_walk_and_full_unwind() {
    for (schema, seed) in [
        (halfkp_dim(), 0xD1CE_BA5E_1234_5678),
        (halfkp_v2_dim(), 0xA11C_E5E7_89AB_CDEF),
    ] {
        assert_long_walk_and_unwind(schema, seed);
    }
}

#[test]
fn incremental_parity_across_sibling_moves_and_special_positions() {
    let positions = [
        // Quiets, captures, and both castling choices.
        "r3k2r/ppp2ppp/2npbn2/3qp3/3P4/2N1PN2/PPP2PPP/R2QK2R w KQkq - 2 9",
        // En-passant is one sibling among ordinary moves.
        "4k3/8/8/3pP3/8/8/4P3/4K3 w - d6 0 1",
        // Quiet and capture promotions to every legal promotion piece.
        "1r5k/P7/8/8/8/8/7p/K7 w - - 0 1",
    ];
    for input_dim in [halfkp_dim(), halfkp_v2_dim()] {
        for fen in positions {
            assert_all_siblings_restore(fen, input_dim);
        }
    }
}

#[test]
fn nnue_eval_flips_with_side_to_move() {
    // Bias-only model makes expected score deterministic.
    let model = make_bias_only_quant_model(8, 120);
    let mut s = Searcher::default();
    s.set_eval_mode(EvalMode::Nnue);
    s.set_use_nnue(true);
    s.set_eval_blend_percent(100);
    s.set_nnue_quant_model(model);

    // Keep a pawn on the board so the search correctly does not adjudicate the
    // fixture as an insufficient-material draw before evaluating it.
    let white = Board::from_fen("8/8/8/8/8/8/P7/K6k w - - 0 1", false).unwrap();
    let black = Board::from_fen("8/8/8/8/8/8/P7/K6k b - - 0 1", false).unwrap();

    assert_eq!(s.qsearch_eval_cp(&white), 120);
    assert_eq!(s.qsearch_eval_cp(&black), -120);
}
