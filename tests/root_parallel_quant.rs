use cozy_chess::Board;

#[test]
fn root_parallel_quant_bestmove_equals_single_thread() {
    use piebot::eval::nnue::features::halfkp_dim;
    use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
    use piebot::search::alphabeta::{SearchParams, Searcher};
    // Deterministic quant model (all zeros); eval always 0
    let input_dim = halfkp_dim();
    let hidden_dim = 8usize;
    let model = QuantNnue {
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
        b2: vec![0],
    };

    let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3";
    let b = Board::from_fen(fen, false).unwrap();

    let mut s1 = Searcher::default();
    s1.set_use_nnue(true);
    s1.set_eval_blend_percent(100);
    s1.set_nnue_quant_model(model.clone());
    let mut p1 = SearchParams::default();
    p1.depth = 3;
    p1.use_tt = true;
    p1.order_captures = true;
    p1.use_history = true;
    p1.threads = 1;
    let r1 = s1.search_with_params(&b, p1);

    let mut s2 = Searcher::default();
    s2.set_use_nnue(true);
    s2.set_eval_blend_percent(100);
    s2.set_nnue_quant_model(model);
    let mut p2 = p1;
    p2.threads = 4;
    let r2 = s2.search_with_params(&b, p2);

    assert_eq!(
        r2.score_cp, r1.score_cp,
        "score differs between single and multi-thread with quant NNUE at fixed depth"
    );
}
