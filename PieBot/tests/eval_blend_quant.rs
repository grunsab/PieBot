use cozy_chess::Board;

#[test]
fn eval_blend_quant_bias_only() {
    use piebot::eval::nnue::features::halfkp_dim;
    use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
    use piebot::search::alphabeta::Searcher;

    // Quant model with constant 50 output: b2=50, others zero
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
        b2: vec![50],
    };

    let b = Board::default();
    let pst = piebot::search::eval::eval_cp(&b);

    let mut s = Searcher::default();
    s.set_use_nnue(true);
    s.set_nnue_quant_model(model);

    s.set_eval_blend_percent(100);
    let q = s.qsearch_eval_cp(&b);
    assert_eq!(
        q, 50,
        "quant bias-only NNUE at blend=100 should output 50 cp"
    );

    s.set_eval_blend_percent(0);
    let q0 = s.qsearch_eval_cp(&b);
    assert_eq!(q0, pst, "blend=0 should equal PST eval");

    s.set_eval_blend_percent(50);
    let q50 = s.qsearch_eval_cp(&b);
    let expected = ((50 + pst) / 2) as i32;
    assert_eq!(q50, expected, "blend=50 should be average of NNUE and PST");
}
