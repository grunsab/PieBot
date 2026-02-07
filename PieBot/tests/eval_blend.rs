use cozy_chess::Board;
use std::fs::File;
use std::io::Write;

#[test]
fn eval_blend_dense_bias_only() {
    use piebot::search::alphabeta::Searcher;
    // Create a dense NNUE file with bias-only output = 50
    let path = "target/blend_bias50.nnue";
    let input_dim = 12u32;
    let hidden_dim = 4u32;
    let output_dim = 1u32;
    let mut f = File::create(path).unwrap();
    f.write_all(b"PIENNUE1").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&input_dim.to_le_bytes()).unwrap();
    f.write_all(&hidden_dim.to_le_bytes()).unwrap();
    f.write_all(&output_dim.to_le_bytes()).unwrap();
    for _ in 0..(input_dim * hidden_dim) {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    for _ in 0..hidden_dim {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    for _ in 0..hidden_dim {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    f.write_all(&50f32.to_le_bytes()).unwrap();
    drop(f);

    let b = Board::default();
    let pst = piebot::search::eval::eval_cp(&b);

    let mut s = Searcher::default();
    s.set_use_nnue(true);
    s.set_eval_blend_percent(100);
    let nn = piebot::eval::nnue::Nnue::load(path).unwrap();
    s.set_nnue_network(Some(nn));
    let q = s.qsearch_eval_cp(&b);
    assert_eq!(
        q, 50,
        "dense bias-only NNUE at blend=100 should output 50 cp"
    );

    s.set_eval_blend_percent(0);
    let q0 = s.qsearch_eval_cp(&b);
    assert_eq!(q0, pst, "blend=0 should equal PST eval");

    s.set_eval_blend_percent(50);
    let q50 = s.qsearch_eval_cp(&b);
    let expected = ((50 + pst) / 2) as i32;
    assert_eq!(q50, expected, "blend=50 should be average of NNUE and PST");
}
