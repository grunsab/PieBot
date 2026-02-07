use std::fs::File;
use std::io::Write;

#[test]
fn nnue_eval_bias_only() {
    use piebot::eval::nnue::Nnue;
    let path = "target/nnue_bias_only.nnue";
    let input_dim = 12u32;
    let hidden_dim = 4u32;
    let output_dim = 1u32;
    let mut f = File::create(path).unwrap();
    f.write_all(b"PIENNUE1").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&input_dim.to_le_bytes()).unwrap();
    f.write_all(&hidden_dim.to_le_bytes()).unwrap();
    f.write_all(&output_dim.to_le_bytes()).unwrap();
    // w1 zeros
    for _ in 0..(input_dim * hidden_dim) {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    // b1 zeros
    for _ in 0..hidden_dim {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    // w2 zeros
    for _ in 0..hidden_dim {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    // b2 = 50.0
    f.write_all(&50f32.to_le_bytes()).unwrap();
    drop(f);
    let nn = Nnue::load(path).unwrap();
    let b = cozy_chess::Board::default();
    assert_eq!(nn.evaluate(&b), 50);
}
