use cozy_chess::Board;

#[test]
fn nnue_stub_evaluate_returns_zero() {
    use piebot::eval::nnue::Nnue;
    use std::fs::File;
    use std::io::Write;
    let path = "target/nnue_stub2.nnue";
    let mut f = File::create(path).unwrap();
    f.write_all(b"PIENNUE1").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&12u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // w1[1*12]
    for _ in 0..12 {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    // b1[1]
    f.write_all(&0f32.to_le_bytes()).unwrap();
    // w2[1]
    f.write_all(&0f32.to_le_bytes()).unwrap();
    // b2[1]
    f.write_all(&0f32.to_le_bytes()).unwrap();
    drop(f);
    let nn = Nnue::load(path).unwrap();
    let b = Board::default();
    assert_eq!(nn.evaluate(&b), 0);
}
