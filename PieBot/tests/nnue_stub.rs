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
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    drop(f);
    let nn = Nnue::load(path).unwrap();
    let b = Board::default();
    assert_eq!(nn.evaluate(&b), 0);
}
