use std::fs::File;
use std::io::Write;

#[test]
fn nnue_loader_reads_header() {
    use piebot::eval::nnue::Nnue;
    let path = "target/nnue_stub.nnue";
    let mut f = File::create(path).unwrap();
    let input_dim = 12u32;
    let hidden_dim = 4u32;
    let output_dim = 1u32;
    // Write magic and header
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
    for _ in 0..(output_dim * hidden_dim) {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    for _ in 0..output_dim {
        f.write_all(&0f32.to_le_bytes()).unwrap();
    }
    drop(f);
    let nn = Nnue::load(path).unwrap();
    assert_eq!(nn.meta.version, 1);
    assert_eq!(nn.meta.input_dim, input_dim as usize);
    assert_eq!(nn.meta.hidden_dim, hidden_dim as usize);
    assert_eq!(nn.meta.output_dim, output_dim as usize);
}

#[test]
fn nnue_loader_rejects_truncated_payload() {
    use piebot::eval::nnue::Nnue;
    let path = "target/nnue_stub_truncated.nnue";
    let mut f = File::create(path).unwrap();
    f.write_all(b"PIENNUE1").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&12u32.to_le_bytes()).unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    drop(f);
    let loaded = Nnue::load(path);
    assert!(loaded.is_err());
}
