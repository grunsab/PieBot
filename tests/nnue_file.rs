use std::fs::File;
use std::io::Write;

#[test]
fn nnue_loader_reads_header() {
    use piebot::eval::nnue::Nnue;
    let path = "target/nnue_stub.nnue";
    let mut f = File::create(path).unwrap();
    // Write magic and header
    f.write_all(b"PIENNUE1").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&100u32.to_le_bytes()).unwrap();
    f.write_all(&256u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    drop(f);
    let nn = Nnue::load(path).unwrap();
    assert_eq!(nn.meta.version, 1);
    assert_eq!(nn.meta.input_dim, 100);
    assert_eq!(nn.meta.hidden_dim, 256);
    assert_eq!(nn.meta.output_dim, 1);
}
