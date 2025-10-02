use std::fs::File;
use std::io::Write;

#[test]
fn nnue_quant_loader_reads_header() {
    use piebot::eval::nnue::loader::QuantNnue;
    let path = "target/nnue_quant_header.nnue";
    let mut f = File::create(path).unwrap();
    // magic PIENNQ01
    f.write_all(b"PIENNQ01").unwrap();
    // version
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // dims: input 12, hidden 32, output 1
    f.write_all(&12u32.to_le_bytes()).unwrap();
    f.write_all(&32u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // scales
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    drop(f);

    let q = QuantNnue::load_quantized(path).unwrap();
    assert_eq!(q.meta.version, 1);
    assert_eq!(q.meta.input_dim, 12);
    assert_eq!(q.meta.hidden_dim, 32);
    assert_eq!(q.meta.output_dim, 1);
}
