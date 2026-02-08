use std::fs::File;
use std::io::Write;

#[test]
fn nnue_quant_loader_reads_header() {
    use piebot::eval::nnue::loader::QuantNnue;
    let path = "target/nnue_quant_header.nnue";
    let input_dim = 12u32;
    let hidden_dim = 32u32;
    let output_dim = 1u32;
    let mut f = File::create(path).unwrap();
    // magic PIENNQ01
    f.write_all(b"PIENNQ01").unwrap();
    // version
    f.write_all(&1u32.to_le_bytes()).unwrap();
    // dims: input 12, hidden 32, output 1
    f.write_all(&input_dim.to_le_bytes()).unwrap();
    f.write_all(&hidden_dim.to_le_bytes()).unwrap();
    f.write_all(&output_dim.to_le_bytes()).unwrap();
    // scales
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    // payload
    for _ in 0..(input_dim * hidden_dim) {
        f.write_all(&[0u8]).unwrap();
    }
    for _ in 0..hidden_dim {
        f.write_all(&0i16.to_le_bytes()).unwrap();
    }
    for _ in 0..(output_dim * hidden_dim) {
        f.write_all(&[0u8]).unwrap();
    }
    for _ in 0..output_dim {
        f.write_all(&0i16.to_le_bytes()).unwrap();
    }
    drop(f);

    let q = QuantNnue::load_quantized(path).unwrap();
    assert_eq!(q.meta.version, 1);
    assert_eq!(q.meta.input_dim, input_dim as usize);
    assert_eq!(q.meta.hidden_dim, hidden_dim as usize);
    assert_eq!(q.meta.output_dim, output_dim as usize);
}

#[test]
fn nnue_quant_loader_rejects_truncated_payload() {
    use piebot::eval::nnue::loader::QuantNnue;
    let path = "target/nnue_quant_truncated.nnue";
    let mut f = File::create(path).unwrap();
    f.write_all(b"PIENNQ01").unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&12u32.to_le_bytes()).unwrap();
    f.write_all(&32u32.to_le_bytes()).unwrap();
    f.write_all(&1u32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    drop(f);
    let loaded = QuantNnue::load_quantized(path);
    assert!(loaded.is_err());
}

#[test]
fn quant_scales_are_applied_in_eval() {
    use cozy_chess::Board;
    use piebot::eval::nnue::features::halfkp_dim;
    use piebot::eval::nnue::loader::{QuantMeta, QuantNnue};
    use piebot::search::alphabeta::{EvalMode, Searcher};

    let input_dim = halfkp_dim();
    let hidden_dim = 4usize;
    let model = QuantNnue {
        meta: QuantMeta {
            version: 1,
            input_dim,
            hidden_dim,
            output_dim: 1,
        },
        // Effective output scale = 0.5 * 0.25 = 0.125
        // With raw int output b2=8, expected cp is 1.
        w1_scale: 0.5,
        w2_scale: 0.25,
        w1: vec![0; hidden_dim * input_dim],
        b1: vec![0; hidden_dim],
        w2: vec![0; hidden_dim],
        b2: vec![8],
    };

    let mut s = Searcher::default();
    s.set_eval_mode(EvalMode::Nnue);
    s.set_use_nnue(true);
    s.set_eval_blend_percent(100);
    s.set_nnue_quant_model(model);

    let b = Board::default();
    let sc = s.qsearch_eval_cp(&b);
    assert_eq!(sc, 1);
}
