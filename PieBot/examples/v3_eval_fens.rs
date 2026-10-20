//! Print the engine's static evaluation (side to move, centipawns) of each FEN
//! in a file, one per line, for parity checks against train_v3.quant_eval_fen.
//!
//!     cargo run --release --example v3_eval_fens -- NET.nnue FENS.txt
use cozy_chess::{Board, Color};
use piebot::eval::nnue::loader::QuantNnue;
use piebot::eval::nnue::network::QuantNetwork;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let net = QuantNetwork::new(QuantNnue::load_quantized(&args[1]).expect("load net"));
    for fen in std::fs::read_to_string(&args[2]).unwrap().lines().filter(|l| !l.trim().is_empty()) {
        let b = Board::from_fen(fen, false).unwrap();
        let white = net.eval_full(&b);
        println!("{}", if b.side_to_move() == Color::White { white } else { -white });
    }
}
