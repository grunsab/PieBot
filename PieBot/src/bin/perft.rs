use cozy_chess::Board;
use piebot::perft::perft;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let depth: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let fen_opt = args.get(2).map(|s| s.to_string());
    let board = if let Some(fen) = fen_opt {
        Board::from_fen(&fen, false).expect("Invalid FEN")
    } else {
        Board::default()
    };
    let nodes = perft(&board, depth);
    println!("nodes: {nodes}");
}
