use cozy_chess::Board;

pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    board.generate_moves(|moves| {
        for m in moves {
            let mut child = board.clone();
            child.play_unchecked(m);
            nodes += perft(&child, depth - 1);
        }
        false
    });
    nodes
}
