// Pleco-based perft using make/unmake (no cloning)
#[cfg(feature = "board-pleco")]
pub fn perft(board: &mut pleco::Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let mut nodes = 0u64;
    let moves = board.generate_moves();
    for mv in moves.iter().copied() {
        board.apply_move(mv);
        nodes += perft(board, depth - 1);
        board.undo_move();
    }
    nodes
}

// Cozy-chess fallback (used when Pleco feature is disabled)
#[cfg(not(feature = "board-pleco"))]
pub fn perft(board: &cozy_chess::Board, depth: u32) -> u64 {
    if depth == 0 { return 1; }
    let mut nodes = 0u64;
    board.generate_moves(|moves| {
        for m in moves {
            let mut child = board.clone();
            child.play(m);
            nodes += perft(&child, depth - 1);
        }
        false
    });
    nodes
}
