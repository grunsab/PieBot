use cozy_chess::{Board, Move};
use piebot::search::alphabeta::{SearchParams, Searcher};
use piebot::search::draw::is_repetition_since;

/// White is a queen and two rooks down and about to be mated; its only
/// salvation is the perpetual Qe8+ Kh7 Qh5+ Kg8, every reply forced.
const PERPETUAL: &str = "6k1/6p1/5p2/7Q/8/1r6/r1q3PP/7K w - - 0 1";

fn plain(depth: u32) -> SearchParams {
    SearchParams {
        depth,
        use_tt: false,
        threads: 1,
        deterministic: true,
        ..SearchParams::default()
    }
}

fn full(depth: u32) -> SearchParams {
    SearchParams {
        depth,
        use_tt: true,
        order_captures: true,
        use_history: true,
        threads: 1,
        use_aspiration: true,
        aspiration_window_cp: 35,
        use_lmr: true,
        use_killers: true,
        use_nullmove: true,
        deterministic: true,
        ..SearchParams::default()
    }
}

#[test]
fn perpetual_check_is_a_draw_at_the_first_repetition_inside_the_search() {
    // The line returns to the root position after four plies. Waiting for a
    // true threefold needs eight. Checks are not extended, so depth 4 and 5
    // reach the first return but not the second.
    let board = Board::from_fen(PERPETUAL, false).unwrap();
    for (name, params) in [
        ("plain depth 4", plain(4)),
        ("plain depth 5", plain(5)),
        ("full depth 4", full(4)),
        ("full depth 5", full(5)),
    ] {
        let mut searcher = Searcher::default();
        let result = searcher.search_with_params(&board, params);
        assert_eq!(result.score_cp, 0, "{name}: the perpetual must score as a draw");
        assert_eq!(
            result.bestmove.as_deref(),
            Some("h5e8"),
            "{name}: the drawing move is Qe8+"
        );
    }
}

#[test]
fn deeper_search_still_holds_the_perpetual() {
    let board = Board::from_fen(PERPETUAL, false).unwrap();
    let mut searcher = Searcher::default();
    let result = searcher.search_with_params(&board, full(7));
    assert_eq!(result.score_cp, 0);
    assert_eq!(result.bestmove.as_deref(), Some("h5e8"));
}

/// Positions of the knight shuffle Nf3 Nf6 Ng1 Ng8 ..., start position first.
fn shuffle(plies: usize) -> Vec<Board> {
    let cycle = ["g1f3", "g8f6", "f3g1", "f6g8"];
    let mut board = Board::default();
    let mut history = vec![board.clone()];
    for ply in 0..plies {
        let mv: Move = cycle[ply % 4].parse().unwrap();
        board.play(mv);
        history.push(board.clone());
    }
    history
}

#[test]
fn a_second_occurrence_counts_only_at_or_after_the_floor() {
    // After four plies the start position has occurred twice: index 0 and 4.
    let history = shuffle(4);
    let current = history.last().unwrap().clone();
    assert!(is_repetition_since(&current, &history, 0));
    // The earlier occurrence lies before the floor: not a repetition yet.
    assert!(!is_repetition_since(&current, &history, 1));
    assert!(!is_repetition_since(&current, &history, 4));
    // A floor past the end can never match.
    assert!(!is_repetition_since(&current, &history, 9));
}

#[test]
fn a_position_seen_for_the_first_time_is_not_a_repetition() {
    let history = shuffle(3);
    let current = history.last().unwrap().clone();
    assert!(!is_repetition_since(&current, &history, 0));
}

#[test]
fn an_irreversible_move_ends_the_repetition_window() {
    // Shuffle back to the start position, then push a pawn: nothing before
    // the pawn move can ever recur, whatever the floor says.
    let mut history = shuffle(4);
    let mut board = history.last().unwrap().clone();
    board.play("e2e4".parse().unwrap());
    history.push(board.clone());
    assert!(!is_repetition_since(&board, &history, 0));
}
