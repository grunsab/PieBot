#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn continuation_history_boosts_child_ordering() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    let b = PBoard::start_pos();
    let mut s = PlecoSearcher::default();
    let ml: Vec<_> = b.generate_moves().iter().copied().collect();
    assert!(ml.len() > 2);
    let parent = ml[0];
    let parent_idx = s.debug_move_hist_index(parent);
    // Pick a tail child
    let child = ml[2];
    let child_idx = s.debug_move_hist_index(child);
    // Neutralize counter-move mapping and set a strong continuation-history bonus
    s.debug_set_counter(parent_idx, usize::MAX);
    s.debug_set_cont(parent_idx, child_idx, 10_000);
    let ordered = s.debug_order_for_parent(&b, parent_idx);
    assert_eq!(ordered[1], child, "continuation history should hoist chosen child to tail head");
}
