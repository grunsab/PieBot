#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn counter_move_boosts_ordering_tail() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    // Start position; choose a candidate parent move index and a preferred child move
    let b = PBoard::start_pos();
    let mut s = PlecoSearcher::default();
    // Disable TT-first so ordering is purely by heuristics on tail
    // SAFETY: Field is not public; rely on default tt_first=true and no TT move at start, we'll still test tail [1..] ranking.
    // Pick a child move from the legal list; use the first move as preferred
    let ml: Vec<_> = b.generate_moves().iter().copied().collect();
    assert!(ml.len() > 2);
    let preferred = ml[1]; // a tail move subject to sorting
    let parent_idx = s.debug_move_hist_index(ml[0]);
    let child_idx = s.debug_move_hist_index(preferred);
    // Set counter move mapping: when parent_idx played, prefer 'preferred' child
    s.debug_set_counter(parent_idx, child_idx);
    // Compute ordering for this parent; expect preferred move to be hoisted to first position in tail (index 1 overall)
    let ordered = s.debug_order_for_parent(&b, parent_idx);
    assert_eq!(ordered[1], preferred, "counter-move should move selected child to position 1 in tail");
}

