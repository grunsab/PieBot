use piebot::search::alphabeta::lmr_reduction;

#[test]
fn test_smooth_history_accumulation_and_limits() {
    // 1. Verify LMR scaling with smooth history
    let r_neutral = lmr_reduction(8, 8, 0);
    let r_high = lmr_reduction(8, 8, 3000);
    assert_eq!(r_high, r_neutral.saturating_sub(1));

    let r_low = lmr_reduction(8, 8, -3000);
    assert_eq!(r_low, r_neutral + 1);

    // 2. Modest history scores (e.g. 500) do NOT prematurely flip LMR
    let r_modest = lmr_reduction(8, 8, 500);
    assert_eq!(r_modest, r_neutral);
}
