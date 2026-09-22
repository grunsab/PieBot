use piebot::search::alphabeta_temp::{lmr_reduction, lmr_reduction_improving};

#[test]
fn test_lmr_reduction_table_and_history_modulation() {
    // 1. Shallow depth or early move: no reduction
    assert_eq!(lmr_reduction(1, 1, 0), 0);
    assert_eq!(lmr_reduction(2, 5, 0), 0);
    assert_eq!(lmr_reduction(5, 1, 0), 0);
    assert_eq!(lmr_reduction(5, 2, 0), 0);

    // 2. Base reductions at normal history (0)
    let r_d3_m4 = lmr_reduction(3, 4, 0);
    assert!(r_d3_m4 >= 1, "Expected r >= 1 for d=3, m=4, got {}", r_d3_m4);

    let r_d8_m8 = lmr_reduction(8, 8, 0);
    assert!(r_d8_m8 >= 2, "Expected r >= 2 for d=8, m=8, got {}", r_d8_m8);

    let r_d16_m16 = lmr_reduction(16, 16, 0);
    assert!(r_d16_m16 >= 3, "Expected r >= 3 for d=16, m=16, got {}", r_d16_m16);

    // Monotonicity: higher depth or later move yields >= reduction
    assert!(r_d16_m16 >= r_d8_m8);
    assert!(r_d8_m8 >= r_d3_m4);

    // 3. History modulation with realistic history scores (clamped to [-400, 400])
    // High history: reduce reduction by 1
    let r_high_hist = lmr_reduction(8, 8, 250);
    assert_eq!(r_high_hist, r_d8_m8.saturating_sub(1));

    // Negative history: increase reduction by 1
    let r_low_hist = lmr_reduction(8, 8, -250);
    assert_eq!(r_low_hist, r_d8_m8 + 1);

    // 4. Improving modulation
    let r_not_improving = lmr_reduction_improving(8, 8, 0, false);
    assert_eq!(r_not_improving, r_d8_m8 + 1);
}
