use piebot::search::tt::{Bound, Entry, Tt};

#[test]
fn tt_capacity_enforced() {
    let mut tt = Tt::new();
    tt.set_capacity_entries(8);
    for i in 0..64u64 {
        let e = Entry {
            key: i,
            depth: (i % 4) as u32,
            score: i as i32,
            best: None,
            bound: Bound::Exact,
            gen: 0,
        };
        tt.put(e);
    }
    // Capacity enforced
    assert!(tt.len() <= 8, "tt size {} exceeds capacity", tt.len());
}

#[test]
fn tt_depth_preferred_on_eviction() {
    let mut tt = Tt::new();
    tt.set_capacity_entries(2);
    // Insert a deep entry we want to keep
    tt.put(Entry {
        key: 1,
        depth: 6,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    // Fill capacity
    tt.put(Entry {
        key: 2,
        depth: 1,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    // Trigger eviction with a shallow entry; deep one should remain
    tt.put(Entry {
        key: 3,
        depth: 1,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    assert!(tt.get(1).is_some(), "deep entry evicted unexpectedly");
}
