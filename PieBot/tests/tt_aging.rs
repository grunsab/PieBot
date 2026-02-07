use piebot::search::tt::{Bound, Entry, Tt};

#[test]
fn aging_eviction_prefers_oldest_when_depth_equal() {
    let mut tt = Tt::new();
    // Single bucket with 4 ways
    tt.set_capacity_entries(4);
    // Insert 4 entries at same depth, bump gen to create age spread
    tt.put(Entry {
        key: 1,
        depth: 5,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    tt.bump_generation();
    tt.put(Entry {
        key: 2,
        depth: 5,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    tt.bump_generation();
    tt.put(Entry {
        key: 3,
        depth: 5,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    tt.bump_generation();
    tt.put(Entry {
        key: 4,
        depth: 5,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    // New entry at same depth should evict the oldest (key=1)
    tt.bump_generation();
    tt.put(Entry {
        key: 99,
        depth: 5,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    assert!(
        tt.get(1).is_none(),
        "oldest entry not evicted at equal depth"
    );
    assert!(tt.get(99).is_some(), "new entry not inserted");
}
