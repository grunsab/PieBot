use piebot::search::tt::{Bound, Entry, Tt};

#[test]
fn bucket_replacement_prefers_deeper() {
    let mut tt = Tt::new();
    // One bucket with 4 ways
    tt.set_capacity_entries(4);
    // Insert 4 shallow entries into same bucket (keys 0..3)
    for i in 0..4u64 {
        tt.put(Entry {
            key: i,
            depth: (i + 1) as u32,
            score: 0,
            best: None,
            bound: Bound::Exact,
            gen: 0,
        });
    }
    // Inserting deeper entry should evict the shallowest (depth=1, key=0)
    tt.put(Entry {
        key: 100,
        depth: 10,
        score: 0,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    assert!(tt.get(0).is_none(), "shallow victim not evicted");
    assert!(tt.get(100).is_some(), "deeper entry not inserted");
}

#[test]
fn bucket_same_key_replaced_if_deeper() {
    let mut tt = Tt::new();
    tt.set_capacity_entries(4);
    tt.put(Entry {
        key: 0,
        depth: 3,
        score: 1,
        best: None,
        bound: Bound::Exact,
        gen: 0,
    });
    // shallower write should not replace
    tt.put(Entry {
        key: 0,
        depth: 2,
        score: 2,
        best: None,
        bound: Bound::Lower,
        gen: 0,
    });
    let e = tt.get(0).unwrap();
    assert_eq!(e.depth, 3);
    assert_eq!(e.score, 1);
    // deeper should replace
    tt.put(Entry {
        key: 0,
        depth: 5,
        score: 3,
        best: None,
        bound: Bound::Upper,
        gen: 0,
    });
    let e2 = tt.get(0).unwrap();
    assert_eq!(e2.depth, 5);
    assert_eq!(e2.score, 3);
}
