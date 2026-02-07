#![cfg(feature = "board-pleco")]
use pleco::BitMove;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub depth: u32,
    pub score: i32,
    pub best: Option<BitMove>,
    pub bound: Bound,
    pub gen: u32,
}

const WAYS: usize = 4;

#[derive(Default, Clone, Copy)]
struct Slot(Option<Entry>);

#[derive(Default)]
struct Bucket {
    slots: [Slot; WAYS],
}

#[derive(Default)]
pub struct TtPleco {
    buckets: Vec<Mutex<Bucket>>,
    gen: std::sync::atomic::AtomicU32,
}

impl TtPleco {
    pub fn new() -> Self {
        Self {
            buckets: Vec::new(),
            gen: std::sync::atomic::AtomicU32::new(0),
        }
    }
    fn ensure(&mut self) {
        if self.buckets.is_empty() {
            self.set_capacity_entries(65_536);
        }
    }
    pub fn set_capacity_entries(&mut self, entries: usize) {
        let buckets = (entries + WAYS - 1) / WAYS;
        self.buckets.clear();
        self.buckets
            .resize_with(buckets, || Mutex::new(Bucket::default()));
    }
    pub fn set_capacity_mb(&mut self, mb: usize) {
        // Approximate entry size ~64 bytes
        let entries = ((mb.saturating_mul(1024) * 1024) / 64).max(WAYS);
        self.set_capacity_entries(entries);
    }
    pub fn get(&self, key: u64) -> Option<Entry> {
        if self.buckets.is_empty() {
            return None;
        }
        let idx = self.bucket_index(key);
        let g = self.buckets[idx].lock().unwrap();
        for s in &g.slots {
            if let Some(e) = s.0 {
                if e.key == key {
                    return Some(e);
                }
            }
        }
        None
    }
    pub fn put(&self, mut e: Entry) {
        if self.buckets.is_empty() {
            return;
        }
        let idx = self.bucket_index(e.key);
        let mut g = self.buckets[idx].lock().unwrap();
        let cur_gen = self.gen.load(std::sync::atomic::Ordering::Relaxed);
        e.gen = cur_gen;
        for s in &mut g.slots {
            if let Some(cur) = s.0 {
                if cur.key == e.key {
                    if e.depth >= cur.depth {
                        s.0 = Some(e);
                    }
                    return;
                }
            }
        }
        for s in &mut g.slots {
            if s.0.is_none() {
                s.0 = Some(e);
                return;
            }
        }
        let mut victim = 0usize;
        let mut keymin = (u32::MAX, u32::MAX);
        for (i, s) in g.slots.iter().enumerate() {
            if let Some(cur) = s.0 {
                let k = (cur.depth, cur.gen);
                if k < keymin {
                    keymin = k;
                    victim = i;
                }
            }
        }
        g.slots[victim].0 = Some(e);
    }
    pub fn bump_generation(&self) {
        let _ = self.gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    fn bucket_index(&self, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        (mixed as usize) % self.buckets.len().max(1)
    }
}
