use cozy_chess::Move;
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
    pub best: Option<Move>,
    pub bound: Bound,
    pub gen: u32,
}

const DEFAULT_WAYS: usize = 4;

#[derive(Default, Clone, Copy)]
struct Slot(Option<Entry>);

#[derive(Default)]
struct Bucket {
    slots: [Slot; DEFAULT_WAYS],
}

#[derive(Default)]
pub struct Tt {
    buckets: Vec<Mutex<Bucket>>,
    gen: std::sync::atomic::AtomicU32,
}

impl Tt {
    pub fn new() -> Self {
        Self {
            buckets: Vec::new(),
            gen: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn ensure_init(&mut self) {
        if self.buckets.is_empty() {
            // Use a larger default to reduce contention and collisions under SMP
            self.set_capacity_entries(65_536);
        }
    }

    pub fn clear(&mut self) {
        self.ensure_init();
        for b in &self.buckets {
            let mut g = b.lock().unwrap();
            *g = Bucket::default();
        }
    }

    fn bucket_index(&self, key: u64) -> usize {
        let mixed = key ^ (key >> 32);
        (mixed as usize) % self.buckets.len().max(1)
    }

    pub fn get(&self, key: u64) -> Option<Entry> {
        if self.buckets.is_empty() {
            return None;
        }
        let idx = self.bucket_index(key);
        let g = self.buckets[idx].lock().unwrap();
        for slot in &g.slots {
            if let Some(e) = slot.0 {
                if e.key == key {
                    return Some(e);
                }
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        if self.buckets.is_empty() {
            return 0;
        }
        let mut count = 0;
        for b in &self.buckets {
            let g = b.lock().unwrap();
            for s in &g.slots {
                if s.0.is_some() {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn set_capacity_entries(&mut self, cap: usize) {
        let entries = cap.max(DEFAULT_WAYS);
        let buckets = (entries + DEFAULT_WAYS - 1) / DEFAULT_WAYS;
        self.buckets.clear();
        self.buckets
            .resize_with(buckets, || Mutex::new(Bucket::default()));
    }

    pub fn set_capacity_mb(&mut self, mb: usize) {
        // Heuristic: ~64 bytes per entry
        let entries = ((mb.saturating_mul(1024) * 1024) / 64).max(DEFAULT_WAYS);
        self.set_capacity_entries(entries);
    }

    pub fn put(&self, e: Entry) {
        // Safety: we only mutate internal bucket; external API remains &self
        if self.buckets.is_empty() {
            return;
        }
        let idx = self.bucket_index(e.key);
        let mut g = self.buckets[idx].lock().unwrap();
        let cur_gen = self.gen.load(std::sync::atomic::Ordering::Relaxed);
        let mut e = e;
        e.gen = cur_gen;
        // Replace same key if deeper
        for slot in &mut g.slots {
            if let Some(cur) = slot.0 {
                if cur.key == e.key {
                    if e.depth >= cur.depth {
                        slot.0 = Some(e);
                    }
                    return;
                }
            }
        }
        // Empty slot first
        for slot in &mut g.slots {
            if slot.0.is_none() {
                slot.0 = Some(e);
                return;
            }
        }
        // Replace lowest depth
        let mut victim = 0usize;
        let mut best_key = (u32::MAX, u32::MAX);
        for (i, slot) in g.slots.iter().enumerate() {
            if let Some(cur) = slot.0 {
                let key = (cur.depth, cur.gen); // lexicographic: prefer evicting lowest depth, then oldest gen
                if key < best_key {
                    best_key = key;
                    victim = i;
                }
            }
        }
        g.slots[victim].0 = Some(e);
    }

    pub fn bump_generation(&self) {
        let _ = self.gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}
