#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, serde::Deserialize)]
struct PosRec {
    fen: String,
    best: String,
}

fn load_positions() -> Vec<PosRec> {
    if let Ok(path) = std::env::var("PIEBOT_TEST_POSITIONS") {
        if let Ok(f) = File::open(path) {
            let rdr = BufReader::new(f);
            return rdr
                .lines()
                .filter_map(|l| l.ok())
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| serde_json::from_str(&l).ok())
                .collect();
        }
    }
    let path = "tests/data/positions_sample.jsonl";
    let f = File::open(path).expect("open positions_sample.jsonl");
    let rdr = BufReader::new(f);
    rdr.lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}

#[test]
fn pleco_positions_shallow_bestmove() {
    let poses = load_positions();
    for rec in poses.iter() {
        let mut b = PBoard::from_fen(&rec.fen).expect("valid fen");
        let (bm, _sc, _nodes) = {
            let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
            s.set_threads(1);
            s.search_movetime(&mut b, 300, 4)
        };
        let best = bm.map(|m| format!("{}", m));
        assert_eq!(best.as_deref(), Some(rec.best.as_str()), "FEN {}", rec.fen);
    }
}

#[test]
fn pleco_root_smp_scales_nodes() {
    let fen = "startpos";
    let mut b1 = if fen == "startpos" {
        PBoard::start_pos()
    } else {
        PBoard::from_fen(fen).unwrap()
    };
    // 1 thread
    let nodes1 = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
            s.set_threads(1);
            let (_bm, _sc, n) = s.search_movetime(&mut b1.clone(), 300, 5);
            n
        })
    };
    // 4 threads
    let nodes4 = {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let mut s = piebot::search::alphabeta_pleco::PlecoSearcher::default();
            s.set_threads(4);
            let (_bm, _sc, n) = s.search_movetime(&mut b1.clone(), 300, 5);
            n
        })
    };
    assert!(
        nodes4 > nodes1,
        "expected nodes to scale: 1T={} 4T={}",
        nodes1,
        nodes4
    );
}
