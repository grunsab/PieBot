use cozy_chess::Board;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, serde::Deserialize)]
struct PosRec {
    fen: String,
    best: String,
}

fn load_positions() -> Vec<PosRec> {
    // Try env var first
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
    // Fallback to bundled sample
    let path = "tests/data/positions_sample.jsonl";
    let f = File::open(path).expect("open bundled positions_sample.jsonl");
    let rdr = BufReader::new(f);
    rdr.lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect()
}

#[test]
fn positions_single_thread_correct_move() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let poses = load_positions();
    for rec in poses.iter() {
        let b = Board::from_fen(&rec.fen, false).expect("valid fen");
        let mut s = Searcher::default();
        let mut p = SearchParams::default();
        p.depth = 4;
        p.use_tt = true;
        p.order_captures = true;
        p.use_history = true;
        p.threads = 1;
        p.deterministic = true;
        let r = s.search_with_params(&b, p);
        assert_eq!(
            r.bestmove.as_deref(),
            Some(rec.best.as_str()),
            "FEN {}",
            rec.fen
        );
    }
}

#[test]
fn positions_consistent_across_threads() {
    use piebot::search::alphabeta::{SearchParams, Searcher};
    let poses = load_positions();
    for rec in poses.iter() {
        let b = Board::from_fen(&rec.fen, false).expect("valid fen");
        // 1 thread deterministic
        let r1 = {
            let mut s = Searcher::default();
            let mut p = SearchParams::default();
            p.depth = 4;
            p.use_tt = true;
            p.order_captures = true;
            p.use_history = true;
            p.threads = 1;
            p.deterministic = true;
            s.search_with_params(&b, p)
        };
        // 4 threads deterministic (no splits)
        let r4 = {
            let mut s = Searcher::default();
            let mut p = SearchParams::default();
            p.depth = 4;
            p.use_tt = true;
            p.order_captures = true;
            p.use_history = true;
            p.threads = 4;
            p.deterministic = true;
            s.search_with_params(&b, p)
        };
        assert_eq!(r1.bestmove, r4.bestmove, "FEN {}", rec.fen);
    }
}
