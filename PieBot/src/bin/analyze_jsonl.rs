use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone, Default)]
struct Rec {
    game: usize,
    ply: usize,
    side: String,
    stm: String,
    score_cp: i32,
    bestmove: String,
    fen: String,
    depth: Option<u32>,
    seldepth: Option<u32>,
}

fn load_jsonl(path: &str) -> Vec<Rec> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: open {}: {}", path, e);
            return Vec::new();
        }
    };
    let rdr = BufReader::new(f);
    let mut out = Vec::new();
    for line in rdr.lines().flatten() {
        let l = line.trim();
        if l.is_empty() {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(l) {
            Ok(v) => {
                let game = v.get("game").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let ply = v.get("ply").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let side = v
                    .get("side")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let stm = v
                    .get("stm")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let score_cp = v.get("score_cp").and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                let bestmove = v
                    .get("bestmove")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let fen = v
                    .get("fen")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let depth = v.get("depth").and_then(|x| x.as_u64()).map(|d| d as u32);
                let seldepth = v.get("seldepth").and_then(|x| x.as_u64()).map(|d| d as u32);
                out.push(Rec {
                    game,
                    ply,
                    side,
                    stm,
                    score_cp,
                    bestmove,
                    fen,
                    depth,
                    seldepth,
                });
            }
            Err(_) => { /* skip */ }
        }
    }
    out
}

fn load_pgn_results(path: &str) -> BTreeMap<usize, String> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("warn: open PGN {} failed: {}", path, e);
            return BTreeMap::new();
        }
    };
    let rdr = BufReader::new(f);
    let mut results = BTreeMap::new();
    let mut cur_round: Option<usize> = None;
    for line in rdr.lines().flatten() {
        let l = line.trim();
        if l.starts_with("[Round \"") {
            if let Some(idx) = l.find('"') {
                if let Some(idx2) = l[idx + 1..].find('"') {
                    let r = &l[idx + 1..idx + 1 + idx2];
                    cur_round = r.parse::<usize>().ok();
                }
            }
        }
        if l.starts_with("[Result \"") {
            if let Some(rnd) = cur_round {
                if let Some(idx) = l.find('"') {
                    if let Some(idx2) = l[idx + 1..].find('"') {
                        let res = &l[idx + 1..idx + 1 + idx2];
                        results.insert(rnd, res.to_string());
                    }
                }
            }
        }
    }
    results
}

fn main() {
    // Args: --jsonl <path> [--pgn <path>] [--mate-thresh 25000]
    let mut jsonl_path: Option<String> = None;
    let mut pgn_path: Option<String> = None;
    let mut mate_thresh: i32 = 25_000;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--jsonl" => {
                jsonl_path = it.next();
            }
            "--pgn" => {
                pgn_path = it.next();
            }
            "--mate-thresh" => {
                if let Some(v) = it.next() {
                    mate_thresh = v.parse().unwrap_or(mate_thresh);
                }
            }
            _ => {}
        }
    }
    let path = jsonl_path.expect("--jsonl <path> required");
    let recs = load_jsonl(&path);
    if recs.is_empty() {
        eprintln!("no records parsed");
        return;
    }
    let mut by_game: BTreeMap<usize, Vec<Rec>> = BTreeMap::new();
    for r in recs {
        by_game.entry(r.game).or_default().push(r);
    }
    for v in by_game.values_mut() {
        v.sort_by_key(|r| r.ply);
    }

    let results = match pgn_path {
        Some(ref p) => load_pgn_results(p),
        None => BTreeMap::new(),
    };

    println!("suspicious (mate->draw) positions:");
    let mut found_any = false;
    for (g, moves) in &by_game {
        let res = results.get(g).cloned().unwrap_or_else(|| "?".into());
        if res == "1/2-1/2" || res == "?" {
            // Look for earliest positive mate score
            if let Some(first_mate) = moves.iter().find(|r| r.score_cp >= mate_thresh) {
                found_any = true;
                println!(
                    "game={} result={} ply={} side={} stm={} score={} depth={} fen={} best={}",
                    g,
                    res,
                    first_mate.ply,
                    first_mate.side,
                    first_mate.stm,
                    first_mate.score_cp,
                    first_mate.depth.unwrap_or(0),
                    first_mate.fen,
                    first_mate.bestmove
                );
            }
        }
    }
    if !found_any {
        println!("(none)");
    }
}
