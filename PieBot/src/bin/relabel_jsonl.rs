use clap::Parser;
use cozy_chess::{Board, Color};
use piebot::search::alphabeta::{SearchParams, Searcher};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "relabel-jsonl",
    about = "Relabel self-play JSONL with a stronger teacher at higher depth"
)]
struct Args {
    /// Input JSONL file or directory containing *.jsonl shards.
    #[arg(long)]
    input: PathBuf,
    /// Output directory to write relabeled JSONL shards.
    #[arg(long)]
    output: PathBuf,
    /// Teacher search depth for relabeling.
    #[arg(long, default_value_t = 8)]
    depth: u32,
    /// Relabel every Nth ply (periodic relabeling).
    #[arg(long, default_value_t = 4)]
    every: usize,
    /// Teacher search threads.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Teacher TT hash size in MB.
    #[arg(long, default_value_t = 64)]
    hash_mb: usize,
    /// Optional cap on number of records relabeled across all shards.
    #[arg(long, default_value_t = 0)]
    max_records: usize,
}

fn collect_inputs(input: &Path) -> std::io::Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(input)? {
        let path = entry?.path();
        if path.extension().and_then(|x| x.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn teacher_label(
    board: &Board,
    depth: u32,
    threads: usize,
    hash_mb: usize,
) -> Option<(String, f32)> {
    let mut s = Searcher::default();
    s.set_tt_capacity_mb(hash_mb.max(1));
    let mut p = SearchParams::default();
    p.depth = depth.max(1);
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    p.threads = threads.max(1);
    p.use_aspiration = true;
    p.aspiration_window_cp = 50;
    p.use_lmr = true;
    p.use_killers = true;
    p.use_nullmove = true;
    let res = s.search_with_params(board, p);
    let best = res.bestmove?;
    let score_white = if board.side_to_move() == Color::White {
        res.score_cp as f32
    } else {
        -(res.score_cp as f32)
    };
    Some((best, score_white))
}

fn ensure_played_move(v: &mut serde_json::Map<String, Value>) {
    if v.contains_key("played_move") {
        return;
    }
    if let Some(Value::String(best)) = v.get("best_move") {
        v.insert("played_move".to_string(), Value::String(best.clone()));
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)?;
    let inputs = collect_inputs(&args.input)?;
    if inputs.is_empty() {
        anyhow::bail!("no jsonl inputs found at {}", args.input.display());
    }

    let mut relabeled = 0usize;
    let max_records = if args.max_records > 0 {
        Some(args.max_records)
    } else {
        None
    };
    let period = args.every.max(1);

    for in_path in inputs {
        let out_path = args.output.join(
            in_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("bad input filename"))?,
        );
        let rdr = BufReader::new(File::open(&in_path)?);
        let mut wr = BufWriter::new(File::create(&out_path)?);

        for line in rdr.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: Result<Value, _> = serde_json::from_str(&line);
            let mut v = if let Ok(Value::Object(map)) = parsed {
                map
            } else {
                wr.write_all(line.as_bytes())?;
                wr.write_all(b"\n")?;
                continue;
            };

            ensure_played_move(&mut v);

            let can_relabel_more = max_records.map(|m| relabeled < m).unwrap_or(true);
            let fen = v.get("fen").and_then(|x| x.as_str()).map(|s| s.to_string());
            let ply = v.get("ply").and_then(|x| x.as_u64()).unwrap_or(0) as usize;

            if can_relabel_more && ply % period == 0 {
                if let Some(fen_str) = fen {
                    if let Ok(board) = Board::from_fen(&fen_str, false) {
                        if let Some((best, cpw)) =
                            teacher_label(&board, args.depth, args.threads, args.hash_mb)
                        {
                            v.insert("target_best_move".to_string(), Value::String(best.clone()));
                            v.insert("best_move".to_string(), Value::String(best));
                            v.insert("value_cp".to_string(), Value::from(cpw));
                            v.insert("target_value_cp".to_string(), Value::from(cpw));
                            v.insert("teacher_depth".to_string(), Value::from(args.depth as u64));
                            relabeled += 1;
                        }
                    }
                }
            }

            serde_json::to_writer(&mut wr, &v)?;
            wr.write_all(b"\n")?;
        }
        wr.flush()?;
    }

    println!("Relabeled records: {}", relabeled);
    Ok(())
}
