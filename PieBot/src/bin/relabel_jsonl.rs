use clap::Parser;
use cozy_chess::{Board, Color};
use piebot::eval::nnue::loader::QuantNnue;
use piebot::search::alphabeta::{EvalMode, SearchParams, Searcher};
use rayon::prelude::*;
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const RELABEL_BATCH_LINES: usize = 4096;

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
    /// Relabel worker threads (parallel positions/files).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Teacher TT hash size in MB.
    #[arg(long, default_value_t = 64)]
    hash_mb: usize,
    /// Optional cap on number of records relabeled across all shards.
    #[arg(long, default_value_t = 0)]
    max_records: usize,
    /// Optional quantized NNUE model used by the relabel teacher search.
    #[arg(long)]
    nnue_quant_file: Option<PathBuf>,
    /// Eval blend percent (0..100) when NNUE is enabled.
    #[arg(long, default_value_t = 100)]
    nnue_blend_percent: u8,
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
    searcher: &mut Searcher,
    board: &Board,
    params: SearchParams,
) -> Option<(String, f32)> {
    let res = searcher.search_with_params(board, params);
    let best = res.bestmove?;
    let score_white = if board.side_to_move() == Color::White {
        res.score_cp as f32
    } else {
        -(res.score_cp as f32)
    };
    Some((best, score_white))
}

fn build_teacher_search_params(depth: u32) -> SearchParams {
    let mut p = SearchParams::default();
    p.depth = depth.max(1);
    p.use_tt = true;
    p.order_captures = true;
    p.use_history = true;
    // Batch relabel throughput is better when we parallelize over many positions
    // and keep each individual teacher search single-threaded.
    p.threads = 1;
    p.use_aspiration = true;
    p.aspiration_window_cp = 50;
    p.use_lmr = true;
    p.use_killers = true;
    p.use_nullmove = true;
    p
}

fn build_teacher_searcher(
    hash_mb: usize,
    nnue_quant_model: Option<&QuantNnue>,
    nnue_blend_percent: u8,
) -> Searcher {
    let mut teacher = Searcher::default();
    teacher.set_tt_capacity_mb(hash_mb.max(1));
    if let Some(model) = nnue_quant_model {
        teacher.set_use_nnue(true);
        teacher.set_eval_mode(EvalMode::Nnue);
        teacher.set_eval_blend_percent(nnue_blend_percent);
        teacher.set_nnue_quant_model(model.clone());
    }
    teacher
}

struct BatchLine {
    original: String,
    parsed: Option<serde_json::Map<String, Value>>,
    should_relabel: bool,
}

fn process_batch(
    pool: &rayon::ThreadPool,
    lines: Vec<String>,
    period: usize,
    remaining_limit: Option<usize>,
    teacher_params: SearchParams,
    teacher_depth: u32,
    hash_mb: usize,
    nnue_quant_model: Option<&QuantNnue>,
    nnue_blend_percent: u8,
) -> (Vec<String>, usize) {
    let mut scheduled = 0usize;
    let tasks: Vec<BatchLine> = lines
        .into_iter()
        .map(|line| {
            let parsed: Result<Value, _> = serde_json::from_str(&line);
            let mut map = if let Ok(Value::Object(map)) = parsed {
                map
            } else {
                return BatchLine {
                    original: line,
                    parsed: None,
                    should_relabel: false,
                };
            };
            ensure_played_move(&mut map);
            let ply = map.get("ply").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let allowed = remaining_limit.map(|m| scheduled < m).unwrap_or(true);
            let should_relabel = allowed && ply % period == 0;
            if should_relabel {
                scheduled += 1;
            }
            BatchLine {
                original: line,
                parsed: Some(map),
                should_relabel,
            }
        })
        .collect();

    let processed: Vec<(String, usize)> = pool.install(|| {
        tasks.into_par_iter()
            .map_init(
                || build_teacher_searcher(hash_mb, nnue_quant_model, nnue_blend_percent),
                |teacher, task| {
                    let Some(mut map) = task.parsed else {
                        return (task.original, 0usize);
                    };
                    if task.should_relabel {
                        let fen = map.get("fen").and_then(|x| x.as_str());
                        if let Some(fen_str) = fen {
                            if let Ok(board) = Board::from_fen(fen_str, false) {
                                if let Some((best, cpw)) = teacher_label(teacher, &board, teacher_params) {
                                    map.insert("target_best_move".to_string(), Value::String(best.clone()));
                                    map.insert("best_move".to_string(), Value::String(best));
                                    map.insert("value_cp".to_string(), Value::from(cpw));
                                    map.insert("target_value_cp".to_string(), Value::from(cpw));
                                    map.insert(
                                        "teacher_depth".to_string(),
                                        Value::from(teacher_depth as u64),
                                    );
                                    let out = serde_json::to_string(&map).unwrap_or(task.original);
                                    return (out, 1usize);
                                }
                            }
                        }
                    }
                    let out = serde_json::to_string(&map).unwrap_or(task.original);
                    (out, 0usize)
                },
            )
            .collect()
    });

    let relabeled = processed.iter().map(|(_, n)| *n).sum();
    let out_lines = processed.into_iter().map(|(line, _)| line).collect();
    (out_lines, relabeled)
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
    let nnue_quant_model = if let Some(path) = args.nnue_quant_file.as_ref() {
        Some(QuantNnue::load_quantized(path)?)
    } else {
        None
    };
    let period = args.every.max(1);
    let teacher_params = build_teacher_search_params(args.depth);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.max(1))
        .build()?;

    for in_path in inputs {
        let out_path = args.output.join(
            in_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("bad input filename"))?,
        );
        let rdr = BufReader::new(File::open(&in_path)?);
        let mut wr = BufWriter::new(File::create(&out_path)?);
        let mut batch: Vec<String> = Vec::with_capacity(RELABEL_BATCH_LINES);
        for line in rdr.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            batch.push(line);
            if batch.len() >= RELABEL_BATCH_LINES {
                let remaining_limit = max_records.map(|m| m.saturating_sub(relabeled));
                let (out_lines, batch_relabeled) = process_batch(
                    &pool,
                    std::mem::take(&mut batch),
                    period,
                    remaining_limit,
                    teacher_params,
                    args.depth,
                    args.hash_mb,
                    nnue_quant_model.as_ref(),
                    args.nnue_blend_percent,
                );
                relabeled += batch_relabeled;
                for out_line in out_lines {
                    wr.write_all(out_line.as_bytes())?;
                    wr.write_all(b"\n")?;
                }
            }
        }
        if !batch.is_empty() {
            let remaining_limit = max_records.map(|m| m.saturating_sub(relabeled));
            let (out_lines, batch_relabeled) = process_batch(
                &pool,
                std::mem::take(&mut batch),
                period,
                remaining_limit,
                teacher_params,
                args.depth,
                args.hash_mb,
                nnue_quant_model.as_ref(),
                args.nnue_blend_percent,
            );
            relabeled += batch_relabeled;
            for out_line in out_lines {
                wr.write_all(out_line.as_bytes())?;
                wr.write_all(b"\n")?;
            }
        }
        wr.flush()?;
    }

    println!("Relabeled records: {}", relabeled);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::build_teacher_search_params;

    #[test]
    fn teacher_search_params_clamp_depth_and_use_single_search_thread() {
        let p = build_teacher_search_params(0);
        assert_eq!(1, p.depth);
        assert_eq!(1, p.threads);
        assert!(p.use_tt);
        assert!(p.use_history);
        assert!(p.use_aspiration);
        assert!(p.use_lmr);
        assert!(p.use_killers);
        assert!(p.use_nullmove);
    }
}
