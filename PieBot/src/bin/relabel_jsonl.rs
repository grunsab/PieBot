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
    /// Aggregate teacher TT hash budget in MB, shared across relabel workers.
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

fn per_worker_hash_mb(total_hash_mb: usize, workers: usize) -> usize {
    (total_hash_mb.max(1) / workers.max(1)).max(1)
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

const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A_PRIME);
    }
    hash
}

fn relabel_phase(map: &serde_json::Map<String, Value>, period: usize) -> usize {
    let period = period.max(1);
    let run_id = map
        .get("run_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    let game_id = map
        .get("game_id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());

    // Preserve the historical ply % period == 0 behavior for records that
    // predate self-play provenance IDs.
    if run_id.is_none() && game_id.is_none() {
        return 0;
    }

    // FNV-1a is deliberately specified here instead of DefaultHasher so the
    // selected phase is stable across Rust versions, processes, and machines.
    let mut hash = fnv1a_update(FNV1A_OFFSET_BASIS, b"piebot-relabel-phase-v1\0");
    if let Some(value) = run_id {
        hash = fnv1a_update(hash, b"run_id\0");
        hash = fnv1a_update(hash, value.as_bytes());
        hash = fnv1a_update(hash, b"\0");
    }
    if let Some(value) = game_id {
        hash = fnv1a_update(hash, b"game_id\0");
        hash = fnv1a_update(hash, value.as_bytes());
        hash = fnv1a_update(hash, b"\0");
    }
    (hash % period as u64) as usize
}

fn should_select_for_relabel(map: &serde_json::Map<String, Value>, period: usize) -> bool {
    let period = period.max(1);
    let ply = map.get("ply").and_then(Value::as_u64).unwrap_or(0) as usize;
    ply % period == relabel_phase(map, period)
}

fn worker_batches<T, F>(items: Vec<T>, max_workers: usize, is_expensive: F) -> Vec<Vec<(usize, T)>>
where
    F: Fn(&T) -> bool,
{
    if items.is_empty() {
        return Vec::new();
    }

    let expensive_count = items.iter().filter(|item| is_expensive(item)).count();
    let worker_count = max_workers
        .max(1)
        .min(expensive_count.max(1))
        .min(items.len());
    let mut batches: Vec<Vec<(usize, T)>> = (0..worker_count).map(|_| Vec::new()).collect();
    let mut next_expensive = 0usize;
    let mut next_inexpensive = 0usize;

    for (index, item) in items.into_iter().enumerate() {
        let target = if is_expensive(&item) {
            let target = next_expensive % worker_count;
            next_expensive += 1;
            target
        } else {
            let target = next_inexpensive % worker_count;
            next_inexpensive += 1;
            target
        };
        batches[target].push((index, item));
    }

    batches
}

fn process_batch_line(
    task: BatchLine,
    teacher: &mut Searcher,
    teacher_params: SearchParams,
    teacher_depth: u32,
) -> (String, usize) {
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
            let allowed = remaining_limit.map(|m| scheduled < m).unwrap_or(true);
            let should_relabel = allowed && should_select_for_relabel(&map, period);
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

    let batches = worker_batches(tasks, pool.current_num_threads(), |task| {
        task.should_relabel
    });
    let grouped: Vec<Vec<(usize, String, usize)>> = pool.install(|| {
        batches
            .into_par_iter()
            .map(|batch| {
                let mut teacher =
                    build_teacher_searcher(hash_mb, nnue_quant_model, nnue_blend_percent);
                batch
                    .into_iter()
                    .map(|(index, task)| {
                        let (line, relabeled) =
                            process_batch_line(task, &mut teacher, teacher_params, teacher_depth);
                        (index, line, relabeled)
                    })
                    .collect()
            })
            .collect()
    });

    let mut processed: Vec<(usize, String, usize)> = grouped.into_iter().flatten().collect();
    processed.sort_unstable_by_key(|(index, _, _)| *index);

    let relabeled = processed.iter().map(|(_, _, n)| *n).sum();
    let out_lines = processed.into_iter().map(|(_, line, _)| line).collect();
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
    let worker_count = args.threads.max(1);
    let worker_hash_mb = per_worker_hash_mb(args.hash_mb, worker_count);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()?;

    eprintln!(
        "Relabel workers: {}, aggregate hash: {} MB, hash per worker: {} MB",
        worker_count,
        args.hash_mb.max(1),
        worker_hash_mb
    );

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
                    worker_hash_mb,
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
                worker_hash_mb,
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
    use super::{
        build_teacher_search_params, build_teacher_searcher, per_worker_hash_mb,
        process_batch_line, relabel_phase, should_select_for_relabel, worker_batches, BatchLine,
    };
    use serde_json::{json, Value};

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

    #[test]
    fn teacher_hash_budget_is_shared_across_workers() {
        assert_eq!(85, per_worker_hash_mb(4096, 48));
        assert_eq!(1, per_worker_hash_mb(1, 48));
        assert_eq!(64, per_worker_hash_mb(64, 0));
    }

    #[test]
    fn expensive_relabel_tasks_are_balanced_across_reusable_workers() {
        let batches = worker_batches((0usize..24).collect(), 4, |item| item % 4 == 0);
        assert_eq!(4, batches.len());

        let expensive_counts: Vec<usize> = batches
            .iter()
            .map(|batch| batch.iter().filter(|(_, item)| *item % 4 == 0).count())
            .collect();
        assert_eq!(6, expensive_counts.iter().sum::<usize>());
        assert!(
            expensive_counts.iter().max().unwrap() - expensive_counts.iter().min().unwrap() <= 1
        );

        let mut indexed: Vec<(usize, usize)> = batches.into_iter().flatten().collect();
        indexed.sort_unstable_by_key(|(index, _)| *index);
        assert_eq!(
            (0usize..24).map(|index| (index, index)).collect::<Vec<_>>(),
            indexed
        );
    }

    #[test]
    fn provenance_phase_is_exact_and_deterministic() {
        let record = json!({
            "run_id": "run-abc",
            "game_id": "game-xyz",
            "ply": 17
        });
        let map = record.as_object().expect("record object");

        assert_eq!(3, relabel_phase(map, 4));
        assert_eq!(relabel_phase(map, 4), relabel_phase(map, 4));
    }

    #[test]
    fn every_position_in_the_same_game_uses_the_same_phase() {
        let phases: Vec<usize> = (0..12)
            .map(|ply| {
                let record = json!({
                    "run_id": "run-same",
                    "game_id": "game-same",
                    "ply": ply,
                    "custom": ply * 3
                });
                relabel_phase(record.as_object().expect("record object"), 4)
            })
            .collect();

        assert!(phases.iter().all(|phase| *phase == phases[0]));
        for ply in 0..12 {
            let record = json!({
                "run_id": "run-same",
                "game_id": "game-same",
                "ply": ply
            });
            assert_eq!(
                ply % 4 == phases[0],
                should_select_for_relabel(record.as_object().expect("record object"), 4)
            );
        }
    }

    #[test]
    fn different_games_can_use_different_phase_offsets() {
        let game_a = json!({"run_id": "run-1", "game_id": "game-a", "ply": 0});
        let game_b = json!({"run_id": "run-1", "game_id": "game-b", "ply": 0});

        assert_ne!(
            relabel_phase(game_a.as_object().expect("record object"), 4),
            relabel_phase(game_b.as_object().expect("record object"), 4)
        );
    }

    #[test]
    fn game_phases_cover_both_side_to_move_parities() {
        let phases: Vec<usize> = (0..64)
            .map(|game| {
                let record = json!({
                    "run_id": "run-balanced",
                    "game_id": format!("game-{game:03}"),
                    "ply": 0
                });
                relabel_phase(record.as_object().expect("record object"), 4)
            })
            .collect();
        let even = phases.iter().filter(|phase| **phase % 2 == 0).count();
        let odd = phases.len() - even;

        assert!(even > 0, "expected white-to-move relabel phases");
        assert!(odd > 0, "expected black-to-move relabel phases");
        assert!(
            even.abs_diff(odd) <= 8,
            "phase parity should be reasonably balanced"
        );
    }

    #[test]
    fn legacy_records_without_provenance_keep_zero_phase_selection() {
        let selected = json!({"ply": 8});
        let skipped = json!({"ply": 9});

        assert_eq!(
            0,
            relabel_phase(selected.as_object().expect("record object"), 4)
        );
        assert!(should_select_for_relabel(
            selected.as_object().expect("record object"),
            4
        ));
        assert!(!should_select_for_relabel(
            skipped.as_object().expect("record object"),
            4
        ));
    }

    #[test]
    fn selected_relabel_preserves_provenance_and_overwrites_teacher_depth() {
        let original = json!({
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "ply": 0,
            "run_id": "run-abc",
            "game_id": "game-xyz",
            "teacher_depth": 2,
            "custom": {"kept": true}
        });
        let task = BatchLine {
            original: original.to_string(),
            parsed: original.as_object().cloned(),
            should_relabel: true,
        };
        let mut teacher = build_teacher_searcher(1, None, 100);
        let (line, relabeled) =
            process_batch_line(task, &mut teacher, build_teacher_search_params(1), 6);
        let output: Value = serde_json::from_str(&line).expect("valid relabeled JSON");

        assert_eq!(relabeled, 1);
        assert_eq!(output["run_id"], "run-abc");
        assert_eq!(output["game_id"], "game-xyz");
        assert_eq!(output["custom"]["kept"], true);
        assert_eq!(output["teacher_depth"], 6);
        assert!(output.get("value_cp").is_some());
    }

    #[test]
    fn unselected_legacy_record_remains_readable_and_keeps_its_teacher_depth() {
        let original = json!({
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "ply": 1,
            "best_move": "e2e4",
            "teacher_depth": 2,
            "legacy_extra": 17
        });
        let task = BatchLine {
            original: original.to_string(),
            parsed: original.as_object().cloned(),
            should_relabel: false,
        };
        let mut teacher = build_teacher_searcher(1, None, 100);
        let (line, relabeled) =
            process_batch_line(task, &mut teacher, build_teacher_search_params(1), 6);
        let output: Value = serde_json::from_str(&line).expect("valid preserved JSON");

        assert_eq!(relabeled, 0);
        assert_eq!(output["teacher_depth"], 2);
        assert_eq!(output["legacy_extra"], 17);
        assert_eq!(output["best_move"], "e2e4");
        assert!(output.get("run_id").is_none());
        assert!(output.get("game_id").is_none());
    }
}
