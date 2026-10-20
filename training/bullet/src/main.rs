//! PieBot arch-v3 trainer on Stockfish binpacks, built on bullet.
//!
//! Subcommands (all take `--key value` options):
//!   train    train a net; checkpoints under --out, each with a float `quantised.bin`
//!   export   convert a checkpoint's `quantised.bin` to PIENNQ03 (engine format)
//!   evalfens print the trainer's own float eval (cp, side to move) for FENs, as JSON
//!   features check bullet's input features against train_v3's reference map
//!   inspect  print counts, filter statistics and sample entries of a binpack

use std::{collections::HashMap, path::Path};

use bullet::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, ModelNode, Shape,
        optimiser::{AdamW, AdamWOptimiser, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainer, ValueTrainerBuilder},
};
use bullet_compiler::tensor::operation::Unary;
use piebot_bullet::*;
use sfbinpack::{ChunkReader, read_chunk_into};

type Trainer = ValueTrainer<AdamWOptimiser, ChessBucketsMirrored, MaterialCount<OUTPUT_BUCKETS>>;

struct Args(HashMap<String, Vec<String>>);

impl Args {
    fn parse(v: &[String]) -> Self {
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        let mut key = String::from("_");
        for a in v {
            if let Some(k) = a.strip_prefix("--") {
                key = k.to_string();
                m.entry(key.clone()).or_default();
            } else {
                m.entry(key.clone()).or_default().push(a.clone());
            }
        }
        Self(m)
    }
    fn get(&self, k: &str, default: &str) -> String {
        self.0.get(k).and_then(|v| v.first().cloned()).unwrap_or_else(|| default.to_string())
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: &str) -> T
    where
        T::Err: std::fmt::Debug,
    {
        self.get(k, default).parse().unwrap_or_else(|e| panic!("--{k}: {e:?}"))
    }
    fn list(&self, k: &str) -> Vec<String> {
        self.0.get(k).cloned().unwrap_or_default()
    }
    fn flag(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
}

fn bce<'a>(output: ModelNode<'a>, target: ModelNode<'a>) -> ModelNode<'a> {
    // Binary cross-entropy on the logit, as train_v3.py: softplus(x) - t * x.
    (output.unary(Unary::Exp) + 1.0).unary(Unary::Log) - target * output
}

fn mse<'a>(output: ModelNode<'a>, target: ModelNode<'a>) -> ModelNode<'a> {
    output.sigmoid().squared_error(target)
}

fn build(buckets: usize, hidden: usize, loss: &str, factoriser: bool) -> Trainer {
    let lay = layout(buckets);
    let nb = num_buckets(&lay);
    let save = vec![
        SavedFormat::id("l0w").transform(move |store, w| {
            if !factoriser {
                return w;
            }
            let f = store.get("l0f").values.f32().repeat(nb);
            w.into_iter().zip(f).map(|(a, b)| a + b).collect()
        }),
        SavedFormat::id("l0b"),
        SavedFormat::id("l1w").transpose(),
        SavedFormat::id("l1b"),
    ];
    let loss_fn = match loss {
        "bce" => bce,
        "mse" => mse,
        other => panic!("--loss must be bce or mse, not {other}"),
    };
    let mut t = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(lay))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&save)
        .loss_fn(loss_fn)
        .build(move |builder, stm, ntm, out_buckets| {
            let mut l0 = builder.new_affine("l0", 768 * nb, hidden);
            // Shared 768-input factoriser (folded into l0w on export).
            let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
            l0.weights = l0.weights + l0f.repeat(nb);
            let l1 = builder.new_affine("l1", 2 * hidden, OUTPUT_BUCKETS);
            let h = l0.forward(stm).screlu().concat(l0.forward(ntm).screlu());
            l1.forward(h).select(out_buckets)
        });
    // Input weights keep bullet's default clip (+-1.98 each for l0w and l0f, so
    // the merged weight fits i16 at QA = 255 with room to spare); output
    // weights must fit i8 at QB = 64 (train_v3's W2_LIMIT = 127 / 64).
    let w2 = 127.0 / QB as f32;
    t.optimiser.set_params_for_weight("l1w", AdamWParams { max_weight: w2, min_weight: -w2, ..Default::default() });
    t
}

fn data_files(a: &Args) -> Vec<String> {
    let mut files = Vec::new();
    for p in a.list("data") {
        let path = Path::new(&p);
        if path.is_dir() {
            let mut v: Vec<String> = std::fs::read_dir(path)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "binpack"))
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    assert!(!files.is_empty(), "--data: no .binpack files");
    files
}

fn filter_from(a: &Args) -> Filter {
    Filter {
        min_ply: a.num("min-ply", "8"),
        skip_check: !a.flag("keep-checks"),
        skip_capture: !a.flag("keep-captures"),
    }
}

fn score_mode(a: &Args) -> ScoreMode {
    match a.get("score", "q").as_str() {
        "q" => ScoreMode::Q,
        "raw" => ScoreMode::Raw,
        s => panic!("--score must be q or raw, not {s}"),
    }
}

fn train(a: &Args) {
    let buckets: usize = a.num("buckets", "16");
    let hidden: usize = a.num("hidden", "1024");
    let mut trainer = build(buckets, hidden, &a.get("loss", "bce"), !a.flag("no-factoriser"));
    let out = a.get("out", "checkpoints");
    let superbatches: usize = a.num("superbatches", "2000");
    let lr0: f32 = a.num("lr", "0.001");
    let final_lr = lr0 * a.num::<f32>("final-lr-frac", "0.03");
    let start: usize = a.num("start-superbatch", "1");
    if let Some(ckpt) = a.0.get("resume").and_then(|v| v.first()) {
        trainer.load_from_checkpoint(ckpt);
        println!("resumed weights and optimiser from {ckpt}; continuing at superbatch {start}");
    }
    let schedule = TrainingSchedule {
        net_id: a.get("net-id", "piebot_v3"),
        eval_scale: EVAL_SCALE as f32,
        steps: TrainingSteps {
            batch_size: a.num("batch", "16384"),
            batches_per_superbatch: a.num("batches-per-superbatch", "6104"),
            start_superbatch: start,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: a.num("wdl", "0.2") },
        lr_scheduler: lr::Warmup {
            inner: lr::CosineDecayLR { initial_lr: lr0, final_lr, final_superbatch: superbatches },
            warmup_batches: a.num("warmup-batches", "1000"),
        },
        save_rate: a.num("save-rate", "20"),
    };
    let threads: usize = a.num("threads", "8");
    let settings = LocalSettings { threads, test_set: None, output_directory: &out, batch_queue_size: 64 };
    let loader = InterleavedBinpackLoader {
        files: data_files(a),
        buffer_positions: a.num::<usize>("buffer-mpos", "16") * 1_000_000,
        threads,
        filter: filter_from(a),
        scores: ScoreMap::new(score_mode(a)),
        seed: a.num("seed", "1"),
    };
    println!("data: {} binpack files", loader.files.len());
    if a.flag("measure-loader") {
        trainer.measure_max_cpu_throughput(&schedule, &settings, &loader);
        return;
    }
    trainer.run(&schedule, &settings, &loader);
}

fn export(a: &Args) {
    let input = a.get("input", "");
    let output = a.get("output", "");
    let net = FloatNet::from_bullet_floats(Path::new(&input), a.num("buckets", "16"), a.num("hidden", "1024"))
        .unwrap_or_else(|e| panic!("{e}"));
    net.write_piennq03_version(Path::new(&output), a.num("file-version", "4")).unwrap();
    println!("wrote {output}");
}

fn evalfens(a: &Args) {
    let buckets: usize = a.num("buckets", "16");
    let hidden: usize = a.num("hidden", "1024");
    let mut trainer = build(buckets, hidden, &a.get("loss", "bce"), !a.flag("no-factoriser"));
    if let Some(ckpt) = a.0.get("checkpoint").and_then(|v| v.first()) {
        trainer.load_from_checkpoint(ckpt);
    }
    // Optionally also save this exact net, so a fresh/random net can be compared.
    if let Some(save) = a.0.get("save").and_then(|v| v.first()) {
        let tmp = format!("{save}.floats");
        trainer.save_quantised(&tmp).unwrap();
        FloatNet::from_bullet_floats(Path::new(&tmp), buckets, hidden).unwrap().write_piennq03(Path::new(save)).unwrap();
        std::fs::remove_file(tmp).ok();
    }
    let fens = std::fs::read_to_string(a.get("fens", "")).unwrap();
    let mut rows = Vec::new();
    for fen in fens.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let raw = trainer.eval_raw_output(fen)[0];
        rows.push(format!("[{:?}, {}]", fen, raw * EVAL_SCALE as f32));
    }
    println!("[{}]", rows.join(",\n "));
}

fn features(a: &Args) {
    let lay = layout(a.num("buckets", "16"));
    let inputs = ChessBucketsMirrored::new(lay);
    let scores = ScoreMap::new(ScoreMode::Raw);
    let max: usize = a.num("positions", "20000");
    let (mut checked, mut bad) = (0usize, 0usize);
    'files: for file in data_files(a) {
        let mut r = std::io::BufReader::new(std::fs::File::open(&file).unwrap());
        let mut chunk = Vec::new();
        while let Ok(true) = read_chunk_into(&mut r, &mut chunk) {
            let mut cr = ChunkReader::default();
            while cr.has_next(&chunk) {
                let e = cr.next(&chunk);
                let board: ChessBoard = to_board(&e, &scores);
                let (mut stm, mut ntm) = (Vec::new(), Vec::new());
                inputs.map_features(&board, |s, n| {
                    stm.push(s);
                    ntm.push(n);
                });
                let [mut w, mut b] = reference_features(&lay, &entry_pieces(&e));
                let white = e.pos.side_to_move().ordinal() == 0;
                if !white {
                    std::mem::swap(&mut w, &mut b);
                }
                for v in [&mut stm, &mut ntm, &mut w, &mut b] {
                    v.sort_unstable();
                }
                if stm != w || ntm != b {
                    bad += 1;
                    if bad <= 3 {
                        eprintln!("mismatch at {}: stm {stm:?} ref {w:?}", e.pos.fen().unwrap_or_default());
                    }
                }
                checked += 1;
                if checked >= max {
                    break 'files;
                }
            }
        }
    }
    println!("{{\"checked\": {checked}, \"mismatches\": {bad}}}");
    assert!(checked > 0 && bad == 0, "feature parity failed");
}

fn inspect(a: &Args) {
    let filter = filter_from(a);
    let show: usize = a.num("show", "5");
    for file in data_files(a) {
        let mut r = std::io::BufReader::new(std::fs::File::open(&file).unwrap());
        let mut chunk = Vec::new();
        let (mut chunks, mut total, mut kept, mut skip, mut check, mut cap, mut early) = (0, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut max_abs = 0i32;
        let mut shown = 0;
        loop {
            match read_chunk_into(&mut r, &mut chunk) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    println!("  stopped after {chunks} chunks: {e:?} (truncated file?)");
                    break;
                }
            }
            chunks += 1;
            let mut cr = ChunkReader::default();
            while cr.has_next(&chunk) {
                let e = cr.next(&chunk);
                total += 1;
                if e.score == SKIP_SCORE {
                    skip += 1;
                } else {
                    max_abs = max_abs.max(i32::from(e.score).abs());
                }
                if e.ply < filter.min_ply {
                    early += 1;
                }
                if e.pos.is_checked(e.pos.side_to_move()) {
                    check += 1;
                }
                if is_capture(&e) {
                    cap += 1;
                }
                if filter.keep(&e) {
                    kept += 1;
                    if shown < show {
                        shown += 1;
                        println!(
                            "  {} | score {} -> q {:.4} | result {} | ply {} | move {}",
                            e.pos.fen().unwrap_or_default(),
                            e.score,
                            binpack_cp_to_q(f64::from(e.score)),
                            e.result,
                            e.ply,
                            e.mv.as_uci()
                        );
                    }
                }
            }
        }
        let bytes = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
        println!(
            "{file}: {chunks} chunks, {total} entries ({:.2} bytes/entry), kept {kept} ({:.1}%), skip-score {skip}, early {early}, in-check {check}, capture {cap}, max |score| {max_abs}",
            bytes as f64 / total.max(1) as f64,
            100.0 * kept as f64 / total.max(1) as f64
        );
    }
}

/// Check a plain binpack against its `.q` twin entry by entry: the `.q` file
/// stores Q * 32000, the plain one QToCentipawns(Q). Confirms the mapping
/// that `--score q` inverts.
fn compare_q(a: &Args) {
    let open = |p: &str| std::io::BufReader::new(std::fs::File::open(p).unwrap());
    let (mut ra, mut rb) = (open(&a.get("plain", "")), open(&a.get("q", "")));
    let (mut ca, mut cb) = (Vec::new(), Vec::new());
    let (mut n, mut off1, mut worst, mut fen_diff) = (0u64, 0u64, 0f64, 0u64);
    let mut ea: Vec<sfbinpack::TrainingDataEntry> = Vec::new();
    let mut eb: Vec<sfbinpack::TrainingDataEntry> = Vec::new();
    loop {
        while ea.len() < 1 {
            if !matches!(read_chunk_into(&mut ra, &mut ca), Ok(true)) { break; }
            ea.extend(sfbinpack::parse_chunk(&ca));
        }
        while eb.len() < ea.len() {
            if !matches!(read_chunk_into(&mut rb, &mut cb), Ok(true)) { break; }
            eb.extend(sfbinpack::parse_chunk(&cb));
        }
        let k = ea.len().min(eb.len());
        if k == 0 { break; }
        for (x, y) in ea.drain(..k).zip(eb.drain(..k)) {
            if x.pos != y.pos { fen_diff += 1; continue; }
            if x.score == SKIP_SCORE || y.score == SKIP_SCORE { continue; }
            n += 1;
            // Compare in Q: near |q| = 1 one cp of the plain file is a tiny step
            // in Q, while one unit of the .q file is a large step in cp.
            let d = (binpack_cp_to_q(f64::from(x.score)) - f64::from(y.score) / 32000.0).abs();
            worst = worst.max(d);
            if d > 0.002 { off1 += 1; }
        }
    }
    println!("{{\"compared\": {n}, \"position_mismatches\": {fen_diff}, \"q_diff_over_0.002\": {off1}, \"worst_q_diff\": {worst:.5}}}");
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = argv.first() else {
        eprintln!("usage: piebot-bullet <train|export|evalfens|features|inspect> [--key value ...]");
        std::process::exit(2);
    };
    let a = Args::parse(&argv[1..]);
    match cmd.as_str() {
        "train" => train(&a),
        "export" => export(&a),
        "evalfens" => evalfens(&a),
        "features" => features(&a),
        "inspect" => inspect(&a),
        "compare-q" => compare_q(&a),
        other => panic!("unknown command {other}"),
    }
}
