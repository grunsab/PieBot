//! `train-threats`: the pairwise net with threat inputs (PIENNQ03 version 6).
//!
//! Same network, targets, loss, schedule and checkpoints as `train --pairwise`,
//! plus threat features (`lib::threat_features`, the engine's own map) as a
//! second sparse input with its own weights `l0t`:
//!
//!   a = l0w * pieces + l0f * pieces768 + l0t * threats + l0b
//!
//! bullet's `ValueTrainerBuilder` takes one input type, so this uses bullet's
//! lower-level API (as its `examples/advanced` does): separate inputs keep
//! `l0t` a separate weight, clipped to the i8 grid (+-127 / QA) the engine
//! stores it in. Everything else mirrors `ValueTrainer::run` and
//! `build_pairwise` in main.rs.

use bullet::{
    game::{formats::bulletformat::ChessBoard, inputs::SparseInputType, outputs::OutputBuckets},
    trainer::schedule::lr::{self, LrScheduler},
    value::{
        loader::LoadableDataType,
        save::{save_to_checkpoint, write_losses},
    },
};
use bullet::game::outputs::MaterialCount;
use bullet::nn::ModelNode;
use bullet_trainer::{
    model::{DenseInput, InitSettings, ModelDefinition, ModelInputs, ModelInputsMapper, ModelWeights, SavedFormat, SparseInput},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, logger, train},
};
use piebot_bullet::*;
use std::cell::RefCell;

type Inputs = ((((((SparseInput, SparseInput), SparseInput), SparseInput), SparseInput), DenseInput<f32>), DenseInput<f32>);

pub struct Config {
    pub meta: NetMeta,
    pub loss: for<'a> fn(ModelNode<'a>, ModelNode<'a>) -> ModelNode<'a>,
    pub act_l1: f32,
    pub out: String,
    pub net_id: String,
    pub superbatches: usize,
    pub start: usize,
    pub lr0: f32,
    pub final_lr: f32,
    pub warmup_batches: usize,
    pub batch_size: usize,
    pub batches_per_superbatch: usize,
    pub wdl: f32,
    pub save_rate: usize,
    pub threads: usize,
    pub resume: Option<String>,
    pub init_from: Option<String>,
    pub measure_loader: bool,
}

fn inputs(pieces: usize, nnz_pieces: usize) -> ModelInputs<Inputs> {
    ModelInputs::default()
        .add_sparse("stm", (pieces, 1), nnz_pieces)
        .add_sparse("nstm", (pieces, 1), nnz_pieces)
        .add_sparse("stm_threats", (N_THREAT_FEATURES, 1), MAX_THREATS)
        .add_sparse("nstm_threats", (N_THREAT_FEATURES, 1), MAX_THREATS)
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1))
        .add_dense("entry_weights", (1, 1))
}

/// Batch mapper: piece features as `ChessBucketsMirroredExperts` (one expert),
/// threat features, output bucket, and `ValueTrainer`'s target
/// `wdl * result + (1 - wdl) * sigmoid(score / EVAL_SCALE)`.
fn mapper(spec: &ModelInputs<Inputs>, pieces: ChessBucketsMirroredExperts, wdl: f32) -> ModelInputsMapper<ChessBoard> {
    let nnz = pieces.max_active();
    let num = pieces.num_inputs();
    let out = MaterialCount::<OUTPUT_BUCKETS>;
    let rscale = 1.0 / EVAL_SCALE as f32;
    ModelInputsMapper::build(spec, move |pos: &ChessBoard, _step, ((((((stm, ntm), stm_t), ntm_t), buckets), targets), weights)| {
        let mut cnt = 0;
        pieces.map_features(pos, |our, opp| {
            assert!(our < num && opp < num);
            stm[cnt] = our as i32;
            ntm[cnt] = opp as i32;
            cnt += 1;
        });
        if cnt < nnz {
            stm[cnt] = -1;
            ntm[cnt] = -1;
        }
        let (ts, tn) = threat_features(pos);
        for (dst, src) in [(&mut *stm_t, &ts), (&mut *ntm_t, &tn)] {
            for (d, &f) in dst.iter_mut().zip(src.iter()) {
                *d = f as i32;
            }
            if src.len() < MAX_THREATS {
                dst[src.len()] = -1;
            }
        }
        buckets[0] = i32::from(out.bucket(pos));
        weights[0] = 1.0;
        let score = 1.0 / (1.0 + (-rscale * f32::from(pos.score())).exp());
        let result = f32::from(pos.result() as u8) / 2.0;
        targets[0] = wdl * result + (1.0 - wdl) * score;
    })
}

fn saved_format(nb: usize) -> Vec<SavedFormat> {
    vec![
        SavedFormat::id("l0w").transform(move |store, w| {
            let f = store.get("l0f").values.f32().repeat(nb);
            w.into_iter().zip(f).map(|(a, b)| a + b).collect()
        }),
        SavedFormat::id("l0b"),
        SavedFormat::id("l1w").transpose(),
        SavedFormat::id("l1b"),
        SavedFormat::id("l2w").transpose(),
        SavedFormat::id("l2b"),
        SavedFormat::id("l3w").transpose(),
        SavedFormat::id("l3b"),
        // Feature-major [N_THREAT_FEATURES][hidden], like l0w.
        SavedFormat::id("l0t"),
    ]
}

type Loss = for<'a> fn(ModelNode<'a>, ModelNode<'a>) -> ModelNode<'a>;

fn definition(meta: &NetMeta, spec: &ModelInputs<Inputs>, loss_fn: Loss, act_l1: f32) -> ModelDefinition {
    let (l1, l2) = meta.stack.expect("--threats needs --pairwise");
    assert!(meta.threats && meta.experts == 1);
    let hidden = meta.hidden;
    let nb = num_buckets(&layout(meta.buckets));
    ModelDefinition::build(&spec, |builder, ((((((stm, ntm), stm_t), ntm_t), out_buckets), targets), entry_weights)| {
        let l0 = builder.new_affine("l0", 768 * nb, hidden);
        let l0f = builder.new_weights("l0f", (hidden, 768), InitSettings::Zeroed);
        let mut l0 = l0;
        l0.weights = l0.weights + l0f.repeat(nb);
        // Threat weights start at zero: the untrained threat inputs leave
        // the net exactly a plain pairwise net.
        let l0t = builder.new_weights("l0t", (hidden, N_THREAT_FEATURES), InitSettings::Zeroed);
        let h1 = builder.new_affine("l1", hidden, OUTPUT_BUCKETS * l1);
        let h2 = builder.new_affine("l2", l1, OUTPUT_BUCKETS * l2);
        let h3 = builder.new_affine("l3", l2, OUTPUT_BUCKETS);
        let half = |p, t, start, end| (l0.slice(start, end).forward(p) + l0t.slice_rows(start, end).matmul(t)).crelu();
        let pw = |p, t| half(p, t, 0, hidden / 2) * half(p, t, hidden / 2, hidden);
        let x = pw(stm, stm_t).concat(pw(ntm, ntm_t));
        let y1 = h1.forward(x).select(out_buckets).crelu();
        let y2 = h2.forward(y1).select(out_buckets).crelu();
        let out = h3.forward(y2).select(out_buckets);
        let mut loss = loss_fn(out, targets);
        if act_l1 != 0.0 {
            loss = loss + x.reduce_sum_rows() * (act_l1 / hidden as f32);
        }
        let loss = (entry_weights * loss).reduce_sum_batch();
        (Some(loss), vec![("output".to_string(), out)])
    })
}

fn optimiser(defn: ModelDefinition) -> Optimiser<bullet::nn::ExecutionContext, AdamW<bullet::nn::ExecutionContext>> {
    let weights = ModelWeights::new(&defn, 0);
    let device = DefaultDevice::new(0).unwrap();
    let mut opt = Optimiser::<_, AdamW<_>>::new(defn, weights, device, AdamWParams::default()).unwrap();
    // As build_pairwise: h1_w on the i8 grid (127 / L1_QB), L2/L3 unclipped;
    // and the threat weights on the engine's i8 grid at QA.
    let w = 127.0 / L1_QB as f32;
    opt.set_params_for_weight("l1w", AdamWParams { max_weight: w, min_weight: -w, ..Default::default() });
    for id in ["l2w", "l2b", "l3w", "l3b", "l1b"] {
        opt.set_params_for_weight(id, AdamWParams { max_weight: 1e6, min_weight: -1e6, ..Default::default() });
    }
    let t = 127.0 / QA as f32;
    opt.set_params_for_weight("l0t", AdamWParams { max_weight: t, min_weight: -t, ..Default::default() });
    opt
}

pub fn run(cfg: Config, loader: InterleavedBinpackLoader) {
    let meta = &cfg.meta;
    let hidden = meta.hidden;
    let lay = layout(meta.buckets);
    let nb = num_buckets(&lay);
    let pieces = ChessBucketsMirroredExperts::new(lay, 1, [0; 33]);
    let spec = inputs(pieces.num_inputs(), pieces.max_active());
    let mut opt = optimiser(definition(meta, &spec, cfg.loss, cfg.act_l1));

    if let Some(ckpt) = &cfg.resume {
        opt.load_from_checkpoint(&format!("{ckpt}/optimiser_state")).unwrap_or_else(|e| panic!("{ckpt}: {e:?}"));
        println!("resumed weights and optimiser from {ckpt}; continuing at superbatch {}", cfg.start);
    } else if let Some(ckpt) = &cfg.init_from {
        init_from_plain(&mut opt, ckpt, hidden * N_THREAT_FEATURES);
    }

    let steps = TrainingSteps {
        batch_size: cfg.batch_size,
        batches_per_superbatch: cfg.batches_per_superbatch,
        start_superbatch: cfg.start,
        end_superbatch: cfg.superbatches,
    };
    let lr_schedule = lr::Warmup {
        inner: lr::CosineDecayLR { initial_lr: cfg.lr0, final_lr: cfg.final_lr, final_superbatch: cfg.superbatches },
        warmup_batches: cfg.warmup_batches,
    };
    let loader = ReadMapLoader::new(loader, mapper(&spec, pieces, cfg.wdl), cfg.threads as u8);
    if cfg.measure_loader {
        bullet_trainer::run::measure_max_cpu_throughput(loader, steps).unwrap();
        return;
    }
    std::fs::create_dir_all(&cfg.out).unwrap();
    let save = saved_format(nb);
    let error_record = RefCell::new(Vec::new());
    let (mut loss_sum, mut ticks) = (0.0, 0.0);
    logger::clear_colours();
    train(
        &mut opt,
        TrainingSchedule { steps, log_rate: 128, lr_schedule: lr_schedule.boxed() },
        loader,
        |_, step, error| {
            loss_sum += error;
            ticks += 1.0;
            if step.batch().is_multiple_of(32) || step.batch() + 1 == step.batches_per_superbatch() {
                error_record.borrow_mut().push((step.superbatch(), step.batch(), loss_sum / ticks));
                loss_sum = 0.0;
                ticks = 0.0;
            }
        },
        |opt, step| {
            let sb = step.superbatch();
            if sb.is_multiple_of(cfg.save_rate) || sb == step.final_superbatch() {
                let name = format!("{}-{sb}", cfg.net_id);
                let path = format!("{}/{name}", cfg.out);
                std::fs::create_dir(&path).unwrap_or(());
                save_to_checkpoint(opt, &save, &path);
                write_losses(&format!("{path}/log.txt"), &error_record.borrow());
                println!("Saved [{name}]");
            }
        },
    )
    .unwrap();
}

/// Start from a plain pairwise checkpoint (`train --pairwise`, same width):
/// every weight it has is copied and the threat weights stay zero, so the
/// starting net evaluates exactly like the checkpoint. Optimiser state starts fresh.
fn init_from_plain<O: bullet_trainer::optimiser::OptimiserState<bullet::nn::ExecutionContext>>(
    opt: &mut Optimiser<bullet::nn::ExecutionContext, O>,
    ckpt: &str,
    n_threat_weights: usize,
) {
    let path = format!("{ckpt}/optimiser_state/weights.bin");
    let buf = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (mut off, mut out) = (0, Vec::new());
    use bullet_trainer::model::utils::{read_from_byte_buffer, write_to_byte_buffer};
    let zero = vec![0.0f32; n_threat_weights];
    while off < buf.len() {
        let (vals, id, n) = read_from_byte_buffer(&buf[off..]);
        off += n;
        assert_ne!(id, "l0t", "{ckpt} already has threat weights");
        out.extend(write_to_byte_buffer(&bullet_compiler::tensor::TValue::F32(vals), &id).unwrap());
    }
    out.extend(write_to_byte_buffer(&bullet_compiler::tensor::TValue::F32(zero), "l0t").unwrap());
    let tmp = std::env::temp_dir().join(format!("piebot_init_threats_{}.bin", std::process::id()));
    std::fs::write(&tmp, out).unwrap();
    opt.load_weights_from_file(tmp.to_str().unwrap()).unwrap();
    std::fs::remove_file(tmp).ok();
    println!("initialised from {ckpt}; threat weights are zero");
}

/// Float evaluation (centipawns, side to move) of `fens` by a threats
/// checkpoint, through the training graph itself: the reference the exported
/// file (quantised) is checked against.
pub fn eval_fens(meta: &NetMeta, ckpt: &str, fens: &[String]) -> Vec<f32> {
    use bullet_trainer::{model::ModelEvaluator, run::{HostPool, Step}};
    let lay = layout(meta.buckets);
    let pieces = ChessBucketsMirroredExperts::new(lay, 1, [0; 33]);
    let spec = inputs(pieces.num_inputs(), pieces.max_active());
    let loss: Loss = |o, t| o.sigmoid().squared_error(t);
    let mut opt = optimiser(definition(meta, &spec, loss, 0.0));
    opt.load_from_checkpoint(&format!("{ckpt}/optimiser_state")).unwrap_or_else(|e| panic!("{ckpt}: {e:?}"));
    let device = opt.device();
    let mut evaluator = ModelEvaluator::new(opt.definition(), device.clone()).unwrap();
    evaluator.load_device_weights(opt.weights()).unwrap();
    let map = mapper(&spec, pieces, 0.0);
    let pool = HostPool::new(device.clone());
    fens.iter()
        .map(|fen| {
            let pos: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();
            let host = map.map(&pool, &[pos], Step::default(), 1).unwrap();
            let out = evaluator.evaluate(&host.to_device(&device).unwrap()).unwrap();
            let v = out.get("output").unwrap().to_host().unwrap();
            let bullet_compiler::tensor::TValue::F32(v) = v else { panic!() };
            v[0] * EVAL_SCALE as f32
        })
        .collect()
}
