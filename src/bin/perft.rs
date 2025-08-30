use piebot::perft::perft;

#[derive(clap::Parser, Debug)]
#[command(name = "perft", about = "Perft driver for PieBot")]
struct Args {
    /// Search depth
    #[arg(value_name = "DEPTH")]
    depth: u32,
    /// FEN string or "startpos"
    #[arg(value_name = "FEN", default_value = "startpos")]
    fen: String,
    /// Number of threads for root-split
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Report elapsed time and NPS
    #[arg(long, default_value_t = false)]
    nps: bool,
    /// Run both Pleco and Cozy (if available) and compare NPS
    #[arg(long, default_value_t = false)]
    compare: bool,
}

#[cfg(feature = "board-pleco")]
fn main() {
    use clap::Parser;
    use rayon::prelude::*;
    use std::time::Instant;

    let args = Args::parse();
    let depth = args.depth;
    if depth == 0 {
        if args.nps || args.compare { println!("nodes: 1 elapsed: 0.000s nps: inf"); } else { println!("nodes: 1"); }
        return;
    }

    // Build base boards from FEN or startpos
    let pleco_base = if args.fen == "startpos" {
        pleco::Board::start_pos()
    } else {
        pleco::Board::from_fen(&args.fen).expect("Invalid FEN")
    };
    let cozy_base = if args.fen == "startpos" {
        cozy_chess::Board::default()
    } else {
        cozy_chess::Board::from_fen(&args.fen, false).expect("Invalid FEN")
    };

    // Helper: parallel root-split perft for Pleco
    let pleco_run = |threads: usize| -> (u64, f64) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
        let mut nodes: u64 = 0;
        let dt = pool.install(|| {
            let t0 = Instant::now();
            if threads <= 1 {
                let mut b = pleco_base.clone();
                nodes = perft(&mut b, depth);
            } else {
                let root_moves: Vec<pleco::BitMove> = pleco_base.generate_moves().iter().copied().collect();
                nodes = root_moves.par_iter().map(|&mv| {
                    let mut b = pleco_base.clone();
                    b.apply_move(mv);
                    perft(&mut b, depth - 1)
                }).sum();
            }
            t0.elapsed().as_secs_f64()
        });
        (nodes, dt)
    };

    // Helper: parallel root-split perft for Cozy
    fn cozy_perft_local(board: &cozy_chess::Board, depth: u32) -> u64 {
        if depth == 0 { return 1; }
        let mut nodes = 0u64;
        board.generate_moves(|moves| {
            for m in moves {
                let mut child = board.clone();
                child.play(m);
                nodes += cozy_perft_local(&child, depth - 1);
            }
            false
        });
        nodes
    }
    let cozy_run = |threads: usize| -> (u64, f64) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
        let mut nodes: u64 = 0;
        let dt = pool.install(|| {
            let t0 = Instant::now();
            if threads <= 1 {
                nodes = cozy_perft_local(&cozy_base, depth);
            } else {
                let mut root_moves: Vec<cozy_chess::Move> = Vec::new();
                cozy_base.generate_moves(|moves| { for m in moves { root_moves.push(m); } false });
                nodes = root_moves.par_iter().map(|&mv| {
                    let mut b = cozy_base.clone();
                    b.play(mv);
                    cozy_perft_local(&b, depth - 1)
                }).sum();
            }
            t0.elapsed().as_secs_f64()
        });
        (nodes, dt)
    };

    if args.compare {
        // Run both and report NPS comparison
        let (p_nodes, p_dt) = pleco_run(args.threads);
        let (c_nodes, c_dt) = cozy_run(args.threads);
        let p_nps = if p_dt > 0.0 { p_nodes as f64 / p_dt } else { f64::INFINITY };
        let c_nps = if c_dt > 0.0 { c_nodes as f64 / c_dt } else { f64::INFINITY };
        println!("pleco: nodes={} elapsed={:.3}s nps={:.1}", p_nodes, p_dt, p_nps);
        println!("cozy:  nodes={} elapsed={:.3}s nps={:.1}", c_nodes, c_dt, c_nps);
        if p_nps.is_finite() && c_nps.is_finite() {
            if p_nps > c_nps {
                let diff = p_nps - c_nps; let pct = diff / c_nps * 100.0;
                println!("winner: pleco by +{:.1} nps (+{:.1}%)", diff, pct);
            } else if c_nps > p_nps {
                let diff = c_nps - p_nps; let pct = diff / p_nps * 100.0;
                println!("winner: cozy by +{:.1} nps (+{:.1}%)", diff, pct);
            } else {
                println!("tie: equal nps");
            }
        }
        return;
    }

    // Single engine mode (Pleco)
    let (nodes, dt) = pleco_run(args.threads);
    if args.nps { println!("nodes: {nodes} elapsed: {:.3}s nps: {:.1}", dt, nodes as f64 / dt.max(f64::EPSILON)); }
    else { println!("nodes: {nodes}"); }
}

#[cfg(not(feature = "board-pleco"))]
fn main() {
    use clap::Parser;
    use rayon::prelude::*;
    use std::time::Instant;

    // Local cozy perft (same as in lib when Pleco feature is disabled)
    fn cozy_perft_local(board: &cozy_chess::Board, depth: u32) -> u64 {
        if depth == 0 { return 1; }
        let mut nodes = 0u64;
        board.generate_moves(|moves| {
            for m in moves {
                let mut child = board.clone();
                child.play(m);
                nodes += cozy_perft_local(&child, depth - 1);
            }
            false
        });
        nodes
    }

    let args = Args::parse();
    let depth = args.depth;
    if args.compare {
        eprintln!("Error: --compare requires --features board-pleco to be enabled.");
        std::process::exit(1);
    }

    let base = if args.fen == "startpos" {
        cozy_chess::Board::default()
    } else {
        cozy_chess::Board::from_fen(&args.fen, false).expect("Invalid FEN")
    };

    let pool = rayon::ThreadPoolBuilder::new().num_threads(args.threads.max(1)).build().expect("thread pool");
    let (nodes, dt) = pool.install(|| {
        let t0 = Instant::now();
        let nodes = if args.threads <= 1 {
            cozy_perft_local(&base, depth)
        } else {
            let mut root_moves: Vec<cozy_chess::Move> = Vec::new();
            base.generate_moves(|moves| { for m in moves { root_moves.push(m); } false });
            root_moves.par_iter().map(|&mv| {
                let mut b = base.clone();
                b.play(mv);
                cozy_perft_local(&b, depth - 1)
            }).sum()
        };
        (nodes, t0.elapsed().as_secs_f64())
    });

    if args.nps { println!("nodes: {nodes} elapsed: {:.3}s nps: {:.1}", dt, nodes as f64 / dt.max(f64::EPSILON)); }
    else { println!("nodes: {nodes}"); }
}
