use clap::Parser;
use piebot::selfplay::{generate_games, write_shards, SelfPlayParams};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "piebot-selfplay",
    about = "Generate self-play games and write shards"
)]
struct Args {
    #[arg(long, default_value_t = 100)]
    games: usize,
    #[arg(long, default_value_t = 100)]
    max_plies: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 4)]
    depth: u32,
    #[arg(long)]
    movetime_ms: Option<u64>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "out/shards")]
    out: PathBuf,
    #[arg(long, default_value_t = 100_000)]
    max_records_per_shard: usize,
    #[arg(long, default_value_t = true)]
    use_engine: bool,
    #[arg(long, default_value_t = 1.0)]
    temperature_tau: f32,
    #[arg(long, default_value_t = 200.0)]
    temp_cp_scale: f32,
    #[arg(long, default_value_t = 0.3)]
    dirichlet_alpha: f32,
    #[arg(long, default_value_t = 0.25)]
    dirichlet_epsilon: f32,
    #[arg(long, default_value_t = 8)]
    dirichlet_plies: usize,
    #[arg(long, default_value_t = 20)]
    temperature_moves: usize,
    #[arg(long)]
    openings: Option<PathBuf>,
    #[arg(long, default_value_t = 0.1)]
    temperature_tau_final: f32,
}

fn main() -> anyhow::Result<()> {
    let a = Args::parse();
    let params = SelfPlayParams {
        games: a.games,
        max_plies: a.max_plies,
        threads: a.threads,
        use_engine: a.use_engine,
        depth: a.depth,
        movetime_ms: a.movetime_ms,
        seed: a.seed,
        temperature_tau: a.temperature_tau,
        temp_cp_scale: a.temp_cp_scale,
        dirichlet_alpha: a.dirichlet_alpha,
        dirichlet_epsilon: a.dirichlet_epsilon,
        dirichlet_plies: a.dirichlet_plies,
        temperature_moves: a.temperature_moves,
        openings_path: a.openings,
        temperature_tau_final: a.temperature_tau_final,
    };
    eprintln!(
        "Generating {} games (depth={}, threads={}, engine={}, tau={}, dir_eps={})",
        a.games, a.depth, a.threads, a.use_engine, a.temperature_tau, a.dirichlet_epsilon
    );
    let games = generate_games(&params);
    eprintln!("Writing shards to {}", a.out.display());
    let shards = write_shards(&games, &a.out, a.max_records_per_shard)?;
    eprintln!("Wrote {} shards", shards.len());
    Ok(())
}
