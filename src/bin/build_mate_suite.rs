use clap::Parser;
use cozy_chess::Board;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "build_mate_suite")]
#[command(about = "Convert Lichess mate puzzle CSV to JSONL suite format")]
struct Args {
    /// Input CSV file (e.g., data/mateIn4_unprocessed.csv)
    #[arg(short, long)]
    input: PathBuf,

    /// Output JSONL file (e.g., src/suites/matein4.txt)
    #[arg(short, long)]
    output: PathBuf,

    /// Maximum number of puzzles to extract
    #[arg(short, long, default_value = "100")]
    limit: usize,

    /// Minimum rating threshold
    #[arg(short = 'r', long, default_value = "1500")]
    min_rating: i32,

    /// Skip header row
    #[arg(short = 's', long, default_value = "true")]
    skip_header: bool,
}

#[derive(Debug)]
struct Puzzle {
    fen: String,
    best_move: String,
    rating: i32,
}

fn parse_uci_move(board: &Board, uci: &str) -> Option<cozy_chess::Move> {
    let mut found_move = None;
    board.generate_moves(|moves| {
        for m in moves {
            if format!("{}", m) == uci {
                found_move = Some(m);
                return true;
            }
        }
        false
    });
    found_move
}

fn parse_puzzle_row(line: &str) -> Option<Puzzle> {
    // CSV format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 4 {
        return None;
    }

    let initial_fen = fields[1].trim();
    let moves = fields[2].trim();
    let rating = fields[3].trim().parse::<i32>().ok()?;

    // Parse moves: "g3g7 a6a7 b8a8"
    // First move is opponent's last move (apply to FEN)
    // Second move is the solution move
    let move_list: Vec<&str> = moves.split_whitespace().collect();
    if move_list.len() < 2 {
        return None;
    }

    let first_move_uci = move_list[0];
    let best_move_uci = move_list[1];

    // Apply first move to get the actual puzzle position
    let mut board = Board::from_fen(initial_fen, false).ok()?;
    let first_move = parse_uci_move(&board, first_move_uci)?;
    board.play(first_move);

    // Get FEN after first move
    let puzzle_fen = board.to_string();

    Some(Puzzle {
        fen: puzzle_fen,
        best_move: best_move_uci.to_string(),
        rating,
    })
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("Reading CSV from: {}", args.input.display());
    println!("Writing JSONL to: {}", args.output.display());
    println!(
        "Limit: {} puzzles, Min rating: {}",
        args.limit, args.min_rating
    );

    let input_file = File::open(&args.input)?;
    let reader = BufReader::new(input_file);

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut output_file = File::create(&args.output)?;

    let mut count = 0;
    let mut skipped_rating = 0;
    let mut skipped_parse = 0;

    for (idx, line) in reader.lines().enumerate() {
        if args.skip_header && idx == 0 {
            continue;
        }

        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match parse_puzzle_row(&line) {
            Some(puzzle) => {
                if puzzle.rating < args.min_rating {
                    skipped_rating += 1;
                    continue;
                }

                // Write JSONL format: {"fen":"...","best":"..."}
                let json_line = format!(
                    r#"{{"fen":"{}","best":"{}"}}"#,
                    puzzle.fen, puzzle.best_move
                );
                writeln!(output_file, "{}", json_line)?;

                count += 1;
                if count >= args.limit {
                    break;
                }

                if count % 10 == 0 {
                    print!("\rProcessed: {} puzzles", count);
                    std::io::stdout().flush()?;
                }
            }
            None => {
                skipped_parse += 1;
            }
        }
    }

    println!("\n\nSummary:");
    println!("  Extracted: {}", count);
    println!("  Skipped (low rating): {}", skipped_rating);
    println!("  Skipped (parse error): {}", skipped_parse);
    println!("\nWrote {} puzzles to: {}", count, args.output.display());

    Ok(())
}
