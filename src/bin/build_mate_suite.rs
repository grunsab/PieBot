use std::env;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

// Utility to convert Lichess puzzle CSV into JSONL suites for mateInN.
// - Filters by tags containing "mateIn7" (preferred), then 6, 5, ... until total positions reach --total (default 1000)
// - For each puzzle line, computes the player-to-move position by applying the first UCI move to the FEN.
// - Emits JSONL with {"fen":"FEN after first move","best":"<second UCI move>"}
//
// Usage:
//   cargo run --release --bin build_mate_suite -- --input path/to/lichess_db_puzzle.csv --out piebot/src/suites --total 1000

#[derive(Debug)]
struct Args {
    input: PathBuf,
    out_dir: PathBuf,
    total: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut total: usize = 1000;
    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--input" => { input = it.next().map(PathBuf::from); },
            "--out" | "--out-dir" => { out_dir = it.next().map(PathBuf::from); },
            "--total" => { if let Some(v) = it.next() { total = v.parse::<usize>().map_err(|e| format!("--total parse: {}", e))?; } },
            _ => {}
        }
    }
    let input = input.ok_or_else(|| "missing --input".to_string())?;
    let out_dir = out_dir.ok_or_else(|| "missing --out".to_string())?;
    Ok(Args { input, out_dir, total })
}

#[derive(Clone, Debug)]
struct SuiteEntry { fen_after_first: String, best_uci: String }

fn uci_to_move(board: &cozy_chess::Board, uci: &str) -> Option<cozy_chess::Move> {
    let mut chosen = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == uci { chosen = Some(m); break; } }
        chosen.is_some()
    });
    chosen
}

fn parse_line_to_entry(line: &str) -> Option<(usize, SuiteEntry)> {
    // Quick prefilter: must be a mate puzzle line
    // We'll check mateInN by N later, but here reject early if not a mate.
    if !line.contains("mateIn") { return None; }

    // CSV columns (approx): id, fen, moves, rating, rd, popularity, nbPlays, themes, url, opening
    // All fields are unquoted and separated by commas; FEN contains spaces but no commas; moves are space-separated UCI.
    let mut parts = line.splitn(10, ',').collect::<Vec<_>>();
    if parts.len() < 8 { return None; }
    let fen_raw = parts[1].trim();
    let moves_raw = parts[2].trim();
    let tags = parts[7]; // themes/tags column

    // Determine mate N from tags
    let mate_n = if tags.contains("mateIn7") { Some(7) }
                 else if tags.contains("mateIn6") { Some(6) }
                 else if tags.contains("mateIn5") { Some(5) }
                 else if tags.contains("mateIn4") { Some(4) }
                 else if tags.contains("mateIn3") { Some(3) }
                 else if tags.contains("mateIn2") { Some(2) }
                 else if tags.contains("mateIn1") { Some(1) }
                 else { None };
    let mate_n = match mate_n { Some(n) => n, None => return None };

    // Moves list: first move to apply to the FEN (opponent's move), second is player's best move
    let toks = moves_raw.split_whitespace().collect::<Vec<_>>();
    if toks.len() < 2 { return None; }
    let first = toks[0];
    let second = toks[1].to_string();

    // Parse FEN and apply first move
    let base = match cozy_chess::Board::from_fen(fen_raw, false) { Ok(b) => b, Err(_) => return None };
    let m1 = match uci_to_move(&base, first) { Some(m) => m, None => return None };
    let mut after = base.clone();
    after.play(m1);
    let fen_after_first = format!("{}", after);

    Some((mate_n, SuiteEntry { fen_after_first, best_uci: second }))
}

fn write_jsonl(path: &PathBuf, entries: &[SuiteEntry]) -> io::Result<()> {
    if entries.is_empty() { return Ok(()); }
    let mut f = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
    for e in entries {
        // JSONL schema aligns with existing tests: {"fen":"...","best":"uci"}
        writeln!(f, "{{\"fen\":\"{}\",\"best\":\"{}\"}}", e.fen_after_first, e.best_uci)?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args = match parse_args() { Ok(a) => a, Err(e) => { eprintln!("Error: {}", e); std::process::exit(2); } };
    std::fs::create_dir_all(&args.out_dir)?;

    let file = File::open(&args.input)?;
    let rdr = BufReader::new(file);

    // Collect up to needed counts in a single pass, prioritizing higher N.
    let mut picked7: Vec<SuiteEntry> = Vec::new();
    let mut picked6: Vec<SuiteEntry> = Vec::new();
    let mut picked5: Vec<SuiteEntry> = Vec::new();
    let mut picked4: Vec<SuiteEntry> = Vec::new();
    let mut picked3: Vec<SuiteEntry> = Vec::new();
    let mut picked2: Vec<SuiteEntry> = Vec::new();
    let mut picked1: Vec<SuiteEntry> = Vec::new();

    for (lineno, line) in rdr.lines().enumerate() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        // Very fast reject to save CPU
        if !(line.contains("mateIn7") || line.contains("mateIn6") || line.contains("mateIn5") || line.contains("mateIn4") || line.contains("mateIn3") || line.contains("mateIn2") || line.contains("mateIn1")) { continue; }
        if let Some((n, entry)) = parse_line_to_entry(&line) {
            // Compute remaining slots for each bucket based on preference and total target
            let total_so_far = picked7.len() + picked6.len() + picked5.len() + picked4.len() + picked3.len() + picked2.len() + picked1.len();
            if total_so_far >= args.total { break; }
            match n {
                7 => if picked7.len() < args.total { picked7.push(entry); },
                6 => if picked7.len() + picked6.len() < args.total { picked6.push(entry); },
                5 => if picked7.len() + picked6.len() + picked5.len() < args.total { picked5.push(entry); },
                4 => if picked7.len() + picked6.len() + picked5.len() + picked4.len() < args.total { picked4.push(entry); },
                3 => if picked7.len() + picked6.len() + picked5.len() + picked4.len() + picked3.len() < args.total { picked3.push(entry); },
                2 => if picked7.len() + picked6.len() + picked5.len() + picked4.len() + picked3.len() + picked2.len() < args.total { picked2.push(entry); },
                1 => if picked7.len() + picked6.len() + picked5.len() + picked4.len() + picked3.len() + picked2.len() + picked1.len() < args.total { picked1.push(entry); },
                _ => {}
            }
        } else {
            // Skip silently; malformed or unparsable line
            if lineno % 100000 == 0 && lineno > 0 { eprintln!("[info] processed {} lines...", lineno); }
        }
    }

    // Trim to total target in preference order
    let mut remaining = args.total;
    if picked7.len() > remaining { picked7.truncate(remaining); }
    remaining = remaining.saturating_sub(picked7.len());
    if remaining == 0 { /* only matein7 file */ } else {
        if picked6.len() > remaining { picked6.truncate(remaining); }
        remaining = remaining.saturating_sub(picked6.len());
        if remaining > 0 { if picked5.len() > remaining { picked5.truncate(remaining); } remaining = remaining.saturating_sub(picked5.len()); }
        if remaining > 0 { if picked4.len() > remaining { picked4.truncate(remaining); } remaining = remaining.saturating_sub(picked4.len()); }
        if remaining > 0 { if picked3.len() > remaining { picked3.truncate(remaining); } remaining = remaining.saturating_sub(picked3.len()); }
        if remaining > 0 { if picked2.len() > remaining { picked2.truncate(remaining); } remaining = remaining.saturating_sub(picked2.len()); }
        if remaining > 0 { if picked1.len() > remaining { picked1.truncate(remaining); } remaining = remaining.saturating_sub(picked1.len()); }
    }

    // Write files
    let mut wrote_any = false;
    if !picked7.is_empty() {
        let p = args.out_dir.join("matein7.txt");
        write_jsonl(&p, &picked7)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked7.len(), p.display());
    }
    if !picked6.is_empty() {
        let p = args.out_dir.join("matein6.txt");
        write_jsonl(&p, &picked6)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked6.len(), p.display());
    }
    if !picked5.is_empty() {
        let p = args.out_dir.join("matein5.txt");
        write_jsonl(&p, &picked5)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked5.len(), p.display());
    }
    if !picked4.is_empty() {
        let p = args.out_dir.join("matein4.txt");
        write_jsonl(&p, &picked4)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked4.len(), p.display());
    }
    if !picked3.is_empty() {
        let p = args.out_dir.join("matein3.txt");
        write_jsonl(&p, &picked3)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked3.len(), p.display());
    }
    if !picked2.is_empty() {
        let p = args.out_dir.join("matein2.txt");
        write_jsonl(&p, &picked2)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked2.len(), p.display());
    }
    if !picked1.is_empty() {
        let p = args.out_dir.join("matein1.txt");
        write_jsonl(&p, &picked1)?; wrote_any = true;
        eprintln!("[write] {} entries -> {}", picked1.len(), p.display());
    }

    if !wrote_any {
        eprintln!("No mateInN entries produced. Check the input file and tags.");
    }

    Ok(())
}

