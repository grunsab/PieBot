use cozy_chess::Board;
use std::path::Path;

/// Load a FEN/EPD opening suite. One position per line; blank lines and
/// `#` comments are skipped; 4-field EPD lines are padded with halfmove and
/// fullmove counters. Any unreadable file or invalid line is a hard error:
/// a configured suite that silently degrades would corrupt an entire
/// training or evaluation campaign.
pub fn load_fen_suite(path: &Path) -> Result<Vec<Board>, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read openings file {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (line_idx, line) in contents.lines().enumerate() {
        let raw = line.trim();
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = raw.split_whitespace().collect();
        let fen = if parts.len() >= 6 {
            parts[0..6].join(" ")
        } else if parts.len() >= 4 {
            let mut v = parts[0..4].to_vec();
            v.push("0");
            v.push("1");
            v.join(" ")
        } else {
            raw.to_string()
        };
        let board = Board::from_fen(&fen, false).map_err(|e| {
            format!(
                "invalid opening at {} line {}: {raw:?} ({e})",
                path.display(),
                line_idx + 1
            )
        })?;
        out.push(board);
    }
    if out.is_empty() {
        return Err(format!(
            "openings file {} contains no valid positions",
            path.display()
        ));
    }
    Ok(out)
}
