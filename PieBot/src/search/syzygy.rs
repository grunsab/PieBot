//! Syzygy endgame tablebase probing (process-wide, set via `SyzygyPath`).
//!
//! Positions are converted from cozy-chess through FEN, which costs a few
//! microseconds; callers probe only positions with few enough pieces, and
//! in-tree only right after a capture or pawn move.

use cozy_chess::Board;
use shakmaty::{fen::Fen, CastlingMode, Chess};
use shakmaty_syzygy::{Tablebase, Wdl};
use std::sync::{Arc, RwLock};

static TABLES: RwLock<Option<Arc<Tablebase<Chess>>>> = RwLock::new(None);
static MAX_PIECES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Load every table under the `;`/`:`-separated directories in `path`.
/// An empty path (or `<empty>`) unloads. Returns the number of tables found.
pub fn set_path(path: &str) -> Result<usize, String> {
    let path = path.trim();
    if path.is_empty() || path == "<empty>" {
        *TABLES.write().unwrap() = None;
        MAX_PIECES.store(0, std::sync::atomic::Ordering::Relaxed);
        return Ok(0);
    }
    let mut tb = Tablebase::new();
    let mut n = 0;
    for dir in path.split([';', ':']).filter(|d| !d.is_empty()) {
        n += tb.add_directory(dir).map_err(|e| format!("{dir}: {e}"))?;
    }
    MAX_PIECES.store(if n > 0 { tb.max_pieces() } else { 0 }, std::sync::atomic::Ordering::Relaxed);
    *TABLES.write().unwrap() = if n > 0 { Some(Arc::new(tb)) } else { None };
    Ok(n)
}

/// Largest piece count (kings included) the loaded tables cover; 0 if none.
#[inline]
pub fn max_pieces() -> usize {
    MAX_PIECES.load(std::sync::atomic::Ordering::Relaxed)
}

fn to_shakmaty(board: &Board) -> Option<Chess> {
    let fen = format!("{board}");
    Fen::from_ascii(fen.as_bytes())
        .ok()?
        .into_position(CastlingMode::Chess960)
        .ok()
}

/// WDL for the side to move, assuming the position was just reached by a
/// capture or pawn move (halfmove clock zero): -2 loss, -1 blessed loss,
/// 0 draw, 1 cursed win, 2 win. None if not covered.
pub fn probe_wdl(board: &Board) -> Option<i32> {
    if board.occupied().len() as usize > max_pieces() || board.castle_rights(cozy_chess::Color::White).short.is_some()
        || board.castle_rights(cozy_chess::Color::White).long.is_some()
        || board.castle_rights(cozy_chess::Color::Black).short.is_some()
        || board.castle_rights(cozy_chess::Color::Black).long.is_some()
    {
        return None;
    }
    let guard = TABLES.read().ok()?;
    let tb = guard.as_ref()?;
    let pos = to_shakmaty(board)?;
    let wdl = tb.probe_wdl_after_zeroing(&pos).ok()?;
    Some(match wdl {
        Wdl::Loss => -2,
        Wdl::BlessedLoss => -1,
        Wdl::Draw => 0,
        Wdl::CursedWin => 1,
        Wdl::Win => 2,
    })
}

/// The tablebase move at the root, in cozy-chess move notation (castling as
/// king-takes-rook), honouring the halfmove clock.
pub fn root_move(board: &Board) -> Option<String> {
    if board.occupied().len() as usize > max_pieces() {
        return None;
    }
    let guard = TABLES.read().ok()?;
    let tb = guard.as_ref()?;
    let pos = to_shakmaty(board)?;
    let (m, _) = tb.best_move(&pos).ok()??;
    Some(m.to_uci(CastlingMode::Chess960).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_probed_without_tables() {
        let b = Board::from_fen("8/8/8/8/8/2k5/2q5/K7 w - - 0 1", false).unwrap();
        assert_eq!(max_pieces(), 0);
        assert_eq!(probe_wdl(&b), None);
        assert_eq!(root_move(&b), None);
    }
}
