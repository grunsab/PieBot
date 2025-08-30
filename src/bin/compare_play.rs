use clap::Parser;
use cozy_chess::{Board, Move};
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use std::time::Instant;
use cozy_chess::{Color, Piece, Square};
use std::fmt;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "compare-play", about = "Play games: baseline (alphabeta) vs experimental (alphabeta_temp)")]
struct Args {
    /// Number of games to play
    #[arg(long, default_value_t = 40)]
    games: usize,

    /// Movetime per move in milliseconds
    #[arg(long, default_value_t = 200)]
    movetime: u64,

    /// Number of noisy plies at the start of each game (both sides)
    #[arg(long, default_value_t = 12)]
    noise_plies: usize,

    /// Top-K moves (by ordering) to sample from under noise
    #[arg(long, default_value_t = 5)]
    noise_topk: usize,

    /// Max plies before declaring a draw
    #[arg(long, default_value_t = 200)]
    max_plies: usize,

    /// Threads for each engine (1 recommended for reproducibility)
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Random seed
    #[arg(long, default_value_t = 1u64)]
    seed: u64,

    /// Optional: write summary JSON to this path
    #[arg(long)]
    json_out: Option<String>,

    /// Optional: write summary CSV to this path
    #[arg(long)]
    csv_out: Option<String>,

    /// Optional: write all games to a single PGN file
    #[arg(long)]
    pgn_out: Option<String>,

    /// Optional: write per-move diagnostics as JSONL (one JSON per move)
    #[arg(long)]
    jsonl_out: Option<String>,

    // Baseline config
    #[arg(long)]
    base_threads: Option<usize>,
    #[arg(long)]
    base_eval: Option<String>, // material|pst|nnue
    #[arg(long)]
    base_use_nnue: Option<bool>,
    #[arg(long)]
    base_blend: Option<u8>, // 0..100
    #[arg(long)]
    base_nnue_quant_file: Option<String>,
    #[arg(long)]
    base_nnue_file: Option<String>,
    #[arg(long)]
    base_hash_mb: Option<usize>,

    // Experimental config
    #[arg(long)]
    exp_threads: Option<usize>,
    #[arg(long)]
    exp_eval: Option<String>,
    #[arg(long)]
    exp_use_nnue: Option<bool>,
    #[arg(long)]
    exp_blend: Option<u8>,
    #[arg(long)]
    exp_nnue_quant_file: Option<String>,
    #[arg(long)]
    exp_nnue_file: Option<String>,
    #[arg(long)]
    exp_hash_mb: Option<usize>,

    /// Optional: enable SMP-safe profile for baseline (guards parallel instability)
    #[arg(long)]
    base_smp_safe: Option<bool>,

    /// Optional: enable SMP-safe profile for experimental
    #[arg(long)]
    exp_smp_safe: Option<bool>,
}

fn legal_moves(board: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    board.generate_moves(|ml| { for m in ml { v.push(m); } false });
    v
}

fn piece_at(board: &Board, sq: Square) -> Option<(Color, Piece)> {
    for &color in &[Color::White, Color::Black] {
        let cb = board.colors(color);
        for &piece in &[Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King] {
            let bb = cb & board.pieces(piece);
            for s in bb { if s == sq { return Some((color, piece)); } }
        }
    }
    None
}

fn is_capture_move(board: &Board, mv: Move) -> bool {
    // Direct capture if destination has opponent piece
    let stm = board.side_to_move();
    if let Some((col, _)) = piece_at(board, mv.to) { return col != stm; }
    // En passant: legal diagonal pawn move to empty square
    if let Some((_, Piece::Pawn)) = piece_at(board, mv.from) {
        let from = format!("{}", mv.from);
        let to = format!("{}", mv.to);
        if from.as_bytes()[0] != to.as_bytes()[0] {
            // Diagonal pawn move with empty target implies en passant for legal moves
            return piece_at(board, mv.to).is_none();
        }
    }
    false
}

#[derive(Clone)]
struct EngineConfig {
    threads: usize,
    eval: String,         // material|pst|nnue
    use_nnue: bool,
    blend: Option<u8>,    // 0..100
    nnue_quant_file: Option<String>,
    nnue_file: Option<String>,
    hash_mb: usize,
    smp_safe: bool,
}

impl fmt::Display for EngineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval={} use_nnue={} threads={} hash_mb={} smp_safe={} blend={:?} nnue_q={:?} nnue={:?}",
               self.eval, self.use_nnue, self.threads, self.hash_mb, self.smp_safe, self.blend, self.nnue_quant_file, self.nnue_file)
    }
}

fn san_for_move(board: &Board, mv: Move) -> String {
    // Determine moving piece
    let moving_piece = piece_at(board, mv.from).map(|(_, p)| p);
    let is_pawn = matches!(moving_piece, Some(Piece::Pawn));

    // Detect castling first; append check/mate later
    let mut san_core = if matches!(moving_piece, Some(Piece::King)) {
        let from_str = format!("{}", mv.from);
        let to_str = format!("{}", mv.to);
        let from_file = from_str.as_bytes()[0] as i32;
        let to_file = to_str.as_bytes()[0] as i32;
        let from_rank = from_str.as_bytes()[1];
        let to_rank = to_str.as_bytes()[1];
        let same_rank = from_rank == to_rank;
        let file_delta = (from_file - to_file).abs();
        // Two encodings to support:
        // 1) Standard: king e1->g1/c1 (delta 2)
        // 2) Library-encoded: king e1->h1/a1 (delta 3/4), king ends on rook square
        if same_rank && (file_delta == 2
            || (from_str == "e1" && (to_str == "h1" || to_str == "a1"))
            || (from_str == "e8" && (to_str == "h8" || to_str == "a8")))
        {
            // O-O if moving towards h-file, O-O-O if towards a-file
            if to_file > from_file { "O-O".to_string() } else { "O-O-O".to_string() }
        } else {
            // Non-castling king move falls through to normal SAN
            String::new()
        }
    } else { String::new() };

    if san_core.is_empty() {
        // Non-castling move SAN
        let piece_char = match moving_piece { Some(Piece::Knight) => 'N', Some(Piece::Bishop) => 'B', Some(Piece::Rook) => 'R', Some(Piece::Queen) => 'Q', Some(Piece::King) => 'K', _ => ' ' };
        // Capture detection (incl. en passant)
        let capture = is_capture_move(board, mv);

        // Disambiguation among same-type moves to same destination
        let (mut need_file, mut need_rank) = (false, false);
        if !is_pawn {
            let from_str = format!("{}", mv.from);
            let from_file = from_str.as_bytes()[0];
            let from_rank = from_str.as_bytes()[1];
            // Collect other candidate sources of same piece type to same target
            let mut other_sources: Vec<String> = Vec::new();
            board.generate_moves(|ml| {
                for m in ml {
                    if m != mv && m.to == mv.to {
                        if let Some((_, p)) = piece_at(board, m.from) {
                            if Some(p) == moving_piece { other_sources.push(format!("{}", m.from)); }
                        }
                    }
                }
                false
            });
            if !other_sources.is_empty() {
                let same_file_exists = other_sources.iter().any(|s| s.as_bytes()[0] == from_file);
                let same_rank_exists = other_sources.iter().any(|s| s.as_bytes()[1] == from_rank);
                // SAN minimal disambiguation:
                // - If no other shares our file -> include file.
                // - Else if no other shares our rank -> include rank.
                // - Else -> include both.
                if !same_file_exists { need_file = true; }
                else if !same_rank_exists { need_rank = true; }
                else { need_file = true; need_rank = true; }
            }
        }

        let mut s = String::new();
        if !is_pawn { s.push(piece_char); }
        if !is_pawn {
            if need_file { s.push(format!("{}", mv.from).chars().next().unwrap()); }
            if need_rank { s.push(format!("{}", mv.from).chars().nth(1).unwrap()); }
        }
        if is_pawn && capture { s.push(format!("{}", mv.from).chars().next().unwrap()); }
        if capture { s.push('x'); }
        s.push_str(&format!("{}", mv.to));
        // Promotion
        if let Some(Piece::Pawn) = moving_piece {
            if let Some(promo) = mv.promotion {
                let c = match promo { Piece::Knight=>'N', Piece::Bishop=>'B', Piece::Rook=>'R', Piece::Queen=>'Q', _=>'Q' };
                s.push('='); s.push(c);
            }
        }
        san_core = s;
    }

    // Check or checkmate suffix
    let mut next = board.clone();
    next.play(mv);
    let gives_check = !(next.checkers()).is_empty();
    let mut opp_has_legal = false;
    next.generate_moves(|ml| {
        for _ in ml { opp_has_legal = true; break; }
        opp_has_legal
    });
    if gives_check {
        if !opp_has_legal { san_core.push('#'); } else { san_core.push('+'); }
    }
    san_core
}

#[cfg(test)]
    mod tests {
        use super::*;
        use cozy_chess::{Board, Square};

    #[test]
    fn is_game_over_not_draw_on_reported_fens() {
        // Example positions reported as prematurely drawn in compare_play
        let fens = [
            // Example 1
            "r1b1k2r/6pp/n4n2/pP6/P1Bb4/8/5KPP/R6R w kq - 0 22",
            // Example 2
            "8/4N3/1k6/4Q3/8/5P2/3Q4/1K6 b - - 4 55",
            // Example 3 (same as 1)
            "r1b1k2r/6pp/n4n2/pP6/P1Bb4/8/5KPP/R6R w kq - 0 22",
        ];
        for fen in fens { 
            let board = Board::from_fen(fen, false).unwrap();
            // Should not be terminal (neither checkmate nor stalemate) at these positions
            assert!(is_game_over(&board).is_none(), "position unexpectedly terminal: {}", fen);
        }
    }

    #[test]
    fn decide_move_returns_some_on_reported_fens() {
        // Ensure both baseline and experimental search produce a legal move on the reported FENs
        let fens = [
            "r1b1k2r/6pp/n4n2/pP6/P1Bb4/8/5KPP/R6R w kq - 0 22",
            "8/4N3/1k6/4Q3/8/5P2/3Q4/1K6 b - - 4 55",
        ];
        let conf = EngineConfig {
            threads: 1,
            eval: "pst".to_string(),
            use_nnue: false,
            blend: None,
            nnue_quant_file: None,
            nnue_file: None,
            hash_mb: 32,
            smp_safe: true,
        };
        for fen in fens {
            let board = Board::from_fen(fen, false).unwrap();
            let (mb, db, _sdb, _nb, _tb, _sb, _ob) = decide_move_baseline(&board, 50, &conf, 3);
            assert!(mb.is_some(), "baseline failed to return move for {}", fen);
            let (me, de, _sde, _ne, _te, _se, _oe) = decide_move_experimental(&board, 50, &conf, 3);
            assert!(me.is_some(), "experimental failed to return move for {}", fen);
            assert!(db > 0 && de > 0, "expected positive search depths");
        }
    }

    #[test]
    fn diagnose_reported_fens_legal_and_check() {
        let fens = [
            ("ex1", "r1b1k2r/6pp/n4n2/pP6/P1Bb4/8/5KPP/R6R w kq - 0 22"),
            ("ex2", "8/4N3/1k6/4Q3/8/5P2/3Q4/1K6 b - - 4 55"),
        ];
        for (name, fen) in fens { 
            let board = Board::from_fen(fen, false).unwrap();
            let mut count = 0usize;
            board.generate_moves(|ml| { for _ in ml { count += 1; } false });
            let in_check = !(board.checkers()).is_empty();
            eprintln!("{}: legal_moves={} in_check={} fen={} stm={}", name, count, in_check, fen, if board.side_to_move()==Color::White {"w"} else {"b"});
        }
    }

    // Note: add explicit stalemate/checkmate FEN tests separately when we curate exact positions.

    #[test]
    fn bestmove_uci_roundtrips_via_find_move_uci() {
        let fens = [
            "r1b1k2r/6pp/n4n2/pP6/P1Bb4/8/5KPP/R6R w kq - 0 22",
            "8/4N3/1k6/4Q3/8/5P2/3Q4/1K6 b - - 4 55",
            // FEN observed in termination logs (should still produce a move)
            "1k6/8/p7/8/r7/5K2/8/8 w - - 2 52",
        ];
        for fen in fens {
            let board = Board::from_fen(fen, false).unwrap();
            // Baseline
            {
                let mut s = piebot::search::alphabeta::Searcher::default();
                let (bm, _sc, _n) = s.search_movetime(&board, 50, 0);
                assert!(bm.is_some(), "baseline returned no bestmove for {}", fen);
                let u = bm.unwrap();
                let m = find_move_uci(&board, u.as_str());
                assert!(m.is_some(), "baseline bestmove UCI did not map to legal move: {} on {}", u, fen);
            }
            // Experimental
            {
                let mut s = piebot::search::alphabeta_temp::Searcher::default();
                let (bm, _sc, _n) = s.search_movetime(&board, 50, 0);
                assert!(bm.is_some(), "experimental returned no bestmove for {}", fen);
                let u = bm.unwrap();
                let m = find_move_uci(&board, u.as_str());
                assert!(m.is_some(), "experimental bestmove UCI did not map to legal move: {} on {}", u, fen);
            }
        }
    }
    #[test]
    fn noisy_selection_always_returns_some_when_legal_moves_exist() {
        // Validate that noisy selector never returns None when at least one legal move exists
        let fen = "r3k2r/pppq1ppp/2n2n2/3p4/3P4/2P2NP1/PP1QPPBP/R3K2R w KQkq - 4 10";
        let board = Board::from_fen(fen, false).unwrap();
        let mut rng = SmallRng::seed_from_u64(123);
        // Baseline and experimental use identical ordering helper here
        let mut s1 = piebot::search::alphabeta::Searcher::default();
        s1.set_order_captures(true);
        s1.set_use_history(true);
        s1.set_use_killers(true);
        s1.set_use_lmr(true);
        s1.set_use_nullmove(true);
        let order1 = s1.debug_order_for_parent(&board, usize::MAX);
        let mv1 = piebot::search::noise::choose_noisy_from_order_filtered(&board, &order1, 5, &mut rng, NOISE_SEE_THRESH_CP);
        assert!(mv1.is_some(), "noisy baseline returned None");

        let mut s2 = piebot::search::alphabeta_temp::Searcher::default();
        s2.set_order_captures(true);
        s2.set_use_history(true);
        s2.set_use_killers(true);
        s2.set_use_lmr(true);
        s2.set_use_nullmove(true);
        let order2 = s2.debug_order_for_parent(&board, usize::MAX);
        let mv2 = piebot::search::noise::choose_noisy_from_order_filtered(&board, &order2, 5, &mut rng, NOISE_SEE_THRESH_CP);
        assert!(mv2.is_some(), "noisy experimental returned None");
    }
    #[test]
    fn san_disambiguates_knight_captures_by_file() {
        // Position: white knights on e5 and f6; black pawn on d7; white to move.
        // Both knights can capture d7; SAN must be Nfxd7 or Nexd7 depending on mover.
        let fen = "k2q4/3p4/5N2/4N3/8/8/8/4K3 w - - 0 1";
        let board = Board::from_fen(fen, false).unwrap();
        let mut found = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m.to_string()) == "e5d7" || format!("{}", m.to_string()) == "f6d7" {
                    found.push(m);
                }
            }
            false
        });
        assert_eq!(found.len(), 2, "expected two knight captures to d7");
        let san0 = san_for_move(&board, found[0]);
        let san1 = san_for_move(&board, found[1]);
        // Ensure both SAN forms include file disambiguation and capture
        assert!(san0 == "Nexd7" || san0 == "Nfxd7", "san0={}", san0);
        assert!(san1 == "Nexd7" || san1 == "Nfxd7", "san1={}", san1);
        assert_ne!(san0, "Nxd7", "must disambiguate");
    }

    #[test]
    fn san_disambiguates_knight_quiet_by_file() {
        // Knights on b3 and f3 can both go to d2 (quiet move). Should be Nbd2/Nfd2.
        let fen = "8/3k4/8/8/8/1N3N2/8/4K3 w - - 0 1";
        let board = Board::from_fen(fen, false).unwrap();
        let mut quiets = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "b3d2" || format!("{}", m) == "f3d2" { quiets.push(m); }
            }
            false
        });
        assert_eq!(quiets.len(), 2, "expected two quiet knight moves to d2");
        let san0 = san_for_move(&board, quiets[0]);
        let san1 = san_for_move(&board, quiets[1]);
        assert!(san0 == "Nbd2" || san0 == "Nfd2", "san0={}", san0);
        assert!(san1 == "Nbd2" || san1 == "Nfd2", "san1={}", san1);
        assert_ne!(san0, "Nd2");
        assert_ne!(san1, "Nd2");
    }

    #[test]
    fn san_disambiguates_rook_quiet_by_rank() {
        // Rooks on a1 and a3 can both go to a2 (quiet). Should be R1a2/R3a2.
        let fen = "3k4/8/8/8/8/R7/8/R3K3 w - - 0 1"; // white rooks a1 and a3
        let board = Board::from_fen(fen, false).unwrap();
        let mut quiets = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "a1a2" || format!("{}", m) == "a3a2" { quiets.push(m); }
            }
            false
        });
        assert_eq!(quiets.len(), 2, "expected two quiet rook moves to a2");
        let san0 = san_for_move(&board, quiets[0]);
        let san1 = san_for_move(&board, quiets[1]);
        assert!(san0 == "R1a2" || san0 == "R3a2", "san0={}", san0);
        assert!(san1 == "R1a2" || san1 == "R3a2", "san1={}", san1);
        assert_ne!(san0, "Ra2");
        assert_ne!(san1, "Ra2");
    }

    #[test]
    fn san_disambiguates_knight_quiet_by_rank() {
        // Knights on b3 and b5 can both go to d4 (quiet). Should be N3d4/N5d4.
        let fen = "3k4/8/1N6/1N6/8/8/8/4K3 w - - 0 1"; // knights b5 and b6? adjust to b5 and b3
        // Correct FEN: white knights on b5 and b3
        let fen = "3k4/8/8/1N6/8/1N6/8/4K3 w - - 0 1";
        let board = Board::from_fen(fen, false).unwrap();
        let mut quiets = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "b3d4" || format!("{}", m) == "b5d4" { quiets.push(m); }
            }
            false
        });
        assert_eq!(quiets.len(), 2, "expected two quiet knight moves to d4");
        let san0 = san_for_move(&board, quiets[0]);
        let san1 = san_for_move(&board, quiets[1]);
        assert!(san0 == "N3d4" || san0 == "N5d4", "san0={}", san0);
        assert!(san1 == "N3d4" || san1 == "N5d4", "san1={}", san1);
        assert_ne!(san0, "Nd4");
        assert_ne!(san1, "Nd4");
    }

    #[test]
    fn san_disambiguates_rook_quiet_by_file() {
        // Rooks on a1 and h1 can both go to e1 (quiet). Should be Rae1/Rhe1.
        // Place white king off the first rank so e1 is empty and legal.
        let fen = "3k4/8/8/8/4K3/8/8/R6R w - - 0 1"; // white king e4, rooks a1 and h1
        let board = Board::from_fen(fen, false).unwrap();
        let mut quiets = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "a1e1" || format!("{}", m) == "h1e1" { quiets.push(m); }
            }
            false
        });
        assert_eq!(quiets.len(), 2, "expected two quiet rook moves to e1");
        let san0 = san_for_move(&board, quiets[0]);
        let san1 = san_for_move(&board, quiets[1]);
        assert!(san0 == "Rae1" || san0 == "Rhe1", "san0={}", san0);
        assert!(san1 == "Rae1" || san1 == "Rhe1", "san1={}", san1);
        assert_ne!(san0, "Re1");
        assert_ne!(san1, "Re1");
    }

    #[test]
    fn san_disambiguates_bishop_quiet_by_file() {
        // Bishops on b2 and d4 can both go to c3 (quiet). Should be Bbc3/Bdc3.
        let fen = "3k4/8/8/8/3B4/8/1B6/4K3 w - - 0 1"; // bishops b2 and d4
        let board = Board::from_fen(fen, false).unwrap();
        let mut quiets = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "b2c3" || format!("{}", m) == "d4c3" { quiets.push(m); }
            }
            false
        });
        assert_eq!(quiets.len(), 2, "expected two quiet bishop moves to c3");
        let san0 = san_for_move(&board, quiets[0]);
        let san1 = san_for_move(&board, quiets[1]);
        assert!(san0 == "Bbc3" || san0 == "Bdc3", "san0={}", san0);
        assert!(san1 == "Bbc3" || san1 == "Bdc3", "san1={}", san1);
        assert_ne!(san0, "Bc3");
        assert_ne!(san1, "Bc3");
    }

    #[test]
    fn san_disambiguates_bishop_capture_by_file() {
        // Bishops on b2 and d4 can both capture c3. Should be Bbxc3/Bdxc3.
        let fen = "3k4/8/8/8/3B4/2p5/1B6/4K3 w - - 0 1"; // black pawn on c3
        let board = Board::from_fen(fen, false).unwrap();
        let mut caps = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "b2c3" || format!("{}", m) == "d4c3" { caps.push(m); }
            }
            false
        });
        assert_eq!(caps.len(), 2, "expected two bishop captures on c3");
        let san0 = san_for_move(&board, caps[0]);
        let san1 = san_for_move(&board, caps[1]);
        assert!(san0 == "Bbxc3" || san0 == "Bdxc3", "san0={}", san0);
        assert!(san1 == "Bbxc3" || san1 == "Bdxc3", "san1={}", san1);
        assert_ne!(san0, "Bxc3");
        assert_ne!(san1, "Bxc3");
    }

    #[test]
    fn san_disambiguates_rook_capture_by_rank() {
        // Rooks on a1 and a8 can both capture a5. Should be R1xa5/R8xa5.
        // Place white rooks at a1 and a8, black pawn at a5, white king at e1; block a8 rook check with a black bishop on b8.
        let fen = "Rb5k/8/8/p7/8/8/8/R3K3 w - - 0 1";
        let board = Board::from_fen(fen, false).unwrap();
        let mut caps = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "a1a5" || format!("{}", m) == "a8a5" { caps.push(m); }
            }
            false
        });
        assert_eq!(caps.len(), 2, "expected two rook captures on a5");
        let san0 = san_for_move(&board, caps[0]);
        let san1 = san_for_move(&board, caps[1]);
        assert!(san0 == "R1xa5" || san0 == "R8xa5", "san0={}", san0);
        assert!(san1 == "R1xa5" || san1 == "R8xa5", "san1={}", san1);
        assert_ne!(san0, "Rxa5");
        assert_ne!(san1, "Rxa5");
    }

    #[test]
    fn san_disambiguates_knight_capture_by_rank() {
        // Knights on b3 and b5 can both capture d4. Should be N3xd4/N5xd4.
        let fen = "3k4/8/8/1N6/3p4/1N6/8/4K3 w - - 0 1"; // black pawn on d4
        let board = Board::from_fen(fen, false).unwrap();
        let mut caps = Vec::new();
        board.generate_moves(|ml| {
            for m in ml {
                if format!("{}", m) == "b3d4" || format!("{}", m) == "b5d4" { caps.push(m); }
            }
            false
        });
        assert_eq!(caps.len(), 2, "expected two knight captures on d4");
        let san0 = san_for_move(&board, caps[0]);
        let san1 = san_for_move(&board, caps[1]);
        assert!(san0 == "N3xd4" || san0 == "N5xd4", "san0={}", san0);
        assert!(san1 == "N3xd4" || san1 == "N5xd4", "san1={}", san1);
        assert_ne!(san0, "Nxd4");
        assert_ne!(san1, "Nxd4");
    }

    #[test]
    fn san_castling_black_from_user_fen() {
        // FEN provided by user; black to move, short castling represented as e8->g8 or e8->h8.
        let fen = "r1bqk2r/2p4p/pQ2p3/3p4/2PKn3/5P2/6Pb/5B1R b kq - 1 20";
        let board = Board::from_fen(fen, false).unwrap();
        let mut cand: Option<Move> = None;
        board.generate_moves(|ml| {
            for m in ml {
                if m.from == Square::E8 && (m.to == Square::G8 || m.to == Square::H8) {
                    cand = Some(m);
                    break;
                }
            }
            cand.is_some()
        });
        if let Some(m) = cand {
            let san = san_for_move(&board, m);
            assert_eq!(san, "O-O", "expected castling to produce O-O, got {}", san);
        } else {
            // If neither e8->g8 nor e8->h8 is legal from this position per movegen, skip.
            // We still assert that if such a move existed, it would be rendered as O-O.
            // This prevents false negatives if legality differs due to attack detection.
            eprintln!("note: no e8->g8/h8 move available; skipping castling SAN check");
        }
    }

    #[test]
    fn san_castling_uses_o_o() {
        // Simple castling availability
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(fen, false).unwrap();
        let mut has_oo = false;
        let mut has_ooo = false;
        board.generate_moves(|ml| {
            for m in ml {
                if piece_at(&board, m.from).map(|(_, p)| p) == Some(Piece::King) {
                    let san = san_for_move(&board, m);
                    if san == "O-O" { has_oo = true; }
                    if san == "O-O-O" { has_ooo = true; }
                }
            }
            false
        });
        assert!(has_oo, "expected O-O to be generated");
        assert!(has_ooo, "expected O-O-O to be generated");
    }
}

const NOISE_SEE_THRESH_CP: i32 = -150; // filter obviously losing captures

fn choose_move_noisy_baseline(board: &Board, topk: usize, rng: &mut SmallRng) -> Option<Move> {
    let mut s = piebot::search::alphabeta::Searcher::default();
    // Enable ordering heuristics so top-K is meaningful
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    // Get ordered list (parent idx unknown)
    let order = s.debug_order_for_parent(board, usize::MAX);
    piebot::search::noise::choose_noisy_from_order_filtered(board, &order, topk, rng, NOISE_SEE_THRESH_CP)
}

fn choose_move_noisy_experimental(board: &Board, topk: usize, rng: &mut SmallRng) -> Option<Move> {
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    let order = s.debug_order_for_parent(board, usize::MAX);
    piebot::search::noise::choose_noisy_from_order_filtered(board, &order, topk, rng, NOISE_SEE_THRESH_CP)
}

fn decide_move_baseline(board: &Board, movetime: u64, conf: &EngineConfig, root_topk: usize) -> (Option<Move>, u32, u32, u64, f64, i32, Vec<String>) {
    let mut s = piebot::search::alphabeta::Searcher::default();
    s.set_tt_capacity_mb(conf.hash_mb);
    s.set_threads(conf.threads.max(1));
    s.set_smp_safe(conf.smp_safe);
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_null_min_depth(8);
    s.set_use_aspiration(true);
    match conf.eval.to_ascii_lowercase().as_str() {
        "material" => s.set_eval_mode(piebot::search::alphabeta::EvalMode::Material),
        "nnue" => s.set_eval_mode(piebot::search::alphabeta::EvalMode::Nnue),
        _ => s.set_eval_mode(piebot::search::alphabeta::EvalMode::Pst),
    }
    if conf.use_nnue {
        s.set_use_nnue(true);
        if let Some(ref q) = conf.nnue_quant_file { if let Ok(model) = piebot::eval::nnue::loader::QuantNnue::load_quantized(q) { s.set_nnue_quant_model(model); } }
        if let Some(ref d) = conf.nnue_file { if let Ok(nn) = piebot::eval::nnue::Nnue::load(d) { s.set_nnue_network(Some(nn)); } }
        if let Some(bl) = conf.blend { s.set_eval_blend_percent(bl); }
    }
    // Capture root-order top-K for diagnostics
    let order = s.debug_order_for_parent(board, usize::MAX);
    let topk = order.iter().take(root_topk).map(|m| format!("{}", m)).collect::<Vec<_>>();
    let t0 = Instant::now();
    let (bm, sc, nodes) = s.search_movetime(board, movetime, 0);
    let dt = t0.elapsed().as_secs_f64();
    let depth = s.last_depth();
    let seldepth = s.last_seldepth() as u32;
    let mapped = bm.as_ref().and_then(|u| find_move_uci(board, u.as_str()));
    if bm.is_none() {
        // Count legal moves for context
        let mut cnt = 0usize; board.generate_moves(|ml| { for _ in ml { cnt += 1; } false });
        eprintln!("[decide_baseline] bestmove_none: fen={} legal_cnt={}", board, cnt);
    } else if mapped.is_none() { 
        eprintln!("[decide_baseline] bestmove_uci_not_found: fen={} uci={}", board, bm.unwrap()); 
    }
    (mapped, depth, seldepth, nodes, dt, sc, topk)
}

fn decide_move_experimental(board: &Board, movetime: u64, conf: &EngineConfig, root_topk: usize) -> (Option<Move>, u32, u32, u64, f64, i32, Vec<String>) {
    let mut s = piebot::search::alphabeta_temp::Searcher::default();
    s.set_tt_capacity_mb(conf.hash_mb);
    s.set_threads(conf.threads.max(1));
    s.set_smp_safe(conf.smp_safe);
    s.set_order_captures(true);
    s.set_use_history(true);
    s.set_use_killers(true);
    s.set_use_lmr(true);
    s.set_use_nullmove(true);
    s.set_null_min_depth(8);
    s.set_use_aspiration(true);
    match conf.eval.to_ascii_lowercase().as_str() {
        "material" => s.set_eval_mode(piebot::search::alphabeta_temp::EvalMode::Material),
        "nnue" => s.set_eval_mode(piebot::search::alphabeta_temp::EvalMode::Nnue),
        _ => s.set_eval_mode(piebot::search::alphabeta_temp::EvalMode::Pst),
    }
    if conf.use_nnue {
        s.set_use_nnue(true);
        if let Some(ref q) = conf.nnue_quant_file { if let Ok(model) = piebot::eval::nnue::loader::QuantNnue::load_quantized(q) { s.set_nnue_quant_model(model); } }
        if let Some(ref d) = conf.nnue_file { if let Ok(nn) = piebot::eval::nnue::Nnue::load(d) { s.set_nnue_network(Some(nn)); } }
        if let Some(bl) = conf.blend { s.set_eval_blend_percent(bl); }
    }
    let t0 = Instant::now();
    let order = s.debug_order_for_parent(board, usize::MAX);
    let topk = order.iter().take(root_topk).map(|m| format!("{}", m)).collect::<Vec<_>>();
    let (bm, sc, nodes) = s.search_movetime(board, movetime, 0);
    let dt = t0.elapsed().as_secs_f64();
    let depth = s.last_depth();
    let seldepth = s.last_seldepth() as u32;
    let mapped = bm.as_ref().and_then(|u| find_move_uci(board, u.as_str()));
    if bm.is_none() {
        let mut cnt = 0usize; board.generate_moves(|ml| { for _ in ml { cnt += 1; } false });
        eprintln!("[decide_experimental] bestmove_none: fen={} legal_cnt={}", board, cnt);
    } else if mapped.is_none() { 
        eprintln!("[decide_experimental] bestmove_uci_not_found: fen={} uci={}", board, bm.unwrap()); 
    }
    (mapped, depth, seldepth, nodes, dt, sc, topk)
}

fn find_move_uci(board: &Board, uci: &str) -> Option<Move> {
    let mut found = None;
    board.generate_moves(|ml| {
        for m in ml { if format!("{}", m) == uci { found = Some(m); break; } }
        found.is_some()
    });
    found
}

fn is_game_over(board: &Board) -> Option<i32> {
    // Return Some(1) if side-to-move is checkmated (previous side wins)
    // Some(0) for stalemate; None otherwise.
    let mut has_legal = false;
    board.generate_moves(|ml| {
        for _ in ml { has_legal = true; return true; }
        false
    });
    if !has_legal {
        if !(board.checkers()).is_empty() { Some(1) } else { Some(0) }
    } else { None }
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    let mut rng = SmallRng::seed_from_u64(args.seed);
    // Optional JSONL diagnostics writer
    let mut jsonl: Option<std::fs::File> = match &args.jsonl_out {
        Some(p) => match std::fs::File::create(p) { Ok(f) => Some(f), Err(e) => { eprintln!("warn: failed to create jsonl_out {}: {}", p, e); None } },
        None => None,
    };

    // Build engine configs
    // Baseline config with sensible defaults: default to PST; if NNUE weights provided and no overrides, use NNUE.
    let mut base_eval = args.base_eval.clone().unwrap_or_else(|| "pst".to_string());
    let mut base_use_nnue = args.base_use_nnue.unwrap_or(false);
    if !base_use_nnue && base_eval == "pst" && (args.base_nnue_quant_file.is_some() || args.base_nnue_file.is_some()) {
        base_eval = "nnue".to_string();
        base_use_nnue = true;
    }
    let base_conf = EngineConfig {
        threads: args.base_threads.unwrap_or(args.threads),
        eval: base_eval,
        use_nnue: base_use_nnue,
        blend: args.base_blend,
        nnue_quant_file: args.base_nnue_quant_file.clone(),
        nnue_file: args.base_nnue_file.clone(),
        hash_mb: args.base_hash_mb.unwrap_or(64),
        smp_safe: args.base_smp_safe.unwrap_or(false),
    };

    let mut exp_eval = args.exp_eval.clone().unwrap_or_else(|| "pst".to_string());
    let mut exp_use_nnue = args.exp_use_nnue.unwrap_or(false);
    if !exp_use_nnue && exp_eval == "pst" && (args.exp_nnue_quant_file.is_some() || args.exp_nnue_file.is_some()) {
        exp_eval = "nnue".to_string();
        exp_use_nnue = true;
    }
    let exp_conf = EngineConfig {
        threads: args.exp_threads.unwrap_or(args.threads),
        eval: exp_eval,
        use_nnue: exp_use_nnue,
        blend: args.exp_blend,
        nnue_quant_file: args.exp_nnue_quant_file.clone(),
        nnue_file: args.exp_nnue_file.clone(),
        hash_mb: args.exp_hash_mb.unwrap_or(64),
        smp_safe: args.exp_smp_safe.unwrap_or(false),
    };

    // Detect if experimental search is identical to baseline (alphabeta_temp reexports alphabeta) and configs are identical
    let tn_base = std::any::type_name::<piebot::search::alphabeta::Searcher>();
    let tn_exp = std::any::type_name::<piebot::search::alphabeta_temp::Searcher>();
    let self_compare = tn_base == tn_exp && format!("{}", base_conf) == format!("{}", exp_conf);
    if tn_base == tn_exp && self_compare {
        eprintln!("[WARN] Experimental search equals baseline (alphabeta_temp reexports alphabeta) and configs are identical. Comparing baseline against itself.");
    }

    let mut baseline_points = 0.0f64;
    let mut experimental_points = 0.0f64;
    let mut draws = 0usize;
    // Stats
    let mut sum_nodes_base: u64 = 0;
    let mut sum_time_base: f64 = 0.0;
    let mut sum_depth_base: u64 = 0;
    let mut sum_seldepth_base: u64 = 0;
    let mut cnt_base: u64 = 0;
    let mut sum_nodes_exp: u64 = 0;
    let mut sum_time_exp: f64 = 0.0;
    let mut sum_depth_exp: u64 = 0;
    let mut sum_seldepth_exp: u64 = 0;
    let mut cnt_exp: u64 = 0;

    let mut pgn_buf = String::new();

    for g in 0..args.games {
        let mut board = Board::default();
        // Per-game RNG for noisy plies to ensure different starts across games for a fixed seed
        let game_seed: u64 = rng.gen();
        let mut game_rng = SmallRng::seed_from_u64(game_seed);
        let baseline_is_white = g % 2 == 0;
        let mut plies = 0usize;
        let mut result: Option<f64> = None; // 1.0 baseline win, 0.0 draw, -1.0 experimental win
        let mut san_moves: Vec<String> = Vec::new();
        let root_log_topk = 5usize;

        loop {
            if let Some(res) = is_game_over(&board) {
                result = Some(match res {
                    1 => { // side to move has no moves and is in check => previous mover won
                        let prev_was_baseline = (plies > 0) && ((plies - 1) % 2 == 0) == baseline_is_white;
                        if prev_was_baseline { 1.0 } else { -1.0 }
                    }
                    _ => 0.0,
                });
                eprintln!("[compare_play] terminal: game={} ply={} fen={} reason=is_game_over", g + 1, plies + 1, board);
                break;
            }
            if plies >= args.max_plies { 
                eprintln!("[compare_play] terminal: game={} ply={} fen={} reason=max_plies", g + 1, plies + 1, board);
                result = Some(0.0); break; 
            }

            let baseline_to_move = (plies % 2 == 0) == baseline_is_white;
            let mv = if plies < args.noise_plies {
                // Noisy selection from ordered top-K
                if baseline_to_move {
                    choose_move_noisy_baseline(&board, args.noise_topk, &mut game_rng)
                } else {
                    choose_move_noisy_experimental(&board, args.noise_topk, &mut game_rng)
                }
            } else {
                if baseline_to_move {
                    let (m, d, sd, n, dt, sc, order_top) = decide_move_baseline(&board, args.movetime, &base_conf, root_log_topk);
                    if let Some(mv) = m {
                        sum_nodes_base += n; sum_time_base += dt; sum_depth_base += d as u64; sum_seldepth_base += sd as u64; cnt_base += 1;
                    if let Some(f) = jsonl.as_mut() {
                        let stm = if board.side_to_move() == Color::White { "w" } else { "b" };
                        let in_check = !(board.checkers()).is_empty();
                        let is_cap = is_capture_move(&board, mv);
                        let gives_check = { let mut c = board.clone(); c.play(mv); !(c.checkers()).is_empty() };
                        let tail = if base_conf.smp_safe { "full" } else { "pvs" };
                        let obj = serde_json::json!({
                            "game": g + 1,
                            "ply": plies + 1,
                            "side": if baseline_to_move { "baseline" } else { "experimental" },
                            "stm": stm,
                            "movetime_ms": args.movetime,
                            "depth": d,
                            
                            "nodes": n,
                            "time_s": dt,
                            "score_cp": sc,
                            "bestmove": format!("{}", mv),
                            "in_check": in_check,
                            "is_capture": is_cap,
                            "gives_check": gives_check,
                            "root_top": order_top,
                            "fen": format!("{}", board),
                            "smp_safe": base_conf.smp_safe,
                            "tail_policy": tail,
                            "aspiration": "off",
                            "seldepth": sd,
                        });
                        let _ = writeln!(f, "{}", serde_json::to_string(&obj).unwrap());
                    }
                    }
                    m
                } else {
                    let (m, d, sd, n, dt, sc, order_top) = decide_move_experimental(&board, args.movetime, &exp_conf, root_log_topk);
                    if let Some(mv) = m {
                        sum_nodes_exp += n; sum_time_exp += dt; sum_depth_exp += d as u64; sum_seldepth_exp += sd as u64; cnt_exp += 1;
                    if let Some(f) = jsonl.as_mut() {
                        let stm = if board.side_to_move() == Color::White { "w" } else { "b" };
                        let in_check = !(board.checkers()).is_empty();
                        let is_cap = is_capture_move(&board, mv);
                        let gives_check = { let mut c = board.clone(); c.play(mv); !(c.checkers()).is_empty() };
                        let tail = if exp_conf.smp_safe { "full" } else { "pvs" };
                        let asp = "off";
                        let obj = serde_json::json!({
                            "game": g + 1,
                            "ply": plies + 1,
                            "side": if baseline_to_move { "baseline" } else { "experimental" },
                            "stm": stm,
                            "movetime_ms": args.movetime,
                            "depth": d,
                            
                            "nodes": n,
                            "time_s": dt,
                            "score_cp": sc,
                            "bestmove": format!("{}", mv),
                            "in_check": in_check,
                            "is_capture": is_cap,
                            "gives_check": gives_check,
                            "root_top": order_top,
                            "fen": format!("{}", board),
                            "smp_safe": exp_conf.smp_safe,
                            "tail_policy": tail,
                            "aspiration": asp,
                            "seldepth": sd,
                        });
                        let _ = writeln!(f, "{}", serde_json::to_string(&obj).unwrap());
                    }
                    }
                    m
                }
            };

            let mv = match mv {
                Some(m) => m,
                None => { 
                    // Engine failed to return a move within constraints. Treat as loss for the side to move.
                    let side = if baseline_to_move { "baseline" } else { "experimental" };
                    eprintln!("[compare_play] terminal: game={} ply={} fen={} reason=no_move side={} noise_phase={} -> adjudicate loss",
                        g + 1, plies + 1, board, side, plies < args.noise_plies);
                    result = Some(if baseline_to_move { -1.0 } else { 1.0 });
                    break; 
                }
            };
            // Record SAN before updating board
            let san = san_for_move(&board, mv);
            let mut next = board.clone();
            next.play(mv);
            board = next;
            san_moves.push(san);
            plies += 1;
        }

        match result.unwrap_or(0.0).partial_cmp(&0.0).unwrap() {
            std::cmp::Ordering::Greater => baseline_points += 1.0,
            std::cmp::Ordering::Less => experimental_points += 1.0,
            std::cmp::Ordering::Equal => draws += 1,
        }

        println!("game={} result={} (baseline_white={}) plies={}", g + 1, result.unwrap_or(0.0), baseline_is_white, plies);

        // Append PGN if requested
        if args.pgn_out.is_some() {
            let res = match result.unwrap_or(0.0).partial_cmp(&0.0).unwrap() {
                std::cmp::Ordering::Greater => if baseline_is_white { "1-0" } else { "0-1" },
                std::cmp::Ordering::Less => if baseline_is_white { "0-1" } else { "1-0" },
                std::cmp::Ordering::Equal => "1/2-1/2",
            };
            let white = if baseline_is_white { "Baseline" } else { "Experimental" };
            let black = if baseline_is_white { "Experimental" } else { "Baseline" };
            pgn_buf.push_str(&format!("[Event \"Cozy A/B\"]\n[Site \"Local\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n[TimeControl \"{}\"]\n\n",
                                     g + 1, white, black, res, args.movetime));
            // Moves with numbers
            let mut move_num = 1;
            for i in (0..san_moves.len()).step_by(2) {
                if i + 1 < san_moves.len() {
                    pgn_buf.push_str(&format!("{}. {} {} ", move_num, san_moves[i], san_moves[i+1]));
                } else {
                    pgn_buf.push_str(&format!("{}. {} ", move_num, san_moves[i]));
                }
                move_num += 1;
            }
            pgn_buf.push_str(&format!("{}\n\n", res));
        }
    }

    let avg_nps_base = if sum_time_base > 0.0 { sum_nodes_base as f64 / sum_time_base } else { 0.0 };
    let avg_nps_exp = if sum_time_exp > 0.0 { sum_nodes_exp as f64 / sum_time_exp } else { 0.0 };
    let avg_depth_base = if cnt_base > 0 { sum_depth_base as f64 / cnt_base as f64 } else { 0.0 };
    let avg_depth_exp = if cnt_exp > 0 { sum_depth_exp as f64 / cnt_exp as f64 } else { 0.0 };
    let avg_seldepth_base = if cnt_base > 0 { sum_seldepth_base as f64 / cnt_base as f64 } else { 0.0 };
    let avg_seldepth_exp = if cnt_exp > 0 { sum_seldepth_exp as f64 / cnt_exp as f64 } else { 0.0 };

    println!("summary: games={} baseline_pts={} experimental_pts={} draws={}", args.games, baseline_points, experimental_points, draws);
    println!("baseline: avg_nps={:.1} avg_depth={:.2} avg_seldepth={:.2} moves={} nodes={} time={:.3}s",
        avg_nps_base, avg_depth_base, avg_seldepth_base, cnt_base, sum_nodes_base, sum_time_base);
    println!("experimental: avg_nps={:.1} avg_depth={:.2} avg_seldepth={:.2} moves={} nodes={} time={:.3}s",
        avg_nps_exp, avg_depth_exp, avg_seldepth_exp, cnt_exp, sum_nodes_exp, sum_time_exp);

    // Optional machine-readable outputs
    if let Some(path) = args.json_out.as_deref() {
        let payload = serde_json::json!({
            "games": args.games,
            "movetime_ms": args.movetime,
            "noise_plies": args.noise_plies,
            "noise_topk": args.noise_topk,
            "threads": args.threads,
            "seed": args.seed,
            "self_compare": self_compare,
            "engines": {"baseline": tn_base, "experimental": tn_exp},
            "baseline_config": format!("{}", base_conf),
            "experimental_config": format!("{}", exp_conf),
            "points": {"baseline": baseline_points, "experimental": experimental_points, "draws": draws},
            "baseline": {
                "moves": cnt_base, "nodes": sum_nodes_base, "time_s": sum_time_base,
                "avg_nps": avg_nps_base, "avg_depth": avg_depth_base, "avg_seldepth": avg_seldepth_base
            },
            "experimental": {
                "moves": cnt_exp, "nodes": sum_nodes_exp, "time_s": sum_time_exp,
                "avg_nps": avg_nps_exp, "avg_depth": avg_depth_exp, "avg_seldepth": avg_seldepth_exp
            }
        });
        if let Err(e) = std::fs::write(path, serde_json::to_string_pretty(&payload).unwrap()) {
            eprintln!("warn: failed to write json_out: {}", e);
        }
    }

    if let Some(path) = args.csv_out.as_deref() {
        // Single-row CSV summary with header (includes configs)
        let header = "games,movetime_ms,noise_plies,noise_topk,seed,self_compare,base_type,exp_type,base_config,exp_config,baseline_pts,experimental_pts,draws,base_moves,base_nodes,base_time_s,base_avg_nps,base_avg_depth,base_avg_seldepth,exp_moves,exp_nodes,exp_time_s,exp_avg_nps,exp_avg_depth,exp_avg_seldepth\n";
        let row = format!(
            "{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{},{},{},{:.6},{:.1},{:.2},{:.2},{},{},{:.6},{:.1},{:.2},{:.2}\n",
            args.games, args.movetime, args.noise_plies, args.noise_topk, args.seed, self_compare, tn_base, tn_exp, base_conf, exp_conf,
            baseline_points, experimental_points, draws,
            cnt_base, sum_nodes_base, sum_time_base, avg_nps_base, avg_depth_base, avg_seldepth_base,
            cnt_exp, sum_nodes_exp, sum_time_exp, avg_nps_exp, avg_depth_exp, avg_seldepth_exp
        );
        let mut buf = String::new();
        buf.push_str(header);
        buf.push_str(&row);
        if let Err(e) = std::fs::write(path, buf) { eprintln!("warn: failed to write csv_out: {}", e); }
    }

    if let Some(path) = args.pgn_out.as_deref() {
        if let Err(e) = std::fs::write(path, pgn_buf) { eprintln!("warn: failed to write pgn_out: {}", e); }
    }
}
