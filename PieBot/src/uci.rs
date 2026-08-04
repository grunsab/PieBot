#[cfg(not(feature = "board-pleco"))]
use crate::board::cozy::Position;
#[cfg(not(feature = "board-pleco"))]
use crate::eval::nnue::loader::QuantNnue;
#[cfg(not(feature = "board-pleco"))]
use crate::eval::nnue::Nnue;
#[cfg(not(feature = "board-pleco"))]
use crate::search::alphabeta::{SearchParams, SearchResult, Searcher};
#[cfg(not(feature = "board-pleco"))]
use cozy_chess::{Color, Piece, Square};
#[cfg(not(feature = "board-pleco"))]
use std::collections::VecDeque;
use std::io::{self, BufRead};
#[cfg(not(feature = "board-pleco"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(feature = "board-pleco"))]
use std::sync::{mpsc, Arc};
#[cfg(not(feature = "board-pleco"))]
use std::thread;
#[cfg(not(feature = "board-pleco"))]
use std::time::Duration;

#[cfg(not(feature = "board-pleco"))]
const ENGINE_NAME: &str = "PieBot NNUE";
#[cfg(not(feature = "board-pleco"))]
const DEFAULT_HASH_MB: usize = 64;
#[cfg(not(feature = "board-pleco"))]
const DEFAULT_GO_MOVETIME_MS: u64 = 1_000;
#[cfg(not(feature = "board-pleco"))]
const MOVE_OVERHEAD_MS: u64 = 10;

#[cfg(not(feature = "board-pleco"))]
#[derive(Debug, Default, PartialEq, Eq)]
struct GoOptions {
    depth: Option<u32>,
    nodes: Option<u64>,
    movetime_ms: Option<u64>,
    wtime_ms: Option<u64>,
    btime_ms: Option<u64>,
    winc_ms: Option<u64>,
    binc_ms: Option<u64>,
    moves_to_go: Option<u64>,
    infinite: bool,
}

#[cfg(not(feature = "board-pleco"))]
impl GoOptions {
    fn parse(args: &str) -> Self {
        let mut options = Self::default();
        let mut tokens = args.split_whitespace();
        while let Some(token) = tokens.next() {
            match token {
                "depth" => {
                    options.depth = tokens
                        .next()
                        .and_then(|value| value.parse::<u32>().ok())
                        .map(|depth| depth.max(1));
                }
                "nodes" => {
                    options.nodes = tokens
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|nodes| nodes.max(1));
                }
                "movetime" => {
                    options.movetime_ms = tokens
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|millis| millis.max(1));
                }
                "wtime" => {
                    options.wtime_ms = tokens.next().and_then(|value| value.parse().ok());
                }
                "btime" => {
                    options.btime_ms = tokens.next().and_then(|value| value.parse().ok());
                }
                "winc" => {
                    options.winc_ms = tokens.next().and_then(|value| value.parse().ok());
                }
                "binc" => {
                    options.binc_ms = tokens.next().and_then(|value| value.parse().ok());
                }
                "movestogo" => {
                    options.moves_to_go = tokens
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|moves| moves.max(1));
                }
                "infinite" => options.infinite = true,
                _ => {}
            }
        }
        options
    }

    fn allocated_time_ms(&self, side_to_move: Color) -> Option<u64> {
        if let Some(movetime_ms) = self.movetime_ms {
            return Some(movetime_ms.max(1));
        }
        let (remaining_ms, increment_ms) = match side_to_move {
            Color::White => (self.wtime_ms?, self.winc_ms.unwrap_or(0)),
            Color::Black => (self.btime_ms?, self.binc_ms.unwrap_or(0)),
        };
        let moves_to_go = self.moves_to_go.unwrap_or(30).max(1);
        let usable_ms = remaining_ms.saturating_sub(MOVE_OVERHEAD_MS);
        let base_ms = usable_ms / moves_to_go;
        let increment_share_ms = increment_ms.saturating_mul(3) / 4;
        let requested_ms = base_ms.saturating_add(increment_share_ms).max(1);
        Some(requested_ms.min(usable_ms.max(1)))
    }

    fn allocated_time_for(&self, board: &cozy_chess::Board) -> Option<u64> {
        if self.movetime_ms.is_some() || self.moves_to_go.is_some() {
            return self.allocated_time_ms(board.side_to_move());
        }

        let (remaining_ms, increment_ms) = match board.side_to_move() {
            Color::White => (self.wtime_ms?, self.winc_ms.unwrap_or(0)),
            Color::Black => (self.btime_ms?, self.binc_ms.unwrap_or(0)),
        };
        let reserve_ms = (remaining_ms / 20).clamp(50, 1_000);
        let usable_ms = remaining_ms.saturating_sub(reserve_ms).max(1);
        let non_king_pieces = (board.occupied().len() as u64).saturating_sub(2);
        let estimated_moves = match non_king_pieces {
            24.. => 24,
            12..=23 => 20,
            6..=11 => 16,
            _ => 12,
        };
        let base_ms = usable_ms / estimated_moves;
        let increment_share_ms = increment_ms.saturating_mul(4) / 5;
        let requested_ms = base_ms.saturating_add(increment_share_ms).max(1);
        Some(requested_ms.min((usable_ms / 3).max(1)))
    }
}

#[cfg(not(feature = "board-pleco"))]
fn search_params_for_go(
    options: &GoOptions,
    board: &cozy_chess::Board,
    threads: usize,
) -> SearchParams {
    let mut params = SearchParams::default();
    params.depth = options.depth.unwrap_or(0);
    params.max_nodes = options.nodes;
    params.use_tt = true;
    params.order_captures = true;
    params.use_history = true;
    params.threads = threads.max(1);
    params.use_aspiration = true;
    params.aspiration_window_cp = 35;
    params.use_lmr = true;
    params.use_killers = true;
    params.use_nullmove = true;
    params.deterministic = params.threads == 1;

    let allocated_ms = if options.infinite {
        None
    } else {
        options.allocated_time_for(board).or_else(|| {
            if options.depth.is_none() && options.nodes.is_none() {
                Some(DEFAULT_GO_MOVETIME_MS)
            } else {
                None
            }
        })
    };
    params.movetime = allocated_ms.map(Duration::from_millis);
    params
}

#[cfg(not(feature = "board-pleco"))]
fn castling_translation(position: &Position, uci: &str, to_cozy: bool) -> Option<&'static str> {
    let (translated, king_square, rook_square) = match (uci, to_cozy) {
        ("e1g1", true) => ("e1h1", Square::E1, Square::H1),
        ("e1c1", true) => ("e1a1", Square::E1, Square::A1),
        ("e8g8", true) => ("e8h8", Square::E8, Square::H8),
        ("e8c8", true) => ("e8a8", Square::E8, Square::A8),
        ("e1h1", false) => ("e1g1", Square::E1, Square::H1),
        ("e1a1", false) => ("e1c1", Square::E1, Square::A1),
        ("e8h8", false) => ("e8g8", Square::E8, Square::H8),
        ("e8a8", false) => ("e8c8", Square::E8, Square::A8),
        _ => return None,
    };
    let board = position.board();
    let side = board.side_to_move();
    let our_pieces = board.colors(side);
    if board.piece_on(king_square) == Some(Piece::King)
        && board.piece_on(rook_square) == Some(Piece::Rook)
        && our_pieces.has(king_square)
        && our_pieces.has(rook_square)
    {
        Some(translated)
    } else {
        None
    }
}

#[cfg(not(feature = "board-pleco"))]
fn normalize_uci_move(position: &Position, uci: &str) -> String {
    castling_translation(position, uci, true)
        .unwrap_or(uci)
        .to_string()
}

#[cfg(not(feature = "board-pleco"))]
fn format_uci_move(position: &Position, cozy_move: &str) -> String {
    castling_translation(position, cozy_move, false)
        .unwrap_or(cozy_move)
        .to_string()
}

#[cfg(not(feature = "board-pleco"))]
fn apply_uci_moves(mut position: Position, moves: &[String]) -> Result<Position, String> {
    for raw_move in moves {
        let normalized = normalize_uci_move(&position, raw_move);
        position.make_move_uci(&normalized)?;
    }
    Ok(position)
}

#[cfg(feature = "board-pleco")]
mod pleco_uci {
    use super::*;
    use crate::search::alphabeta_pleco::PlecoSearcher;
    use pleco::{BitMove as PMove, Board as PBoard};
    use rayon::ThreadPoolBuilder;

    fn move_to_uci(m: PMove) -> String {
        format!("{}", m)
    }
    fn uci_to_move(board: &PBoard, uci: &str) -> Option<PMove> {
        let ml = board.generate_moves();
        for m in ml.iter() {
            if move_to_uci(*m) == uci {
                return Some(*m);
            }
        }
        None
    }

    pub struct UciEnginePleco {
        board: PBoard,
        threads: usize,
        hash_mb: usize,
        searcher: PlecoSearcher,
    }
    impl UciEnginePleco {
        pub fn new() -> Self {
            Self {
                board: PBoard::start_pos(),
                threads: 1,
                hash_mb: 64,
                searcher: PlecoSearcher::default(),
            }
        }
        fn cmd_uci(&self) {
            println!("id name PieBot (Pleco)");
            println!("id author PieBot Team");
            println!("option name Threads type spin default 1 min 1 max 512");
            println!("option name Hash type spin default 64 min 1 max 4096");
            println!("uciok");
        }
        fn cmd_isready(&self) {
            println!("readyok");
        }
        fn cmd_ucinewgame(&mut self) {
            self.board = PBoard::start_pos();
            self.searcher.clear();
        }
        fn apply_setoption(&mut self, name: &str, value: &str) {
            match name.to_lowercase().as_str() {
                "threads" => {
                    if let Ok(t) = value.parse::<usize>() {
                        self.threads = t.max(1);
                    }
                }
                "hash" => {
                    if let Ok(mb) = value.parse::<usize>() {
                        self.hash_mb = mb.max(1);
                        self.searcher.set_tt_capacity_mb(self.hash_mb);
                    }
                }
                _ => {}
            }
        }
        fn cmd_setoption(&mut self, args: &str) {
            let mut it = args.split_whitespace();
            if it.next() != Some("name") {
                return;
            }
            let mut name = Vec::new();
            let mut val = None;
            for tok in it {
                if tok == "value" {
                    val = Some(String::new());
                    continue;
                }
                if let Some(v) = val.as_mut() {
                    if !v.is_empty() {
                        v.push(' ');
                    }
                    v.push_str(tok);
                } else {
                    name.push(tok.to_string());
                }
            }
            self.apply_setoption(&name.join(" "), &val.unwrap_or_default());
        }
        fn cmd_position(&mut self, args: &str) {
            let mut it = args.split_whitespace();
            match it.next() {
                Some("startpos") => {
                    self.board = PBoard::start_pos();
                    if let Some("moves") = it.next() {
                        for m in it {
                            if let Some(bm) = uci_to_move(&self.board, m) {
                                self.board.apply_move(bm);
                            }
                        }
                    }
                }
                Some("fen") => {
                    let fen: Vec<&str> = it.by_ref().take(6).collect();
                    if fen.len() == 6 {
                        if let Ok(b) = PBoard::from_fen(&fen.join(" ")) {
                            self.board = b;
                        }
                    }
                    if let Some("moves") = it.next() {
                        for m in it {
                            if let Some(bm) = uci_to_move(&self.board, m) {
                                self.board.apply_move(bm);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        fn cmd_go(&mut self, args: &str) {
            let mut depth: u32 = 6;
            let mut movetime: Option<u64> = None;
            let mut it = args.split_whitespace();
            while let Some(t) = it.next() {
                match t {
                    "depth" => {
                        if let Some(d) = it.next().and_then(|s| s.parse().ok()) {
                            depth = d
                        }
                    }
                    "movetime" => {
                        if let Some(ms) = it.next().and_then(|s| s.parse().ok()) {
                            movetime = Some(ms)
                        }
                    }
                    _ => {}
                }
            }
            // Ensure TT size
            self.searcher.set_tt_capacity_mb(self.hash_mb);
            self.searcher.set_threads(self.threads);
            let pool = ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build()
                .unwrap();
            let (best, _sc, _nodes) = pool.install(|| {
                if let Some(ms) = movetime {
                    self.searcher
                        .search_movetime(&mut self.board.clone(), ms, depth)
                } else {
                    self.searcher
                        .search_movetime(&mut self.board.clone(), 1000, depth)
                }
            });
            if let Some(bm) = best {
                println!("bestmove {}", move_to_uci(bm));
            } else {
                println!("bestmove 0000");
            }
        }
        pub fn run_loop(&mut self) {
            let stdin = io::stdin();
            for line in stdin.lock().lines() {
                let line = match line {
                    Ok(s) => s.trim().to_string(),
                    Err(_) => break,
                };
                if line.is_empty() {
                    continue;
                }
                if line == "uci" {
                    self.cmd_uci();
                    continue;
                }
                if line == "isready" {
                    self.cmd_isready();
                    continue;
                }
                if line == "ucinewgame" {
                    self.cmd_ucinewgame();
                    continue;
                }
                if let Some(rest) = line.strip_prefix("setoption ") {
                    self.cmd_setoption(rest);
                    continue;
                }
                if line == "quit" {
                    break;
                }
                if let Some(rest) = line.strip_prefix("position ") {
                    self.cmd_position(rest);
                    continue;
                }
                if let Some(rest) = line.strip_prefix("go ") {
                    self.cmd_go(rest);
                    continue;
                }
                if line == "stop" {
                    continue;
                }
            }
        }
    }
}

#[cfg(feature = "board-pleco")]
pub use pleco_uci::UciEnginePleco as UciEngine;

#[cfg(not(feature = "board-pleco"))]
pub struct UciEngine {
    pos: Position,
    searcher: Searcher,
    hash_mb: usize,
    threads: usize,
    use_nnue: bool,
    nnue_loaded: bool,
}

#[cfg(not(feature = "board-pleco"))]
impl Default for UciEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "board-pleco"))]
struct SearchOutcome {
    searcher: Searcher,
    position: Position,
    result: SearchResult,
}

#[cfg(not(feature = "board-pleco"))]
struct ActiveSearch {
    stop: Arc<AtomicBool>,
    outcome_rx: mpsc::Receiver<SearchOutcome>,
}

#[cfg(not(feature = "board-pleco"))]
impl UciEngine {
    pub fn new() -> Self {
        let mut searcher = Searcher::default();
        searcher.set_tt_capacity_mb(DEFAULT_HASH_MB);
        Self {
            pos: Position::startpos(),
            searcher,
            hash_mb: DEFAULT_HASH_MB,
            threads: 1,
            use_nnue: false,
            nnue_loaded: false,
        }
    }

    fn cmd_uci(&self) {
        println!("id name {ENGINE_NAME}");
        println!("id author PieBot Team");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("option name Hash type spin default 64 min 1 max 16384");
        println!("option name UseNNUE type check default false");
        println!("option name NNUEFile type string default ");
        println!("option name NNUEQuantFile type string default ");
        println!("option name EvalBlend type spin default 100 min 0 max 100");
        println!("uciok");
    }

    fn cmd_isready(&self) {
        println!("readyok");
    }

    fn cmd_ucinewgame(&mut self) {
        self.pos = Position::startpos();
        self.searcher.set_tt_capacity_mb(self.hash_mb);
    }

    pub(crate) fn apply_setoption(&mut self, name: &str, value: &str) -> Option<String> {
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.hash_mb = mb.clamp(1, 16_384);
                    self.searcher.set_tt_capacity_mb(self.hash_mb);
                }
                None
            }
            "threads" => {
                if let Ok(t) = value.parse::<usize>() {
                    self.threads = t.clamp(1, 512);
                }
                None
            }
            "usennue" => {
                let on = matches!(value.to_lowercase().as_str(), "true" | "1" | "on" | "yes");
                self.use_nnue = on;
                self.searcher.set_use_nnue(on && self.nnue_loaded);
                None
            }
            "nnuefile" => {
                // Attempt to load the dense-f32 dev format (PIENNUE1)
                match Nnue::load(value) {
                    Ok(nn) => {
                        if crate::eval::nnue::features::HalfKpSchema::from_input_dim(
                            nn.meta.input_dim,
                        )
                        .is_none()
                        {
                            let legacy = crate::eval::nnue::features::halfkp_dim();
                            let full = crate::eval::nnue::features::halfkp_v2_dim();
                            return Some(format!(
                                "info string failed to load NNUEFile: incompatible input_dim {}; expected supported HalfKP input_dim {legacy} or {full}",
                                nn.meta.input_dim
                            ));
                        }
                        self.searcher.clear_nnue_quant();
                        self.searcher.set_nnue_network(Some(nn));
                        self.nnue_loaded = true;
                        self.searcher.set_use_nnue(self.use_nnue);
                        None
                    }
                    Err(error) => Some(format!("info string failed to load NNUEFile: {error}")),
                }
            }
            "nnuequantfile" => match QuantNnue::load_quantized(value) {
                Ok(model) => {
                    if crate::eval::nnue::features::HalfKpSchema::from_input_dim(
                        model.meta.input_dim,
                    )
                    .is_none()
                    {
                        let legacy = crate::eval::nnue::features::halfkp_dim();
                        let full = crate::eval::nnue::features::halfkp_v2_dim();
                        return Some(format!(
                            "info string failed to load NNUEQuantFile: incompatible input_dim {}; expected supported HalfKP input_dim {legacy} or {full}",
                            model.meta.input_dim
                        ));
                    }
                    self.searcher.set_nnue_network(None);
                    self.searcher.set_nnue_quant_model(model);
                    self.nnue_loaded = true;
                    self.searcher.set_use_nnue(self.use_nnue);
                    None
                }
                Err(error) => Some(format!("info string failed to load NNUEQuantFile: {error}")),
            },
            "evalblend" => {
                if let Ok(p) = value.parse::<u8>() {
                    self.searcher.set_eval_blend_percent(p);
                }
                None
            }
            _ => None,
        }
    }

    fn cmd_position(&mut self, args: &str) {
        // Supports: 'position startpos [moves ...]' and 'position fen <fen> [moves ...]'
        let mut tokens = args.split_whitespace();
        match tokens.next() {
            Some("startpos") => {
                let moves = if let Some("moves") = tokens.next() {
                    tokens.map(str::to_string).collect()
                } else {
                    Vec::new()
                };
                if let Ok(position) = apply_uci_moves(Position::startpos(), &moves) {
                    self.pos = position;
                }
            }
            Some("fen") => {
                // FEN is 6 fields; collect them
                let fen_fields: Vec<&str> = tokens.by_ref().take(6).collect();
                if fen_fields.len() == 6 {
                    let fen = fen_fields.join(" ");
                    let moves = if let Some("moves") = tokens.next() {
                        tokens.map(str::to_string).collect()
                    } else {
                        Vec::new()
                    };
                    if let Ok(position) = Position::from_fen(&fen)
                        .and_then(|position| apply_uci_moves(position, &moves))
                    {
                        self.pos = position;
                    }
                }
            }
            _ => {}
        }
    }

    fn cmd_setoption(&mut self, args: &str) {
        // setoption name <name> [value <value>]
        let mut tokens = args.split_whitespace();
        if tokens.next() != Some("name") {
            return;
        }
        let mut name_parts = Vec::new();
        let mut value: Option<String> = None;
        for tok in tokens {
            if tok == "value" {
                value = Some(String::new());
                continue;
            }
            if let Some(v) = value.as_mut() {
                if !v.is_empty() {
                    v.push(' ');
                }
                v.push_str(tok);
            } else {
                name_parts.push(tok.to_string());
            }
        }
        let name = name_parts.join(" ");
        let val = value.unwrap_or_default();
        if let Some(message) = self.apply_setoption(&name, &val) {
            println!("{message}");
        }
    }

    fn start_search(&mut self, args: &str) -> ActiveSearch {
        let options = GoOptions::parse(args);
        let params = search_params_for_go(&options, self.pos.board(), self.threads);
        let position = self.pos.clone();
        let mut searcher = std::mem::take(&mut self.searcher);
        searcher.set_position_history(position.history());

        let stop = Arc::new(AtomicBool::new(false));
        searcher.set_stop_flag(Some(stop.clone()));
        let (outcome_tx, outcome_rx) = mpsc::channel();
        thread::Builder::new()
            .name("piebot-uci-search".to_string())
            .spawn(move || {
                let result = searcher.search_with_params(position.board(), params);
                searcher.clear_stop_flag();
                let _ = outcome_tx.send(SearchOutcome {
                    searcher,
                    position,
                    result,
                });
            })
            .expect("failed to start UCI search worker");

        ActiveSearch { stop, outcome_rx }
    }

    fn finish_search(&mut self, outcome: SearchOutcome) {
        self.searcher = outcome.searcher;
        let result = outcome.result;
        println!(
            "info depth {} seldepth {} score cp {} nodes {}",
            self.searcher.last_depth(),
            self.searcher.last_seldepth(),
            result.score_cp,
            result.nodes
        );
        if let Some(best) = result.bestmove {
            println!("bestmove {}", format_uci_move(&outcome.position, &best));
        } else {
            println!("bestmove 0000");
        }
    }

    pub fn run_loop(&mut self) {
        // Reading commands on a dedicated thread lets the protocol loop observe
        // `stop` while search is running. Search itself remains owned by one
        // worker and cooperatively polls the shared stop flag at every node.
        let (command_tx, command_rx) = mpsc::channel();
        thread::Builder::new()
            .name("piebot-uci-input".to_string())
            .spawn(move || {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    match line {
                        Ok(line) => {
                            if command_tx.send(line.trim().to_string()).is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
            .expect("failed to start UCI input worker");

        let mut active: Option<ActiveSearch> = None;
        let mut pending = VecDeque::new();
        let mut quitting = false;

        loop {
            if let Some(search) = active.as_ref() {
                match search.outcome_rx.try_recv() {
                    Ok(outcome) => {
                        active = None;
                        self.finish_search(outcome);
                        if quitting {
                            break;
                        }
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        active = None;
                        println!("info string search worker terminated unexpectedly");
                        println!("bestmove 0000");
                        if quitting {
                            break;
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }

            let line = if active.is_some() {
                match command_rx.recv_timeout(Duration::from_millis(5)) {
                    Ok(line) => line,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        if let Some(search) = active.as_ref() {
                            search.stop.store(true, Ordering::Relaxed);
                            quitting = true;
                            continue;
                        }
                        break;
                    }
                }
            } else if let Some(line) = pending.pop_front() {
                line
            } else {
                match command_rx.recv() {
                    Ok(line) => line,
                    Err(_) => break,
                }
            };

            if line.is_empty() {
                continue;
            }

            if let Some(search) = active.as_ref() {
                match line.as_str() {
                    "stop" => search.stop.store(true, Ordering::Relaxed),
                    "quit" => {
                        search.stop.store(true, Ordering::Relaxed);
                        quitting = true;
                    }
                    "isready" if pending.is_empty() => self.cmd_isready(),
                    "isready" => pending.push_back(line),
                    "uci" => self.cmd_uci(),
                    _ => {
                        // State-changing commands must be applied to the real
                        // searcher after it returns from the worker thread.
                        search.stop.store(true, Ordering::Relaxed);
                        pending.push_back(line);
                    }
                }
                continue;
            }

            if line == "uci" {
                self.cmd_uci();
                continue;
            }
            if line == "isready" {
                self.cmd_isready();
                continue;
            }
            if line == "ucinewgame" {
                self.cmd_ucinewgame();
                continue;
            }
            if let Some(rest) = line.strip_prefix("setoption ") {
                self.cmd_setoption(rest);
                continue;
            }
            if line == "quit" {
                break;
            }
            if let Some(rest) = line.strip_prefix("position ") {
                self.cmd_position(rest);
                continue;
            }
            if line == "go" {
                active = Some(self.start_search(""));
                continue;
            }
            if let Some(rest) = line.strip_prefix("go ") {
                active = Some(self.start_search(rest));
                continue;
            }
            if line == "stop" {
                continue;
            }
        }
    }
}

#[cfg(all(test, not(feature = "board-pleco")))]
mod tests {
    use super::*;
    use crate::eval::nnue::features::{halfkp_dim, halfkp_v2_dim};
    use cozy_chess::{Piece, Square};
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn quant_model_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "piebot_uci_{name}_{}_{}.nnue",
            std::process::id(),
            nanos
        ))
    }

    fn write_quant_model(path: &PathBuf, input_dim: usize, output_bias: i16) {
        let hidden_dim = 1usize;
        let mut file = File::create(path).expect("create quant model");
        file.write_all(b"PIENNQ01").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(input_dim as u32).to_le_bytes()).unwrap();
        file.write_all(&(hidden_dim as u32).to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.write_all(&1.0f32.to_le_bytes()).unwrap();
        file.write_all(&vec![0u8; input_dim * hidden_dim]).unwrap();
        file.write_all(&0i16.to_le_bytes()).unwrap();
        file.write_all(&[0u8]).unwrap();
        file.write_all(&output_bias.to_le_bytes()).unwrap();
    }

    fn write_dense_model(path: &PathBuf, input_dim: usize, output_bias: f32) {
        let hidden_dim = 1usize;
        let mut file = File::create(path).expect("create dense model");
        file.write_all(b"PIENNUE1").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&(input_dim as u32).to_le_bytes()).unwrap();
        file.write_all(&(hidden_dim as u32).to_le_bytes()).unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&vec![0u8; input_dim * hidden_dim * 4])
            .unwrap();
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&0.0f32.to_le_bytes()).unwrap();
        file.write_all(&output_bias.to_le_bytes()).unwrap();
    }

    #[test]
    fn engine_defaults_match_advertised_hash_and_identity() {
        let engine = UciEngine::new();
        assert_eq!(engine.hash_mb, DEFAULT_HASH_MB);
        assert_eq!(ENGINE_NAME, "PieBot NNUE");
        assert!(!ENGINE_NAME.to_ascii_lowercase().contains("skeleton"));
    }

    #[test]
    fn movetime_search_is_iterative_and_uses_standard_heuristics() {
        let go = GoOptions::parse("movetime 25");
        let params = search_params_for_go(&go, &cozy_chess::Board::default(), 1);

        assert_eq!(
            params.depth, 0,
            "timed search must not be capped at depth 6"
        );
        assert_eq!(params.movetime, Some(Duration::from_millis(25)));
        assert!(params.use_tt);
        assert!(params.order_captures);
        assert!(params.use_history);
        assert!(params.use_aspiration);
        assert!(params.use_lmr);
        assert!(params.use_killers);
        assert!(params.use_nullmove);
        assert!(params.deterministic);
    }

    #[test]
    fn depth_search_has_an_exact_depth_and_no_deadline() {
        let go = GoOptions::parse("depth 4");
        let params = search_params_for_go(&go, &cozy_chess::Board::default(), 1);

        assert_eq!(params.depth, 4);
        assert_eq!(params.movetime, None);
    }

    #[test]
    fn infinite_and_node_limited_go_commands_do_not_get_a_default_deadline() {
        let board = cozy_chess::Board::default();
        let infinite = search_params_for_go(&GoOptions::parse("infinite"), &board, 1);
        let nodes = search_params_for_go(&GoOptions::parse("nodes 12345"), &board, 1);

        assert_eq!(infinite.depth, 0);
        assert_eq!(infinite.movetime, None);
        assert_eq!(infinite.max_nodes, None);
        assert_eq!(nodes.max_nodes, Some(12_345));
        assert_eq!(nodes.movetime, None);
    }

    #[test]
    fn clock_allocation_uses_side_to_move_clock_and_increment() {
        let go = GoOptions::parse("wtime 60000 btime 30000 winc 1000 binc 0 movestogo 30");
        let white = cozy_chess::Board::default();
        let black = cozy_chess::Board::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            false,
        )
        .unwrap();

        assert_eq!(go.allocated_time_for(&white), Some(2749));
        assert_eq!(go.allocated_time_for(&black), Some(999));
    }

    #[test]
    fn automatic_clock_allocation_uses_more_time_late_and_keeps_a_reserve() {
        let go = GoOptions::parse("wtime 60000 btime 60000 winc 500 binc 500");
        let opening = cozy_chess::Board::default();
        let ending =
            cozy_chess::Board::from_fen("8/8/8/3k4/8/4K3/4P3/8 w - - 0 50", false).unwrap();

        let opening_ms = go.allocated_time_for(&opening).unwrap();
        let ending_ms = go.allocated_time_for(&ending).unwrap();
        assert!((2_500..=3_500).contains(&opening_ms), "{opening_ms}");
        assert!(
            ending_ms > opening_ms,
            "opening={opening_ms} ending={ending_ms}"
        );
        assert!(
            ending_ms < 30_000,
            "must preserve recovery time: {ending_ms}"
        );
    }

    #[test]
    fn invalid_nnue_file_reports_a_uci_info_string() {
        let mut engine = UciEngine::new();
        let message = engine
            .apply_setoption("NNUEQuantFile", "/definitely/missing/piebot-network.nnue")
            .expect("invalid model load should be visible to the GUI");

        assert!(message.starts_with("info string failed to load NNUEQuantFile:"));
        assert!(!engine.nnue_loaded);
    }

    #[test]
    fn incompatible_nnue_models_are_rejected_without_replacing_active_model() {
        let valid_path = quant_model_path("valid_halfkp");
        let incompatible_path = quant_model_path("wrong_dimension");
        let incompatible_dense_path = quant_model_path("wrong_dense_dimension");
        write_quant_model(&valid_path, halfkp_dim(), 73);
        write_quant_model(&incompatible_path, 12, -41);
        write_dense_model(&incompatible_dense_path, 12, -29.0);

        let mut engine = UciEngine::new();
        assert_eq!(
            engine.apply_setoption("NNUEQuantFile", valid_path.to_str().unwrap()),
            None
        );
        engine.apply_setoption("UseNNUE", "true");
        let score_before = engine.searcher.qsearch_eval_cp(engine.pos.board());

        let message = engine
            .apply_setoption("NNUEQuantFile", incompatible_path.to_str().unwrap())
            .expect("incompatible model must be reported to the GUI");
        let dense_message = engine
            .apply_setoption("NNUEFile", incompatible_dense_path.to_str().unwrap())
            .expect("incompatible dense model must be reported to the GUI");
        let score_after_rejections = engine.searcher.qsearch_eval_cp(engine.pos.board());

        let _ = std::fs::remove_file(valid_path);
        let _ = std::fs::remove_file(incompatible_path);
        let _ = std::fs::remove_file(incompatible_dense_path);
        assert!(message.starts_with("info string failed to load NNUEQuantFile:"));
        assert!(message.contains("input_dim"));
        assert!(dense_message.starts_with("info string failed to load NNUEFile:"));
        assert!(dense_message.contains("input_dim"));
        assert_eq!(
            score_after_rejections, score_before,
            "active model must be preserved"
        );
        assert!(engine.nnue_loaded);
    }

    #[test]
    fn loading_dense_nnue_replaces_a_previously_loaded_quant_model() {
        let quant_path = quant_model_path("replace_quant");
        let dense_path = quant_model_path("replace_dense");
        write_quant_model(&quant_path, halfkp_dim(), 73);
        write_dense_model(&dense_path, halfkp_dim(), -29.0);

        let mut engine = UciEngine::new();
        assert_eq!(
            engine.apply_setoption("NNUEQuantFile", quant_path.to_str().unwrap()),
            None
        );
        engine.apply_setoption("UseNNUE", "true");
        assert_eq!(
            engine.searcher.qsearch_eval_cp(engine.pos.board()),
            73,
            "quantized fixture should be active"
        );

        assert_eq!(
            engine.apply_setoption("NNUEFile", dense_path.to_str().unwrap()),
            None
        );
        let score_after_dense_load = engine.searcher.qsearch_eval_cp(engine.pos.board());

        let _ = std::fs::remove_file(quant_path);
        let _ = std::fs::remove_file(dense_path);
        assert_eq!(
            score_after_dense_load, -29,
            "new dense model must replace the old quantized model"
        );
    }

    #[test]
    fn uci_accepts_full_perspective_nnue_models() {
        let quant_path = quant_model_path("full_perspective_quant");
        let dense_path = quant_model_path("full_perspective_dense");
        write_quant_model(&quant_path, halfkp_v2_dim(), 37);
        write_dense_model(&dense_path, halfkp_v2_dim(), -19.0);

        let mut engine = UciEngine::new();
        assert_eq!(
            engine.apply_setoption("NNUEQuantFile", quant_path.to_str().unwrap()),
            None
        );
        engine.apply_setoption("UseNNUE", "true");
        assert_eq!(engine.searcher.qsearch_eval_cp(engine.pos.board()), 37);

        assert_eq!(
            engine.apply_setoption("NNUEFile", dense_path.to_str().unwrap()),
            None
        );
        assert_eq!(engine.searcher.qsearch_eval_cp(engine.pos.board()), -19);

        let _ = std::fs::remove_file(quant_path);
        let _ = std::fs::remove_file(dense_path);
    }

    #[test]
    fn standard_uci_castling_is_translated_to_and_from_cozy_encoding() {
        let start = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            .expect("valid castling position");
        assert_eq!(normalize_uci_move(&start, "e1g1"), "e1h1");
        assert_eq!(format_uci_move(&start, "e1h1"), "e1g1");

        let castled = apply_uci_moves(start, &["e1g1".to_string()])
            .expect("standard UCI castling move should apply");
        assert_eq!(castled.board().piece_on(Square::G1), Some(Piece::King));
        assert_eq!(castled.board().piece_on(Square::F1), Some(Piece::Rook));
    }
}
