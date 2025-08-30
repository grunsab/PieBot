use std::io::{self, BufRead};
#[cfg(not(feature = "board-pleco"))]
use crate::board::cozy::Position;
#[cfg(not(feature = "board-pleco"))]
use crate::eval::nnue::Nnue;
#[cfg(not(feature = "board-pleco"))]
use crate::eval::nnue::loader::QuantNnue;
#[cfg(not(feature = "board-pleco"))]
use crate::search::alphabeta::{Searcher, SearchParams};
#[cfg(not(feature = "board-pleco"))]
use std::time::Duration;

#[cfg(feature = "board-pleco")]
mod pleco_uci {
    use super::*;
    use pleco::{Board as PBoard, BitMove as PMove};
    use rayon::ThreadPoolBuilder;
    use crate::search::alphabeta_pleco::PlecoSearcher;

    fn move_to_uci(m: PMove) -> String { format!("{}", m) }
    fn uci_to_move(board: &PBoard, uci: &str) -> Option<PMove> {
        let ml = board.generate_moves();
        for m in ml.iter() { if move_to_uci(*m) == uci { return Some(*m); } }
        None
    }

    pub struct UciEnginePleco {
        board: PBoard,
        threads: usize,
        hash_mb: usize,
        searcher: PlecoSearcher,
        smp_mode: crate::search::alphabeta_pleco::SmpMode,
        tm_finish_one: bool,
        tm_factor: f32,
    }
    impl UciEnginePleco {
        pub fn new() -> Self { Self { board: PBoard::start_pos(), threads: 1, hash_mb: 64, searcher: PlecoSearcher::default(), smp_mode: crate::search::alphabeta_pleco::SmpMode::InTree, tm_finish_one: true, tm_factor: 1.9 } }
        fn cmd_uci(&self) {
            println!("id name PieBot (Pleco)"); println!("id author PieBot Team");
            println!("option name Threads type spin default 1 min 1 max 512");
            println!("option name Hash type spin default 64 min 1 max 4096");
            println!("option name SMPMode type combo default in-tree var off var in-tree var lazy-indep var lazy-coop var lazy-hybrid");
            println!("option name TMPolicy type combo default finish var finish var spend");
            println!("option name TMFactor type spin default 1.9 min 0 max 10");
            println!("uciok");
        }
        fn cmd_isready(&self) { println!("readyok"); }
        fn cmd_ucinewgame(&mut self) { self.board = PBoard::start_pos(); self.searcher.clear(); }
        fn apply_setoption(&mut self, name:&str, value:&str) {
            match name.to_lowercase().as_str() {
                "threads" => if let Ok(t)=value.parse::<usize>(){ self.threads=t.max(1);} ,
                "hash" => if let Ok(mb)=value.parse::<usize>(){ self.hash_mb = mb.max(1); self.searcher.set_tt_capacity_mb(self.hash_mb); },
                "smpmode" => {
                    let mode = match value.to_ascii_lowercase().as_str() {
                        "off" => crate::search::alphabeta_pleco::SmpMode::Off,
                        "in-tree" => crate::search::alphabeta_pleco::SmpMode::InTree,
                        "lazy-indep" => crate::search::alphabeta_pleco::SmpMode::LazyIndep,
                        "lazy-coop" => crate::search::alphabeta_pleco::SmpMode::LazyCoop,
                        "lazy-hybrid" => crate::search::alphabeta_pleco::SmpMode::LazyHybrid,
                        _ => crate::search::alphabeta_pleco::SmpMode::InTree,
                    };
                    self.smp_mode = mode;
                },
                "tmpolicy" => {
                    self.tm_finish_one = match value.to_ascii_lowercase().as_str() { "spend" => false, _ => true };
                },
                "tmfactor" => {
                    if let Ok(f) = value.parse::<f32>() { self.tm_factor = f; }
                },
                _=>{}
            }
        }
        fn cmd_setoption(&mut self, args:&str){ let mut it=args.split_whitespace(); if it.next()!=Some("name"){return;} let mut name=Vec::new(); let mut val=None; for tok in it{ if tok=="value"{ val=Some(String::new()); continue;} if let Some(v)=val.as_mut(){ if !v.is_empty(){v.push(' ');} v.push_str(tok);} else {name.push(tok.to_string());}} self.apply_setoption(&name.join(" "), &val.unwrap_or_default()); }
        fn cmd_position(&mut self, args:&str){ let mut it=args.split_whitespace(); match it.next(){ Some("startpos")=>{ self.board=PBoard::start_pos(); if let Some("moves")=it.next(){ for m in it { if let Some(bm)=uci_to_move(&self.board, m){ self.board.apply_move(bm);} } } }, Some("fen")=>{ let fen: Vec<&str>=it.by_ref().take(6).collect(); if fen.len()==6{ if let Ok(b)=PBoard::from_fen(&fen.join(" ")){ self.board=b; } } if let Some("moves")=it.next(){ for m in it { if let Some(bm)=uci_to_move(&self.board, m){ self.board.apply_move(bm);} } } }, _=>{} } }
        fn cmd_go(&mut self, args:&str){
            let mut depth: u32=6; let mut movetime: Option<u64>=None; let mut it=args.split_whitespace();
            while let Some(t)=it.next(){ match t{ "depth"=> if let Some(d)=it.next().and_then(|s|s.parse().ok()){ depth=d }, "movetime"=> if let Some(ms)=it.next().and_then(|s|s.parse().ok()){ movetime=Some(ms)}, _=>{} } }
            // Ensure TT size
            self.searcher.set_tt_capacity_mb(self.hash_mb);
            self.searcher.set_threads(self.threads);
            self.searcher.set_smp_mode(self.smp_mode);
            self.searcher.set_time_manager(self.tm_finish_one, self.tm_factor);
            let pool=ThreadPoolBuilder::new().num_threads(self.threads).build().unwrap();
            let (best,_sc,_nodes)=pool.install(||{
                if let Some(ms)=movetime{ self.searcher.search_movetime(&mut self.board.clone(), ms, depth)} else { self.searcher.search_movetime(&mut self.board.clone(), 1000, depth)}
            });
            if let Some(bm)=best{ println!("bestmove {}", move_to_uci(bm)); } else { println!("bestmove 0000"); }
        }
        pub fn run_loop(&mut self){ let stdin=io::stdin(); for line in stdin.lock().lines(){ let line=match line{Ok(s)=>s.trim().to_string(),Err(_)=>break}; if line.is_empty(){continue;} if line=="uci"{ self.cmd_uci(); continue;} if line=="isready"{ self.cmd_isready(); continue;} if line=="ucinewgame"{ self.cmd_ucinewgame(); continue;} if let Some(rest)=line.strip_prefix("setoption "){ self.cmd_setoption(rest); continue;} if line=="quit"{ break;} if let Some(rest)=line.strip_prefix("position "){ self.cmd_position(rest); continue;} if let Some(rest)=line.strip_prefix("go "){ self.cmd_go(rest); continue;} if line=="stop"{ continue;} } }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::search::alphabeta_pleco::SmpMode;

        #[test]
        fn setoptions_update_internal_state() {
            let mut eng = UciEnginePleco::new();
            eng.apply_setoption("Threads", "4");
            eng.apply_setoption("Hash", "128");
            eng.apply_setoption("SMPMode", "lazy-hybrid");
            eng.apply_setoption("TMPolicy", "spend");
            eng.apply_setoption("TMFactor", "2.5");
            assert_eq!(eng.threads, 4);
            assert_eq!(eng.hash_mb, 128);
            assert_eq!(eng.smp_mode, SmpMode::LazyHybrid);
            assert_eq!(eng.tm_finish_one, false);
            assert!((eng.tm_factor - 2.5).abs() < 1e-6);
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
impl UciEngine {
    pub fn new() -> Self { Self { pos: Position::startpos(), searcher: Searcher::default(), hash_mb: 64, threads: 1, use_nnue: false, nnue_loaded: false } }

    fn cmd_uci(&self) {
        println!("id name PieBot NNUE (skeleton)");
        println!("id author PieBot Team");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("option name Hash type spin default 64 min 1 max 16384");
        println!("option name UseNNUE type check default false");
        println!("option name NNUEFile type string default ");
        println!("option name NNUEQuantFile type string default ");
        println!("option name EvalBlend type spin default 100 min 0 max 100");
        println!("uciok");
    }

    fn cmd_isready(&self) { println!("readyok"); }

    fn cmd_ucinewgame(&mut self) { self.pos = Position::startpos(); }

    pub(crate) fn apply_setoption(&mut self, name: &str, value: &str) {
        match name.to_lowercase().as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() { self.hash_mb = mb; self.searcher.set_tt_capacity_mb(mb); }
            }
            "threads" => {
                if let Ok(t) = value.parse::<usize>() { self.threads = t.max(1); }
            }
            "usennue" => {
                let on = matches!(value.to_lowercase().as_str(), "true" | "1" | "on" | "yes");
                self.use_nnue = on;
                self.searcher.set_use_nnue(on && self.nnue_loaded);
            }
            "nnuefile" => {
                // Attempt to load the dense-f32 dev format (PIENNUE1)
                match Nnue::load(value) {
                    Ok(nn) => {
                        self.searcher.set_nnue_network(Some(nn));
                        self.nnue_loaded = true;
                        self.searcher.set_use_nnue(self.use_nnue);
                    }
                    Err(_e) => {
                        // Ignore errors silently for now
                    }
                }
            }
            "nnuequantfile" => {
                match QuantNnue::load_quantized(value) {
                    Ok(model) => {
                        self.searcher.set_nnue_quant_model(model);
                        self.nnue_loaded = true;
                        self.searcher.set_use_nnue(self.use_nnue);
                    }
                    Err(_e) => {
                        // Ignore errors silently for now
                    }
                }
            }
            "evalblend" => {
                if let Ok(p) = value.parse::<u8>() {
                    self.searcher.set_eval_blend_percent(p);
                }
            }
            _ => {}
        }
    }

    fn cmd_position(&mut self, args: &str) {
        // Supports: 'position startpos [moves ...]' and 'position fen <fen> [moves ...]'
        let mut tokens = args.split_whitespace();
        match tokens.next() {
            Some("startpos") => {
                self.pos = Position::startpos();
                if let Some("moves") = tokens.next() {
                    let moves: Vec<String> = tokens.map(|s| s.to_string()).collect();
                    if let Ok(p) = Position::set_from_start_and_moves(&moves) {
                        self.pos = p;
                    }
                }
            }
            Some("fen") => {
                // FEN is 6 fields; collect them
                let fen_fields: Vec<&str> = tokens.by_ref().take(6).collect();
                if fen_fields.len() == 6 {
                    let fen = fen_fields.join(" ");
                    if let Ok(p) = Position::from_fen(&fen) { self.pos = p; }
                }
                if let Some("moves") = tokens.next() {
                    let moves: Vec<String> = tokens.map(|s| s.to_string()).collect();
                    if let Ok(p) = Position::set_from_start_and_moves(&moves) {
                        self.pos = p;
                    }
                }
            }
            _ => {}
        }
    }

    fn cmd_setoption(&mut self, args: &str) {
        // setoption name <name> [value <value>]
        let mut tokens = args.split_whitespace();
        if tokens.next() != Some("name") { return; }
        let mut name_parts = Vec::new();
        let mut value: Option<String> = None;
        for tok in tokens {
            if tok == "value" {
                value = Some(String::new());
                continue;
            }
            if let Some(v) = value.as_mut() { if !v.is_empty() { v.push(' '); } v.push_str(tok); } else { name_parts.push(tok.to_string()); }
        }
        let name = name_parts.join(" ");
        let val = value.unwrap_or_else(|| "".to_string());
        self.apply_setoption(&name, &val);
    }

    fn cmd_go(&mut self, args: &str) {
        // Support minimal: go depth N | go movetime T
        let mut depth: u32 = 6;
        let mut movetime_ms: Option<u64> = None;
        let mut tokens = args.split_whitespace();
        while let Some(tok) = tokens.next() {
            match tok {
                "depth" => {
                    if let Some(d) = tokens.next().and_then(|s| s.parse::<u32>().ok()) { depth = d; }
                }
                "movetime" => {
                    if let Some(t) = tokens.next().and_then(|s| s.parse::<u64>().ok()) { movetime_ms = Some(t); }
                }
                _ => {}
            }
        }
        let mut params = SearchParams::default();
        params.depth = depth;
        params.use_tt = true;
        params.order_captures = true;
        params.use_history = true;
        params.movetime = movetime_ms.map(Duration::from_millis);
        params.threads = self.threads;
        let res = self.searcher.search_with_params(self.pos.board(), params);
        if let Some(best) = res.bestmove { println!("bestmove {}", best); } else { println!("bestmove 0000"); }
    }

    pub fn run_loop(&mut self) {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = match line { Ok(s) => s.trim().to_string(), Err(_) => break };
            if line.is_empty() { continue; }
            if line == "uci" { self.cmd_uci(); continue; }
            if line == "isready" { self.cmd_isready(); continue; }
            if line == "ucinewgame" { self.cmd_ucinewgame(); continue; }
            if let Some(rest) = line.strip_prefix("setoption ") { self.cmd_setoption(rest); continue; }
            if line == "quit" { break; }
            if let Some(rest) = line.strip_prefix("position ") { self.cmd_position(rest); continue; }
            if let Some(rest) = line.strip_prefix("go ") { self.cmd_go(rest); continue; }
            if line == "stop" { /* ignore in skeleton */ continue; }
        }
    }
}
