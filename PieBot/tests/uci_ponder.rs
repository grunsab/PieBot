//! UCI pondering: `go ponder`, `ponderhit`, `stop` and `bestmove ... ponder`.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

struct Engine {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
}

impl Engine {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_uci"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("start UCI engine");
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = Self { child, stdin, rx };
        engine.send("uci");
        engine.wait_for("uciok", Duration::from_secs(5)).expect("uciok");
        engine
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// The first line starting with `prefix` within `timeout`, if any.
    fn wait_for(&self, prefix: &str, timeout: Duration) -> Option<String> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            match self.rx.recv_timeout(Duration::from_millis(20)) {
                Ok(line) if line.starts_with(prefix) => return Some(line),
                Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
        None
    }

    fn lines_for(&self, d: Duration) -> Vec<String> {
        let start = Instant::now();
        let mut out = Vec::new();
        while start.elapsed() < d {
            if let Ok(line) = self.rx.recv_timeout(Duration::from_millis(20)) {
                out.push(line);
            }
        }
        out
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(2) {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = self.child.kill();
    }
}

#[test]
fn uci_advertises_the_ponder_option() {
    let mut e = Engine::start();
    e.send("uci");
    let lines = e.lines_for(Duration::from_millis(300));
    assert!(
        lines.iter().any(|l| l.starts_with("option name Ponder type check")),
        "no Ponder option in {lines:?}"
    );
}

#[test]
fn bestmove_names_a_ponder_move() {
    let mut e = Engine::start();
    e.send("position startpos moves e2e4");
    e.send("go depth 8");
    let best = e.wait_for("bestmove", Duration::from_secs(10)).expect("bestmove");
    let parts: Vec<&str> = best.split_whitespace().collect();
    assert_eq!(parts.len(), 4, "{best}");
    assert_eq!(parts[2], "ponder", "{best}");
    assert!((4..=5).contains(&parts[3].len()), "{best}");
}

#[test]
fn go_ponder_waits_for_ponderhit_then_moves_on_its_clock() {
    let mut e = Engine::start();
    e.send("position startpos moves e2e4 e7e5");
    e.send("go ponder wtime 3000 btime 3000");
    let early = e.lines_for(Duration::from_millis(700));
    assert!(
        !early.iter().any(|l| l.starts_with("bestmove")),
        "bestmove while pondering: {early:?}"
    );
    let hit = Instant::now();
    e.send("ponderhit");
    let best = e.wait_for("bestmove", Duration::from_secs(3));
    assert!(best.is_some(), "no bestmove after ponderhit");
    assert!(hit.elapsed() < Duration::from_millis(2500), "ponderhit took {:?}", hit.elapsed());
}

#[test]
fn a_finished_ponder_search_holds_its_move_until_stop() {
    let mut e = Engine::start();
    e.send("position startpos");
    e.send("go ponder depth 2");
    let early = e.lines_for(Duration::from_millis(400));
    assert!(
        !early.iter().any(|l| l.starts_with("bestmove")),
        "bestmove before stop: {early:?}"
    );
    e.send("stop");
    assert!(e.wait_for("bestmove", Duration::from_secs(1)).is_some(), "no bestmove after stop");
    // The engine is reusable afterwards.
    e.send("position startpos moves d2d4");
    e.send("go depth 3");
    assert!(e.wait_for("bestmove", Duration::from_secs(5)).is_some());
}

#[test]
fn a_finished_ponder_search_moves_at_once_on_ponderhit() {
    let mut e = Engine::start();
    e.send("position startpos");
    e.send("go ponder depth 2 wtime 60000 btime 60000");
    std::thread::sleep(Duration::from_millis(300));
    let hit = Instant::now();
    e.send("ponderhit");
    assert!(e.wait_for("bestmove", Duration::from_secs(1)).is_some(), "no bestmove after ponderhit");
    assert!(hit.elapsed() < Duration::from_millis(500));
}
