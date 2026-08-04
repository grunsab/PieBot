use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[test]
fn stop_interrupts_an_active_search_and_returns_bestmove() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_uci"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("start UCI engine");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    writeln!(stdin, "uci").unwrap();
    writeln!(stdin, "isready").unwrap();
    writeln!(stdin, "position startpos").unwrap();
    writeln!(stdin, "go infinite").unwrap();
    stdin.flush().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let stop_sent = Instant::now();
    writeln!(stdin, "stop").unwrap();
    stdin.flush().unwrap();

    let mut bestmove = None;
    while stop_sent.elapsed() < Duration::from_secs(2) {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(line) if line.starts_with("bestmove ") => {
                bestmove = Some(line);
                break;
            }
            Ok(_) => {}
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let mut second_bestmove = None;
    if bestmove.is_none() {
        let _ = child.kill();
    } else {
        writeln!(stdin, "position startpos moves e2e4").ok();
        writeln!(stdin, "go nodes 1000").ok();
        stdin.flush().ok();
        let second_started = Instant::now();
        while second_started.elapsed() < Duration::from_secs(2) {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(line) if line.starts_with("bestmove ") => {
                    second_bestmove = Some(line);
                    break;
                }
                Ok(_) => {}
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        writeln!(stdin, "quit").ok();
        stdin.flush().ok();
    }
    let _ = child.wait();
    drop(stdin);
    let _ = reader.join();

    assert!(
        bestmove.is_some(),
        "UCI stop did not interrupt the active search within two seconds"
    );
    assert!(
        second_bestmove.is_some(),
        "UCI engine was not reusable after an interrupted search"
    );
}
