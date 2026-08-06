use piebot::io::openings::load_fen_suite;
use std::path::PathBuf;

fn write_temp_suite(name: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "piebot_suite_{}_{}.fen",
        name,
        std::process::id()
    ));
    std::fs::write(&path, contents).expect("write suite file");
    path
}

#[test]
fn missing_suite_file_is_an_error() {
    let err = load_fen_suite(std::path::Path::new("/nonexistent/piebot_suite.fen"))
        .err()
        .expect("missing suite file must fail");
    assert!(err.contains("/nonexistent/piebot_suite.fen"), "{err}");
}

#[test]
fn invalid_suite_line_is_an_error_citing_the_line() {
    let path = write_temp_suite(
        "badline",
        "# comment\nrnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1\nnot a fen\n",
    );
    let err = load_fen_suite(&path)
        .err()
        .expect("invalid suite line must fail");
    std::fs::remove_file(&path).ok();
    assert!(err.contains("line 3"), "{err}");
}

#[test]
fn suite_with_no_positions_is_an_error() {
    let path = write_temp_suite("empty", "# nothing here\n\n");
    let err = load_fen_suite(&path)
        .err()
        .expect("empty suite must fail");
    std::fs::remove_file(&path).ok();
    assert!(err.contains("no valid"), "{err}");
}

#[test]
fn four_field_epd_lines_are_padded_and_loaded() {
    let path = write_temp_suite(
        "epd",
        "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -\n\
         rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1\n",
    );
    let boards = load_fen_suite(&path).expect("valid suite");
    std::fs::remove_file(&path).ok();
    assert_eq!(2, boards.len());
}
