//! Tablebase integration for the experimental search. Needs real Syzygy
//! files: set PIEBOT_SYZYGY_PATH to a 3-4-5 directory, otherwise the test
//! returns early.

use cozy_chess::Board;
use piebot::search::alphabeta_temp::Searcher;
use piebot::search::syzygy;

#[test]
fn tablebases_drive_root_moves_and_in_tree_scores() {
    let Ok(path) = std::env::var("PIEBOT_SYZYGY_PATH") else {
        eprintln!("PIEBOT_SYZYGY_PATH unset; skipping");
        return;
    };
    let n = syzygy::set_path(&path).expect("tables load");
    assert!(n > 0 && syzygy::max_pieces() >= 5);

    // KRvK: winning for White.
    let won = Board::from_fen("8/8/8/4k3/8/8/8/R3K3 w - - 0 1", false).unwrap();
    assert_eq!(syzygy::probe_wdl(&won), Some(2));
    // KNvK: drawn.
    let drawn = Board::from_fen("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", false).unwrap();
    assert_eq!(syzygy::probe_wdl(&drawn), Some(0));

    // Rxd5 trades into a tablebase win; the search must see it as such.
    // Six pieces, so the root itself is not probed.
    let trade = Board::from_fen("8/8/4k3/3r4/8/8/3R2P1/3RK3 w - - 0 1", false).unwrap();
    let mut s = Searcher::default();
    let r = s.search_depth(&trade, 6);
    assert_eq!(r.bestmove.as_deref(), Some("d2d5"));
    assert!(r.score_cp > 15_000, "trade into KRvK should score as a TB win: {}", r.score_cp);
    syzygy::set_path("").unwrap();
}
