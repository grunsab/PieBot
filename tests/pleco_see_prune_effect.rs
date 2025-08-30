#![cfg(feature = "board-pleco")]
use pleco::Board as PBoard;

#[test]
fn see_pruning_reduces_qsearch_nodes_on_capture_heavy_position() {
    use piebot::search::alphabeta_pleco::PlecoSearcher;
    // Artificially capture-heavy middlegame-ish position
    let fen = "4k3/8/2pppp2/2rprp2/2PPRP2/2PPPP2/8/4K3 w - - 0 1";
    let mut b = PBoard::from_fen(fen).expect("valid fen");

    let (_bm1, _sc1, n1) = {
        let mut s = PlecoSearcher::default();
        s.set_threads(1);
        s.set_see_prune(false);
        s.search_movetime(&mut b.clone(), 50, 3)
    };

    let (_bm2, _sc2, n2) = {
        let mut s = PlecoSearcher::default();
        s.set_threads(1);
        s.set_see_prune(true);
        s.search_movetime(&mut b.clone(), 50, 3)
    };

    assert!(n2 <= n1, "SEE pruning should not increase nodes: n1={} n2={}", n1, n2);
}
