use piebot::uci::UciEngine;

fn main() {
    let mut engine = UciEngine::new();
    engine.run_loop();
}
