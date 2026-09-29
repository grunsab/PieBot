use piebot::uci::UciEngine;

fn main() {
    let mut engine = UciEngine::new();
    if engine.auto_load_default_model() {
        if let Some(path) = &engine.default_model_path {
            eprintln!("info string Loaded default NNUE model: {path}");
        }
    }
    engine.run_loop();
}
