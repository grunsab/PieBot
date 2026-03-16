use crate::eval::nnue::loader::QuantNnue;
use anyhow::{Context, Result};

pub struct TestNnueConfig {
    pub quant_model: Option<QuantNnue>,
    pub blend_percent: u8,
}

pub fn load_test_nnue_config_from_env() -> Result<TestNnueConfig> {
    let blend_percent = std::env::var("PIEBOT_TEST_NNUE_BLEND_PERCENT")
        .ok()
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(100)
        .min(100);

    let quant_model = if let Some(path) = std::env::var("PIEBOT_TEST_NNUE_QUANT_FILE")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        Some(
            QuantNnue::load_quantized(&path)
                .with_context(|| format!("load PIEBOT_TEST_NNUE_QUANT_FILE={path}"))?,
        )
    } else {
        None
    };

    Ok(TestNnueConfig {
        quant_model,
        blend_percent,
    })
}

#[cfg(test)]
mod tests {
    use super::load_test_nnue_config_from_env;
    use crate::eval::nnue::features::halfkp_dim;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn tmp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}.nnue", name, std::process::id(), nanos))
    }

    fn write_quant_file(path: &PathBuf) {
        let input_dim = halfkp_dim() as u32;
        let hidden_dim = 4u32;
        let mut f = File::create(path).unwrap();
        f.write_all(b"PIENNQ01").unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&input_dim.to_le_bytes()).unwrap();
        f.write_all(&hidden_dim.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        f.write_all(&1.0f32.to_le_bytes()).unwrap();
        for _ in 0..(input_dim as usize * hidden_dim as usize) {
            f.write_all(&[0u8]).unwrap();
        }
        for _ in 0..hidden_dim {
            f.write_all(&0i16.to_le_bytes()).unwrap();
        }
        for _ in 0..hidden_dim {
            f.write_all(&[0u8]).unwrap();
        }
        f.write_all(&0i16.to_le_bytes()).unwrap();
    }

    #[test]
    fn loads_quant_nnue_test_config_from_env() {
        let _guard = env_lock().lock().unwrap();
        let path = tmp_path("test_nnue_env");
        write_quant_file(&path);

        let prev_path = std::env::var_os("PIEBOT_TEST_NNUE_QUANT_FILE");
        let prev_blend = std::env::var_os("PIEBOT_TEST_NNUE_BLEND_PERCENT");
        std::env::set_var("PIEBOT_TEST_NNUE_QUANT_FILE", &path);
        std::env::set_var("PIEBOT_TEST_NNUE_BLEND_PERCENT", "87");

        let cfg = load_test_nnue_config_from_env().unwrap();
        assert!(cfg.quant_model.is_some());
        assert_eq!(87, cfg.blend_percent);

        if let Some(value) = prev_path {
            std::env::set_var("PIEBOT_TEST_NNUE_QUANT_FILE", value);
        } else {
            std::env::remove_var("PIEBOT_TEST_NNUE_QUANT_FILE");
        }
        if let Some(value) = prev_blend {
            std::env::set_var("PIEBOT_TEST_NNUE_BLEND_PERCENT", value);
        } else {
            std::env::remove_var("PIEBOT_TEST_NNUE_BLEND_PERCENT");
        }

        let _ = std::fs::remove_file(path);
    }
}
