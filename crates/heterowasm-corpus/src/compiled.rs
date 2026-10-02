use std::path::{Path, PathBuf};

use crate::types::{CompiledCase, Language};


pub fn compiled_output_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/source-compiled/out")
}


fn compiled_file_name(prefix: &str, opt_level: &str) -> String {
    format!("{prefix}-o{opt_level}.wasm")
}


pub fn compiled_cases() -> std::io::Result<Vec<CompiledCase>> {
    const LEVELS: [&str; 3] = ["0", "1", "3"];
    const SOURCES: [(Language, &str); 2] = [(Language::Rust, "rust"), (Language::C, "c")];

    let dir = compiled_output_dir();
    if !dir.is_dir() {
        return Ok(Vec::new());
    }

    let mut cases = Vec::new();
    for (language, prefix) in SOURCES {
        for opt_level in LEVELS {
            let path = dir.join(compiled_file_name(prefix, opt_level));
            if path.is_file() {
                cases.push(CompiledCase {
                    language,
                    opt_level,
                    bytes: std::fs::read(&path)?,
                });
            }
        }
    }
    Ok(cases)
}
