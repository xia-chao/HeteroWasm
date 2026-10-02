use heterowasm_trace::{Stage, Trace};

use crate::error::LoadError;


pub fn load_module<'a>(bytes: &'a [u8], trace: &Trace) -> Result<waffle::Module<'a>, LoadError> {
    let _scope = trace.stage(Stage::Frontend, "load_module");
    let options = waffle::FrontendOptions::default();
    match waffle::Module::from_wasm_bytes(bytes, &options) {
        Ok(module) => {
            trace
                .info(Stage::Frontend, "module load succeeded")
                .field("bytes", bytes.len())
                .field("functions", module.funcs.len())
                .emit();
            Ok(module)
        }
        Err(err) => {
            let reason = format!("{err:#}");
            trace
                .error(Stage::Frontend, "module load failed")
                .field("bytes", bytes.len())
                .field("reason", reason.as_str())
                .emit();
            Err(LoadError::Malformed { reason })
        }
    }
}
