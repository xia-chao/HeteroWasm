use std::path::PathBuf;

use thiserror::Error;


#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("failed to read artifact {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("artifact missing {name} (run `heterowasm compile` first)")]
    Missing { name: String },

    #[error("manifest {path} is malformed: {detail}")]
    BadManifest { path: PathBuf, detail: String },

    #[error(
        "kernel count mismatches dispatch count: call {call} has no kernel ({available} available)"
    )]
    KernelCountMismatch { call: usize, available: usize },
    #[error("cannot obtain GPU adapter: {reason}")]
    GpuUnavailable { reason: String },

    #[error("spec §15 rejects GPU: {reason}")]
    DispatchRefused { reason: String },
    #[error("wasm execution failed: {reason}")]
    Execute { reason: String },
    #[error("module does not export a memory — host cannot hand linear memory to the GPU")]
    NoMemory,
    #[error("module does not export function `{name}`")]
    NoEntry { name: String },
}
