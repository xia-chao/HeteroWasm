use std::path::PathBuf;

use thiserror::Error;


#[derive(Debug, Error)]
pub enum CliError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot create {path}: {source}")]
    CreateDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot create log file {path}: {source}")]
    TraceFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to write WGSL: {source}")]
    WriteShader { source: std::io::Error },
    #[error("failed to write manifest: {source}")]
    WriteManifest { source: std::io::Error },
    #[error("failed to write rewrite result: {source}")]
    WriteRewritten { source: std::io::Error },
    #[error("failed to inject ABI import: {reason}")]
    Inject { reason: String },

    #[error("{0}")]
    Message(String),
}
