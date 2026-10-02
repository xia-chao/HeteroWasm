use thiserror::Error;


#[derive(Debug, Error)]
pub enum LoadError {

    #[error("cannot parse input as a WebAssembly module: {reason}")]
    Malformed { reason: String },

    #[error("module uses an unsupported feature: {feature}")]
    Unsupported { feature: String },
}
