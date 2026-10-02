use thiserror::Error;


#[derive(Debug, Error)]
pub enum EvalError {
    #[error("case {name} failed to compile to wasm: {message}")]
    Compile { name: String, message: String },
    #[error("case {name} failed to load: {message}")]
    Load { name: String, message: String },
    #[error("failed to read compiled case: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
}
