pub mod analyze;
pub mod baseline;
pub mod bench;
pub mod cli;
pub mod compile;
pub mod consts;
pub mod dispatch;
pub mod error;
pub mod eval;
pub mod execute;
pub mod human;
pub mod json;
pub mod types;

pub use cli::{Cli, Command};
pub use dispatch::run;
pub use error::CliError;
pub use json::render_json;
pub use types::*;


#[cfg(test)]
mod tests;
