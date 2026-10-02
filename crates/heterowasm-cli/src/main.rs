#![expect(
    unused_crate_dependencies,
    reason = "this package is lib + a thin bin; deps are used by the lib, bin only calls heterowasm_cli::run"
)]

use std::process::ExitCode;

use clap::Parser;
use heterowasm_cli::{run, Cli};

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
