use std::path::Path;

use heterowasm_trace::{Level, Trace};

use crate::cli::{Cli, Command, Options};
use crate::error::CliError;


pub fn run(cli: &Cli) -> Result<(), CliError> {
    match &cli.command {
        Command::Analyze {
            input,
            trace_jsonl,
            dump_ir,
            json,
        } => {
            let options = Options {
                trace: build_trace(trace_jsonl.as_deref())?,
                dump_ir: *dump_ir,
                json: *json,
            };
            crate::analyze::analyze(input, &options)
        }
        Command::Compile {
            input,
            output,
            trace_jsonl,
        } => crate::compile::compile(input, output, build_trace(trace_jsonl.as_deref())?),
        Command::Run {
            dir,
            entry,
            args,
            seed,
            cpu,
            repeat,
            dump_words,
        } => {
            let trace = build_trace(None)?;
            if *cpu {
                crate::baseline::run_cpu(dir, entry, args, seed, *repeat, *dump_words, &trace)
            } else {
                crate::execute::run(dir, entry, args, seed, *repeat, *dump_words, trace)
            }
        }
        Command::Bench {
            dir,
            entry,
            args,
            seed,
            repeat,
        } => crate::bench::run(dir, entry, args, seed, *repeat, build_trace(None)?),
        Command::Eval { json } => crate::eval::run_eval(*json),
    }
}

fn build_trace(path: Option<&Path>) -> Result<Trace, CliError> {
    match path {
        Some(path) => Trace::to_jsonl(path, Level::Debug).map_err(|source| CliError::TraceFile {
            path: path.to_path_buf(),
            source,
        }),
        None => Ok(Trace::to_stderr(Level::Info)),
    }
}
