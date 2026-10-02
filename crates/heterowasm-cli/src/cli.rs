use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "heterowasm",
    version,
    about = "Soundness-first Wasm → WebGPU heterogeneous compiler"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Analyze {

        input: PathBuf,

        #[arg(long)]
        trace_jsonl: Option<PathBuf>,

        #[arg(long)]
        dump_ir: bool,

        #[arg(long, conflicts_with = "dump_ir")]
        json: bool,
    },

    Compile {

        input: PathBuf,

        #[arg(long, default_value = "heterowasm-out")]
        output: PathBuf,

        #[arg(long)]
        trace_jsonl: Option<PathBuf>,
    },

    Run {

        dir: PathBuf,

        #[arg(long, default_value = "main")]
        entry: String,

        #[arg(long = "arg", allow_hyphen_values = true)]
        args: Vec<i32>,

        #[arg(long = "seed")]
        seed: Vec<String>,

        #[arg(long)]
        cpu: bool,

        #[arg(long, default_value_t = 1)]
        repeat: usize,


        #[arg(long, default_value_t = 16)]
        dump_words: usize,
    },

    Bench {

        dir: PathBuf,

        #[arg(long, default_value = "main")]
        entry: String,

        #[arg(long = "arg", allow_hyphen_values = true)]
        args: Vec<i32>,

        #[arg(long = "seed")]
        seed: Vec<String>,

        #[arg(long, default_value_t = 20)]
        repeat: usize,
    },

    Eval {

        #[arg(long)]
        json: bool,
    },
}


pub struct Options {
    pub(crate) trace: heterowasm_trace::Trace,
    pub(crate) dump_ir: bool,
    pub(crate) json: bool,
}
