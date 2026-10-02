use std::path::Path;
use std::time::{Duration, Instant};

use heterowasm_runtime::Runtime;
use heterowasm_trace::{Stage, Trace};
use wasmtime::Val;

use crate::error::CliError;


pub(crate) fn run(
    dir: &Path,
    entry: &str,
    args: &[i32],
    seed: &[String],
    repeat: usize,
    dump_words: usize,
    trace: Trace,
) -> Result<(), CliError> {
    let _scope = trace.stage(Stage::Runtime, "cli run");
    let repeat = repeat.max(1);


    let mut runtime =
        Runtime::load(dir, trace.share()).map_err(|error| CliError::Message(error.to_string()))?;

    let seeds: Vec<(usize, Vec<u8>)> = seed
        .iter()
        .map(|spec| {
            parse_seed(spec).map(|(offset, words)| {
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                (offset, bytes)
            })
        })
        .collect::<Result<_, _>>()?;

    let wasm_args: Vec<Val> = args.iter().map(|value| Val::I32(*value)).collect();

    let mut elapsed = Duration::ZERO;
    for _ in 0..repeat {
        for (offset, bytes) in &seeds {
            runtime
                .write_memory(*offset, bytes)
                .map_err(|error| CliError::Message(error.to_string()))?;
        }
        let started = Instant::now();
        runtime
            .run(entry, &wasm_args)
            .map_err(|error| CliError::Message(error.to_string()))?;
        elapsed += started.elapsed();
    }

    println!("backend       {}", runtime.adapter());
    println!("dispatch count {}", runtime.dispatches());
    println!("entry         {entry}");
    if repeat > 1 {
        report_timing(repeat, elapsed);
    }


    let memory = runtime
        .memory()
        .map_err(|error| CliError::Message(error.to_string()))?;
    print_memory(&memory, dump_words);

    Ok(())
}


pub(crate) fn report_timing(repeat: usize, elapsed: Duration) {
    let per = elapsed.as_secs_f64() * 1000.0 / repeat as f64;
    println!(
        "repeat        {repeat} times, total {:.3} ms, {per:.4} ms each",
        elapsed.as_secs_f64() * 1000.0
    );
}


pub(crate) fn print_memory(memory: &[u8], dump_words: usize) {
    let words = dump_words.min(memory.len() / 4);
    print!("first {words} words of memory ");
    for index in 0..words {
        let at = index * 4;
        let value =
            u32::from_le_bytes([memory[at], memory[at + 1], memory[at + 2], memory[at + 3]]);
        if index > 0 {
            print!(" ");
        }
        print!("{value}");
    }
    println!();
}


pub(crate) fn parse_seed(spec: &str) -> Result<(usize, Vec<u32>), CliError> {
    let (offset, list) = spec.split_once(':').ok_or_else(|| {
        CliError::Message(format!(
            "--seed format should be `<offset>:<u32>,<u32>,…`, got `{spec}`"
        ))
    })?;
    let offset: usize = offset
        .trim()
        .parse()
        .map_err(|_| CliError::Message(format!("--seed offset is not an integer: `{offset}`")))?;
    let mut words = Vec::new();
    for piece in list.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let word: u32 = piece
            .parse()
            .map_err(|_| CliError::Message(format!("--seed value is not a u32: `{piece}`")))?;
        words.push(word);
    }
    if words.is_empty() {
        return Err(CliError::Message(format!("--seed `{spec}` has no values")));
    }
    Ok((offset, words))
}
