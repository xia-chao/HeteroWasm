use std::path::Path;
use std::time::{Duration, Instant};

use heterowasm_trace::Stage;
use wasmtime::{Engine, Extern, Module, Store, Val};

use crate::error::CliError;
use crate::execute::{parse_seed, print_memory, report_timing};


pub(crate) fn run_cpu(
    dir: &Path,
    entry: &str,
    args: &[i32],
    seed: &[String],
    repeat: usize,
    dump_words: usize,
    trace: &heterowasm_trace::Trace,
) -> Result<(), CliError> {
    let _scope = trace.stage(Stage::Runtime, "cli run --cpu");
    let repeat = repeat.max(1);

    let path = dir.join("original.wasm");
    let wasm = std::fs::read(&path).map_err(|source| CliError::Read { path, source })?;

    let engine = Engine::default();
    let module = Module::new(&engine, &wasm)
        .map_err(|error| CliError::Message(format!("failed to parse original.wasm: {error}")))?;
    let mut memory_name: Option<String> = None;
    for export in module.exports() {
        if !matches!(export.ty(), wasmtime::ExternType::Memory(_)) {
            continue;
        }
        let name = export.name().to_string();
        if memory_name.is_none() || name == "mem" {
            memory_name = Some(name);
        }
    }
    let memory_name = memory_name
        .ok_or_else(|| CliError::Message("original.wasm exports no memory".to_string()))?;
    trace
        .info(Stage::Runtime, "memory export")
        .field("name", memory_name.as_str())
        .emit();
    let mut store: Store<()> = Store::new(&engine, ());
    let mut linker = wasmtime::Linker::new(&engine);

    for import in module.imports() {
        let wasmtime::ExternType::Func(ty) = import.ty() else {
            continue;
        };
        let module_name = import.module().to_string();
        let import_name = import.name().to_string();
        trace
            .info(Stage::Runtime, "host import")
            .field("module", module_name.as_str())
            .field("name", import_name.as_str())
            .field("params", ty.params().len())
            .field("results", ty.results().len())
            .emit();
        linker
            .func_new(
                &module_name,
                &import_name,
                ty,
                |_caller, _params, results| {
                    for slot in results.iter_mut() {
                        *slot = Val::I32(0);
                    }
                    Ok(())
                },
            )
            .map_err(|error| CliError::Message(format!("failed to define import: {error}")))?;
    }
    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|error| CliError::Message(format!("instantiation failed: {error}")))?;

    let func = instance.get_func(&mut store, entry).ok_or_else(|| {
        CliError::Message(format!("original.wasm does not export function `{entry}`"))
    })?;
    let mut results = vec![Val::I32(0); func.ty(&store).results().len()];
    let wasm_args: Vec<Val> = args.iter().map(|value| Val::I32(*value)).collect();


    let parsed_seeds: Vec<(usize, Vec<u8>)> = seed
        .iter()
        .map(|spec| {
            parse_seed(spec).map(|(offset, words)| {
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                (offset, bytes)
            })
        })
        .collect::<Result<_, _>>()?;

    let mut elapsed = Duration::ZERO;
    for _ in 0..repeat {
        for (offset, bytes) in &parsed_seeds {
            let memory = match instance.get_export(&mut store, &memory_name) {
                Some(Extern::Memory(memory)) => memory,
                _ => {
                    return Err(CliError::Message(
                        "original.wasm exports no memory".to_string(),
                    ))
                }
            };
            let target = memory.data_mut(&mut store);
            target[*offset..*offset + bytes.len()].copy_from_slice(bytes);
        }
        let started = Instant::now();
        func.call(&mut store, &wasm_args, &mut results)
            .map_err(|error| CliError::Message(error.to_string()))?;
        elapsed += started.elapsed();
    }

    println!("backend       CPU (wasmtime JIT, unrewritten)");
    println!("entry         {entry}");
    if repeat > 1 {
        report_timing(repeat, elapsed);
    }

    let memory = match instance.get_export(&mut store, &memory_name) {
        Some(Extern::Memory(memory)) => memory.data(&store).to_vec(),
        _ => {
            return Err(CliError::Message(
                "original.wasm exports no memory".to_string(),
            ))
        }
    };
    print_memory(&memory, dump_words);

    Ok(())
}
