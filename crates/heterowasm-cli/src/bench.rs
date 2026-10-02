use std::path::Path;
use std::time::{Duration, Instant};

use heterowasm_runtime::Runtime;
use heterowasm_trace::{Stage, Trace};
use wasmtime::{Engine, Extern, Module, Store, Val};

use crate::error::CliError;
use crate::execute::parse_seed;


struct Measurement {
    adapter: String,
    dispatches: usize,
    cpu: Duration,
    gpu: Duration,

    matched: bool,

    mismatch_at: Option<usize>,
}


pub(crate) fn run(
    dir: &Path,
    entry: &str,
    args: &[i32],
    seed: &[String],
    repeat: usize,
    trace: Trace,
) -> Result<(), CliError> {
    let _scope = trace.stage(Stage::Runtime, "cli bench");
    let repeat = repeat.max(1);

    let seeds = parse_seeds(seed)?;
    let wasm_args: Vec<Val> = args.iter().map(|value| Val::I32(*value)).collect();


    let mut runtime =
        Runtime::load(dir, trace.share()).map_err(|error| CliError::Message(error.to_string()))?;
    let module_bytes =
        std::fs::read(dir.join("original.wasm")).map_err(|source| CliError::Read {
            path: dir.join("original.wasm"),
            source,
        })?;
    let engine = Engine::default();
    let module = Module::new(&engine, &module_bytes)
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
    let mut store: Store<()> = Store::new(&engine, ());
    let mut linker = wasmtime::Linker::new(&engine);

    for import in module.imports() {
        let wasmtime::ExternType::Func(ty) = import.ty() else {
            continue;
        };
        let module_name = import.module().to_string();
        let import_name = import.name().to_string();
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


    seed_memory(&instance, &mut store, &seeds, &memory_name)?;

    let cpu_initial = read_memory(&instance, &mut store, &memory_name)?;
    func.call(&mut store, &wasm_args, &mut results)
        .map_err(|error| CliError::Message(error.to_string()))?;
    let cpu_memory = read_memory(&instance, &mut store, &memory_name)?;

    for (offset, bytes) in &seeds {
        runtime
            .write_memory(*offset, bytes)
            .map_err(|error| CliError::Message(error.to_string()))?;
    }
    let gpu_initial = runtime
        .memory()
        .map_err(|error| CliError::Message(error.to_string()))?;
    runtime
        .run(entry, &wasm_args)
        .map_err(|error| CliError::Message(error.to_string()))?;
    let gpu_memory = runtime
        .memory()
        .map_err(|error| CliError::Message(error.to_string()))?;


    let cpu = time_cpu(
        &func,
        &mut store,
        &wasm_args,
        &mut results,
        repeat,
        &instance,
        &memory_name,
        &cpu_initial,
    )?;
    let gpu = time_gpu(&mut runtime, entry, &wasm_args, repeat, &gpu_initial)?;

    let measurement = Measurement {
        adapter: runtime.adapter().to_string(),
        dispatches: runtime.dispatches(),
        cpu,
        gpu,
        matched: cpu_memory == gpu_memory,
        mismatch_at: first_mismatch(&cpu_memory, &gpu_memory),
    };
    report(&measurement, dir, entry, repeat);
    Ok(())
}

fn time_cpu(
    func: &wasmtime::Func,
    store: &mut Store<()>,
    args: &[Val],
    results: &mut [Val],
    repeat: usize,
    instance: &wasmtime::Instance,
    memory_name: &str,
    initial: &[u8],
) -> Result<Duration, CliError> {
    let mut elapsed = Duration::ZERO;
    for _ in 0..repeat {
        let memory = match instance.get_export(&mut *store, memory_name) {
            Some(Extern::Memory(memory)) => memory,
            _ => {
                return Err(CliError::Message(
                    "original.wasm exports no memory".to_string(),
                ))
            }
        };
        let target = memory.data_mut(&mut *store);
        if target.len() != initial.len() {
            return Err(CliError::Message(
                "memory length changed before a timed repeat".to_string(),
            ));
        }
        target.copy_from_slice(initial);
        let started = Instant::now();
        func.call(&mut *store, args, results)
            .map_err(|error| CliError::Message(error.to_string()))?;
        elapsed += started.elapsed();
    }
    Ok(elapsed)
}

fn time_gpu(
    runtime: &mut Runtime,
    entry: &str,
    args: &[Val],
    repeat: usize,
    initial: &[u8],
) -> Result<Duration, CliError> {
    let mut elapsed = Duration::ZERO;
    for _ in 0..repeat {
        runtime
            .write_memory(0, initial)
            .map_err(|error| CliError::Message(error.to_string()))?;
        let started = Instant::now();
        runtime
            .run(entry, args)
            .map_err(|error| CliError::Message(error.to_string()))?;
        elapsed += started.elapsed();
    }
    Ok(elapsed)
}


fn report(measurement: &Measurement, dir: &Path, entry: &str, repeat: usize) {
    let per_cpu = measurement.cpu.as_secs_f64() * 1000.0 / repeat as f64;
    let per_gpu = measurement.gpu.as_secs_f64() * 1000.0 / repeat as f64;

    println!("artifact       {}", dir.display());
    println!("entry         {entry}");
    println!("backend       {}", measurement.adapter);
    println!("dispatch count {}", measurement.dispatches);
    println!("repeat        {repeat} times (memory restored each time)");
    println!();

    if measurement.matched {
        println!("correctness   CPU and GPU memory **match word-for-word** ✓");
    } else {
        println!("correctness   ✖ **mismatch** — CPU and GPU memory differ");
        if let Some(at) = measurement.mismatch_at {
            println!("              first difference at byte offset {at}");
        }
        println!();
        println!("**No speedup reported.** A wrong result is worthless no matter how fast —");
        println!("that is what this project's \"Soundness before Coverage\" means.");
        return;
    }

    println!();
    println!("CPU           {per_cpu:.4} ms/run");
    println!("GPU           {per_gpu:.4} ms/run");
    if per_gpu > 0.0 {
        let ratio = per_cpu / per_gpu;
        println!();
        if ratio >= 1.0 {
            println!("verdict       GPU **{ratio:.2}× faster**");
        } else {
            println!(
                "verdict       GPU **{:.2}× slower** — arithmetic intensity too low; this path is not worth it",
                1.0 / ratio
            );
        }
    }
}

fn parse_seeds(seed: &[String]) -> Result<Vec<(usize, Vec<u8>)>, CliError> {
    seed.iter()
        .map(|spec| {
            parse_seed(spec).map(|(offset, words)| {
                let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
                (offset, bytes)
            })
        })
        .collect()
}

fn seed_memory(
    instance: &wasmtime::Instance,
    store: &mut Store<()>,
    seeds: &[(usize, Vec<u8>)],
    memory_name: &str,
) -> Result<(), CliError> {
    if seeds.is_empty() {
        return Ok(());
    }
    let memory = match instance.get_export(&mut *store, memory_name) {
        Some(Extern::Memory(memory)) => memory,
        _ => {
            return Err(CliError::Message(
                "original.wasm exports no memory".to_string(),
            ))
        }
    };
    let target = memory.data_mut(&mut *store);
    for (offset, bytes) in seeds {
        let end = offset + bytes.len();
        if end > target.len() {
            return Err(CliError::Message(format!(
                "seed [{offset}, {end}) exceeds memory of {} bytes",
                target.len()
            )));
        }
        target[*offset..end].copy_from_slice(bytes);
    }
    Ok(())
}

fn read_memory(
    instance: &wasmtime::Instance,
    store: &mut Store<()>,
    memory_name: &str,
) -> Result<Vec<u8>, CliError> {
    match instance.get_export(&mut *store, memory_name) {
        Some(Extern::Memory(memory)) => Ok(memory.data(&*store).to_vec()),
        _ => Err(CliError::Message(
            "original.wasm exports no memory".to_string(),
        )),
    }
}

fn first_mismatch(left: &[u8], right: &[u8]) -> Option<usize> {
    left.iter()
        .zip(right.iter())
        .position(|(a, b)| a != b)
        .or_else(|| (left.len() != right.len()).then_some(left.len().min(right.len())))
}
