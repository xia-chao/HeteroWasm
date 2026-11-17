use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use heterowasm_runtime::{Runtime, RuntimeError};
use heterowasm_trace::{Level, Trace};
use wasmtime::{Caller, Engine, Extern, Instance, Linker, Memory, Module, Store, Val};


use clap as _;
use heterowasm_address as _;
use heterowasm_bounds as _;
use heterowasm_cfg as _;
use heterowasm_cli as _;
use heterowasm_dependence as _;
use heterowasm_eval as _;
use heterowasm_frontend as _;
use heterowasm_ir as _;
use heterowasm_legality as _;
use heterowasm_scev as _;
use heterowasm_wgsl as _;
use thiserror as _;
use wasmprinter as _;
use wat as _;


const MEMORY_EXPORT: &str = "h";


static IMPORT_CALLS: [AtomicUsize; 7] = [const { AtomicUsize::new(0) }; 7];
const IMPORT_NAMES: [&str; 7] = ["a::a", "a::b", "a::c", "a::d", "a::e", "a::f", "a::g"];


struct Decoded {
    status_ok: i32,
    status_code: i32,
    points: i32,
    faces: i32,
    attributes: i32,
    attribute_ok: i32,
    float_count: i32,
    position_bits: Vec<u32>,
    indices: Vec<i32>,
    memory: Vec<u8>,
    dispatches: usize,
    elapsed: Duration,
}


trait Host {
    fn call(&mut self, entry: &str, args: &[i32]) -> Result<Vec<Val>, String>;
    fn write_memory(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String>;
    fn memory(&mut self) -> Result<Vec<u8>, String>;
    fn dispatches(&self) -> usize;
}

fn main() -> Result<(), String> {
    let mut argv = std::env::args().skip(1);
    let dir = argv.next().ok_or_else(usage)?;
    let sample_path = argv.next().ok_or_else(usage)?;
    let dir = Path::new(&dir);
    let sample = fs::read(&sample_path)
        .map_err(|error| format!("cannot read sample {sample_path}: {error}"))?;

    reset_imports();
    let mut original_host = OriginalHost::new(&dir.join("original.wasm"))?;
    let original = decode_with(&mut original_host, &sample)?;
    let original_imports = import_report();

    reset_imports();
    let mut rewritten_host = RewrittenHost::new(dir)?;
    let rewritten = decode_with(&mut rewritten_host, &sample)?;
    let rewritten_imports = import_report();

    let report = compare(&original, &rewritten)?;

    println!("sample        {sample_path} ({} bytes)", sample.len());
    println!(
        "original      ok={} code={} points={} faces={} attributes={} floats={} indices={} elapsed_ms={:.3}",
        original.status_ok,
        original.status_code,
        original.points,
        original.faces,
        original.attributes,
        original.position_bits.len(),
        original.indices.len(),
        milliseconds(original.elapsed),
    );
    println!(
        "rewritten     ok={} code={} points={} faces={} attributes={} floats={} indices={} elapsed_ms={:.3}",
        rewritten.status_ok,
        rewritten.status_code,
        rewritten.points,
        rewritten.faces,
        rewritten.attributes,
        rewritten.position_bits.len(),
        rewritten.indices.len(),
        milliseconds(rewritten.elapsed),
    );
    println!("imports       original  {original_imports}");
    println!("              rewritten {rewritten_imports}");
    print!("{report}");
    println!(
        "dispatches    {} (during this decode)",
        rewritten.dispatches
    );
    if rewritten.dispatches > 0 {
        println!("verdict       decode output matches CPU; the rewritten module dispatched to GPU");
    } else {
        println!("verdict       decode output matches CPU; no GPU dispatch during this decode");
    }
    Ok(())
}


fn decode_with<H: Host>(host: &mut H, sample: &[u8]) -> Result<Decoded, String> {
    let started = Instant::now();

    host.call("i", &[])?;

    let length = i32::try_from(sample.len()).map_err(|_| String::from("sample is too large"))?;
    let input = call_i32(host, "Tb", &[length])?;
    if input <= 0 {
        return Err(String::from("malloc returned no buffer for the sample"));
    }
    host.write_memory(input as usize, sample)?;

    let decoder = call_i32(host, "Ma", &[])?;
    let mesh = call_i32(host, "S", &[])?;
    let status = call_i32(host, "Oa", &[decoder, input, length, mesh])?;
    let status_ok = call_i32(host, "_", &[status])?;
    let status_code = call_i32(host, "Z", &[status])?;

    let points = call_i32(host, "V", &[mesh])?;
    let faces = call_i32(host, "T", &[mesh])?;
    let attributes = call_i32(host, "U", &[mesh])?;


    let position_id = call_i32(host, "Pa", &[decoder, mesh, 0])?;
    let attribute = call_i32(host, "Sa", &[decoder, mesh, position_id])?;
    let floats = call_i32(host, "ba", &[])?;
    let attribute_ok = call_i32(host, "$a", &[decoder, mesh, attribute, floats])?;
    let float_count = call_i32(host, "da", &[floats])?;

    let mut position_bits: Vec<u32> = Vec::with_capacity(float_count.max(0) as usize);
    for index in 0..float_count {
        position_bits.push(call_f32_bits(host, "ca", &[floats, index])?);
    }

    let ints = call_i32(host, "va", &[])?;
    let mut indices: Vec<i32> = Vec::with_capacity(faces.max(0) as usize * 3);
    for face in 0..faces {
        let got = call_i32(host, "Wa", &[decoder, mesh, face, ints])?;
        if got == 0 {
            return Err(format!("GetFaceFromMesh failed at face {face}"));
        }
        for corner in 0..3 {
            indices.push(call_i32(host, "wa", &[ints, corner])?);
        }
    }

    let memory = host.memory()?;
    Ok(Decoded {
        status_ok,
        status_code,
        points,
        faces,
        attributes,
        position_bits,
        indices,
        memory,
        attribute_ok,
        float_count,
        dispatches: host.dispatches(),
        elapsed: started.elapsed(),
    })
}


struct OriginalHost {
    store: Store<()>,
    instance: Instance,
}

impl OriginalHost {
    fn new(wasm: &Path) -> Result<Self, String> {
        let bytes =
            fs::read(wasm).map_err(|error| format!("cannot read {}: {error}", wasm.display()))?;
        let engine = Engine::default();
        let module = Module::new(&engine, &bytes)
            .map_err(|error| format!("cannot parse {}: {error}", wasm.display()))?;
        let mut store: Store<()> = Store::new(&engine, ());
        let mut linker: Linker<()> = Linker::new(&engine);
        install_emscripten_imports(&mut linker).map_err(|error| error.to_string())?;
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|error| format!("cannot instantiate {}: {error}", wasm.display()))?;
        Ok(Self { store, instance })
    }

    fn memory_export(&mut self) -> Result<Memory, String> {
        match self.instance.get_export(&mut self.store, MEMORY_EXPORT) {
            Some(Extern::Memory(memory)) => Ok(memory),
            _ => Err(String::from("original.wasm exports no `h` memory")),
        }
    }
}

impl Host for OriginalHost {
    fn call(&mut self, entry: &str, args: &[i32]) -> Result<Vec<Val>, String> {
        let func = self
            .instance
            .get_func(&mut self.store, entry)
            .ok_or_else(|| format!("original.wasm does not export `{entry}`"))?;
        let ty = func.ty(&self.store);
        let params: Vec<Val> = args.iter().map(|value| Val::I32(*value)).collect();
        let mut results: Vec<Val> = (0..ty.results().len()).map(|_| Val::I32(0)).collect();
        func.call(&mut self.store, &params, &mut results)
            .map_err(|error| format!("original `{entry}` failed: {error}"))?;
        Ok(results)
    }

    fn write_memory(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        let memory = self.memory_export()?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| String::from("sample offset overflow"))?;
        memory
            .data_mut(&mut self.store)
            .get_mut(offset..end)
            .ok_or_else(|| String::from("sample does not fit linear memory"))?
            .copy_from_slice(bytes);
        Ok(())
    }

    fn memory(&mut self) -> Result<Vec<u8>, String> {
        let memory = self.memory_export()?;
        Ok(memory.data(&self.store).to_vec())
    }

    fn dispatches(&self) -> usize {
        0
    }
}


struct RewrittenHost {
    runtime: Runtime,
}

impl RewrittenHost {
    fn new(dir: &Path) -> Result<Self, String> {
        let runtime = Runtime::load_with(
            dir,
            Trace::to_stderr(Level::Error),
            |_engine, linker, _module| {
                install_emscripten_imports(linker).map_err(|error| RuntimeError::Import {
                    reason: error.to_string(),
                })
            },
        )
        .map_err(|error| format!("cannot load artifact {}: {error}", dir.display()))?;
        Ok(Self { runtime })
    }
}

impl Host for RewrittenHost {
    fn call(&mut self, entry: &str, args: &[i32]) -> Result<Vec<Val>, String> {
        let params: Vec<Val> = args.iter().map(|value| Val::I32(*value)).collect();
        self.runtime
            .run(entry, &params)
            .map_err(|error| format!("rewritten `{entry}` failed: {error}"))
    }

    fn write_memory(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        self.runtime
            .write_memory(offset, bytes)
            .map_err(|error| format!("rewritten write_memory failed: {error}"))
    }

    fn memory(&mut self) -> Result<Vec<u8>, String> {
        self.runtime
            .memory()
            .map_err(|error| format!("rewritten memory snapshot failed: {error}"))
    }

    fn dispatches(&self) -> usize {
        self.runtime.dispatches()
    }
}


fn compare(original: &Decoded, rewritten: &Decoded) -> Result<String, String> {
    if original.status_ok != 1 {
        return Err(format!(
            "original decode failed: ok={} code={}",
            original.status_ok, original.status_code
        ));
    }
    if original.points <= 0
        || original.attribute_ok != 1
        || original.float_count != original.points.saturating_mul(3)
    {
        return Err(format!(
            "decode did not produce a readable POSITION attribute: points={}, attribute_ok={}, floats={} (want points * 3)",
            original.points, original.attribute_ok, original.float_count
        ));
    }
    if rewritten.status_ok != original.status_ok {
        return Err(format!(
            "status differs: original ok={} rewritten ok={} (code={})",
            original.status_ok, rewritten.status_ok, rewritten.status_code
        ));
    }
    if rewritten.points != original.points
        || rewritten.faces != original.faces
        || rewritten.attributes != original.attributes
    {
        return Err(format!(
            "geometry counts differ: points {} vs {}, faces {} vs {}, attributes {} vs {}",
            original.points,
            rewritten.points,
            original.faces,
            rewritten.faces,
            original.attributes,
            rewritten.attributes
        ));
    }
    if original.attribute_ok != rewritten.attribute_ok
        || original.float_count != rewritten.float_count
    {
        return Err(format!(
            "attribute readback differs: ok {} vs {}, floats {} vs {}",
            original.attribute_ok,
            rewritten.attribute_ok,
            original.float_count,
            rewritten.float_count
        ));
    }
    if original.position_bits != rewritten.position_bits {
        return Err(format!(
            "position values differ (first value {} vs {})",
            original.position_bits.first().copied().unwrap_or_default(),
            rewritten.position_bits.first().copied().unwrap_or_default()
        ));
    }
    if original.indices != rewritten.indices {
        return Err(String::from("triangle indices differ"));
    }
    if original.memory.len() != rewritten.memory.len() {
        return Err(format!(
            "linear memory size differs: {} vs {} bytes",
            original.memory.len(),
            rewritten.memory.len()
        ));
    }
    if let Some(at) = first_mismatch(&original.memory, &rewritten.memory) {
        return Err(format!("linear memory differs at byte {at}"));
    }

    let mut report = String::new();
    report.push_str(&format!(
        "memory        match word-for-word ({} bytes)\n",
        original.memory.len()
    ));
    report.push_str(&format!(
        "positions     bit-identical ({} f32 values)\n",
        original.position_bits.len()
    ));
    report.push_str(&format!(
        "indices       identical ({} i32 values)\n",
        original.indices.len()
    ));
    Ok(report)
}


fn first_mismatch(left: &[u8], right: &[u8]) -> Option<usize> {
    left.iter().zip(right.iter()).position(|(a, b)| a != b)
}


fn install_emscripten_imports<T: 'static>(linker: &mut Linker<T>) -> Result<(), wasmtime::Error> {
    linker.func_wrap("a", "a", |_caller: Caller<'_, T>| {
        count_import(0);
    })?;
    linker.func_wrap(
        "a",
        "b",
        |_caller: Caller<'_, T>, _p0: i32, _p1: i32, _p2: i32| {
            count_import(1);
        },
    )?;
    linker.func_wrap(
        "a",
        "c",
        |mut caller: Caller<'_, T>, _fd: i32, iovs: i32, iovs_len: i32, written: i32| {
            count_import(2);
            let memory = linear_memory(&mut caller)?;
            let mut out: Vec<u8> = Vec::new();
            for index in 0..iovs_len.max(0) {
                let base = iovs as usize + index as usize * 8;
                let data = memory.data(&caller);
                let ptr = read_u32(data, base)? as usize;
                let len = read_u32(data, base + 4)? as usize;
                let bytes = data
                    .get(ptr..ptr + len)
                    .ok_or_else(|| wasmtime::Error::msg("fd_write iov out of memory"))?;
                out.extend_from_slice(bytes);
            }
            eprint!("{}", String::from_utf8_lossy(&out));
            let data = memory.data_mut(&mut caller);
            write_u32(data, written as usize, out.len() as u32)?;
            Ok::<i32, wasmtime::Error>(0)
        },
    )?;
    linker.func_wrap(
        "a",
        "d",
        |_caller: Caller<'_, T>, _p0: i32, _p1: i32, _p2: i32, _p3: i32, _p4: i32| {
            count_import(3);
            70i32
        },
    )?;
    linker.func_wrap("a", "e", |mut caller: Caller<'_, T>, requested: i32| {
        count_import(4);
        let memory = linear_memory(&mut caller)?;
        let current_bytes = memory.data_size(&caller) as u64;
        let wanted_bytes = (requested.max(0) as u64).div_ceil(65536) * 65536;
        let current_pages = current_bytes / 65536;
        let wanted_pages = (wanted_bytes / 65536).max(current_pages);
        let grown = if wanted_pages > current_pages {
            memory
                .grow(&mut caller, wanted_pages - current_pages)
                .is_ok()
        } else {
            true
        };
        Ok::<i32, wasmtime::Error>(i32::from(grown))
    })?;
    linker.func_wrap("a", "f", |_caller: Caller<'_, T>, _p0: i32| {
        count_import(5);
        52i32
    })?;
    linker.func_wrap(
        "a",
        "g",
        |mut caller: Caller<'_, T>, dest: i32, src: i32, len: i32| {
            count_import(6);
            let memory = linear_memory(&mut caller)?;
            let dest = dest.max(0) as usize;
            let src = src.max(0) as usize;
            let len = len.max(0) as usize;
            let data = memory.data(&caller);
            let copy: Vec<u8> = data
                .get(src..src + len)
                .ok_or_else(|| wasmtime::Error::msg("memcpy source out of memory"))?
                .to_vec();
            let data = memory.data_mut(&mut caller);
            data.get_mut(dest..dest + len)
                .ok_or_else(|| wasmtime::Error::msg("memcpy dest out of memory"))?
                .copy_from_slice(&copy);
            Ok::<(), wasmtime::Error>(())
        },
    )?;
    Ok(())
}


fn linear_memory<T>(caller: &mut Caller<'_, T>) -> Result<Memory, wasmtime::Error> {
    match caller.get_export(MEMORY_EXPORT) {
        Some(Extern::Memory(memory)) => Ok(memory),
        _ => Err(wasmtime::Error::msg("module exports no `h` memory")),
    }
}

fn read_u32(data: &[u8], at: usize) -> Result<u32, wasmtime::Error> {
    let bytes = data
        .get(at..at + 4)
        .ok_or_else(|| wasmtime::Error::msg("read out of linear memory"))?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_u32(data: &mut [u8], at: usize, value: u32) -> Result<(), wasmtime::Error> {
    let slot = data
        .get_mut(at..at + 4)
        .ok_or_else(|| wasmtime::Error::msg("write out of linear memory"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn call_i32<H: Host>(host: &mut H, entry: &str, args: &[i32]) -> Result<i32, String> {
    let results = host.call(entry, args)?;
    match results.first() {
        Some(Val::I32(value)) => Ok(*value),
        other => Err(format!("`{entry}` should return one i32, got {other:?}")),
    }
}

fn call_f32_bits<H: Host>(host: &mut H, entry: &str, args: &[i32]) -> Result<u32, String> {
    let results = host.call(entry, args)?;
    match results.first() {
        Some(Val::F32(bits)) => Ok(*bits),
        other => Err(format!("`{entry}` should return one f32, got {other:?}")),
    }
}

fn count_import(slot: usize) {
    if let Some(counter) = IMPORT_CALLS.get(slot) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

fn reset_imports() {
    for counter in &IMPORT_CALLS {
        counter.store(0, Ordering::Relaxed);
    }
}

fn import_report() -> String {
    IMPORT_NAMES
        .iter()
        .zip(IMPORT_CALLS.iter())
        .map(|(name, counter)| format!("{name}={}", counter.load(Ordering::Relaxed)))
        .collect::<Vec<String>>()
        .join(" ")
}

fn milliseconds(elapsed: Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

fn usage() -> String {
    String::from("usage: draco_decode <artifact-dir> <bunny.drc>")
}
