use heterowasm_trace::{Stage, Trace};
use wasmtime::{Caller, Engine, Extern, Linker, Memory, Val, ValType};

use crate::error::RuntimeError;
use crate::gpu::{Compiled, GpuContext, Workspace};
use crate::types::{KernelSpec, Shader};


pub struct HostState {
    pub(crate) kernels: Vec<KernelSpec>,
    pub(crate) compiled: Vec<Compiled>,
    pub(crate) context: GpuContext,
    pub(crate) workspaces: Vec<Option<Workspace>>,

    pub(crate) calls: usize,

    pub(crate) total: usize,
    pub(crate) trace: Trace,
    pub(crate) memory_export: String,
}


pub(crate) fn import_arity(kernels: &[KernelSpec]) -> usize {
    kernels.iter().map(uniform_slots).max().unwrap_or(0) + 1
}


pub(crate) fn import_results(kernels: &[KernelSpec]) -> usize {
    kernels
        .iter()
        .map(|kernel| kernel.result_slots)
        .max()
        .unwrap_or(0)
}


fn uniform_slots(kernel: &KernelSpec) -> usize {
    let from_fields = kernel.fields.iter().map(|field| field.slot + 1).max();

    if kernel.fields_resolved {
        return from_fields.unwrap_or(0);
    }
    from_fields.unwrap_or(kernel.uniform_size / 4)
}


pub(crate) fn link_dynamic(
    engine: &Engine,
    linker: &mut Linker<HostState>,
    arity: usize,
    results: usize,
) -> Result<(), RuntimeError> {

    let ty = wasmtime::FuncType::new(
        engine,
        (0..arity).map(|_| ValType::I32),
        (0..results).map(|_| ValType::I32),
    );
    linker
        .func_new(
            "heterowasm",
            "__hw_dispatch",
            ty,
            move |mut caller: Caller<'_, HostState>, params: &[Val], results: &mut [Val]| {
                dispatch(&mut caller, params, results)
            },
        )
        .map_err(|error| RuntimeError::Execute {
            reason: error.to_string(),
        })?;
    Ok(())
}


fn dispatch(
    caller: &mut Caller<'_, HostState>,
    params: &[Val],
    results: &mut [Val],
) -> Result<(), wasmtime::Error> {
    let args: Vec<u32> = params
        .iter()
        .map(|value| match value {
            Val::I32(raw) => Ok(*raw as u32),
            other => Err(wasmtime::Error::msg(format!(
                "argument is not i32, got {other:?}"
            ))),
        })
        .collect::<Result<_, _>>()?;


    let (index, uniforms) = match args.split_last() {
        Some((selector, uniforms)) => {
            let index = usize::try_from(*selector).unwrap_or(usize::MAX);
            (index, uniforms)
        }
        None => (0, &[][..]),
    };
    {
        let state = caller.data_mut();
        state.calls += 1;
        state.total += 1;
    }


    let (
        source,
        workgroup_size,
        dispatch_spec,
        uniform_size,
        min_offset_bytes,
        max_constant_bytes,
        max_stride_bytes,
        index_field,
        launch_count,
    ) = {
        let state = caller.data();
        let kernel = state
            .kernels
            .get(index)
            .ok_or(RuntimeError::KernelCountMismatch {
                call: index,
                available: state.kernels.len(),
            })?;
        (
            kernel.source.clone(),
            kernel.workgroup_size,
            kernel.dispatch,
            kernel.uniform_size,
            kernel.min_offset_bytes,
            kernel.max_constant_bytes,
            kernel.max_stride_bytes,
            kernel.index_field,
            kernel.launch_count,
        )
    };
    let shader = Shader {
        source: &source,
        workgroup_size,
        dispatch: dispatch_spec,
        min_offset_bytes,
        max_constant_bytes,
        max_stride_bytes,
        index_field,
        launch_count,
    };


    let memory_export = caller.data().memory_export.clone();
    let memory = match caller.get_export(&memory_export) {
        Some(Extern::Memory(memory)) => memory,
        _ => return Err(wasmtime::Error::from(RuntimeError::NoMemory)),
    };

    let memory_len = memory.data(&mut *caller).len();


    crate::dispatch::check(&shader, uniforms, memory_len as u64).map_err(wasmtime::Error::from)?;

    let ranges = GpuContext::copy_ranges(&shader, uniforms, memory_len);
    let parts = read_memory(&memory, caller, &ranges)?;

    let trace = caller.data().trace.share();
    let ran = {
        let state = caller.data_mut();
        run_on_gpu(
            state,
            index,
            &shader,
            memory_len,
            &parts,
            uniforms,
            uniform_size,
            &trace,
        )
    }?;

    let (regions, exit_values) = ran;
    write_memory(&memory, caller, &regions).map_err(wasmtime::Error::from)?;


    for slot in results.iter_mut() {
        *slot = Val::I32(0);
    }
    for (slot, value) in results.iter_mut().zip(exit_values.iter()) {
        *slot = Val::I32(*value as i32);
    }

    trace
        .debug(Stage::Runtime, "__hw_dispatch")
        .field("call", index)
        .field("params", uniforms.len())
        .field("words_out", memory_len / 4)
        .field("results_out", exit_values.len() as i64)
        .emit();
    Ok(())
}


fn run_on_gpu(
    state: &mut HostState,
    index: usize,
    shader: &Shader<'_>,
    memory_len: usize,
    parts: &[(usize, Vec<u8>)],
    params: &[u32],
    uniform_size: usize,
    trace: &Trace,
) -> Result<(Vec<(usize, Vec<u8>)>, Vec<u32>), RuntimeError> {
    let size = memory_len.max(4);

    let result_slots = state
        .kernels
        .get(index)
        .map(|kernel| kernel.result_slots)
        .unwrap_or(0);
    let needs_new = match state.workspaces.get(index) {
        Some(Some(workspace)) => workspace.size() != size,
        _ => true,
    };
    if needs_new {
        let workspace = state.context.workspace(size, uniform_size, result_slots);
        if state.workspaces.len() <= index {
            state.workspaces.resize_with(index + 1, || None);
        }
        state.workspaces[index] = Some(workspace);
    }

    let compiled = state
        .compiled
        .get(index)
        .ok_or(RuntimeError::KernelCountMismatch {
            call: index,
            available: state.compiled.len(),
        })?;
    let workspace = state.workspaces[index]
        .as_ref()
        .ok_or(RuntimeError::KernelCountMismatch {
            call: index,
            available: state.workspaces.len(),
        })?;

    let context = &state.context;
    let limit = memory_len.min(workspace.size());
    context.run_parts(compiled, shader, workspace, limit, parts, params, trace)
}


fn read_memory(
    memory: &Memory,
    caller: &Caller<'_, HostState>,
    ranges: &[(usize, usize, usize, usize)],
) -> Result<Vec<(usize, Vec<u8>)>, RuntimeError> {
    let data = memory.data(caller);
    let mut parts = Vec::with_capacity(ranges.len());
    for (_, _, copy_lo, copy_hi) in ranges {
        let bytes = data
            .get(*copy_lo..*copy_hi)
            .ok_or_else(|| RuntimeError::DispatchRefused {
                reason: "copy range is outside linear memory".to_string(),
            })?;
        parts.push((*copy_lo, bytes.to_vec()));
    }
    Ok(parts)
}


fn write_memory(
    memory: &Memory,
    caller: &mut Caller<'_, HostState>,
    regions: &[(usize, Vec<u8>)],
) -> Result<(), RuntimeError> {
    let target = memory.data_mut(caller);
    for (lo, bytes) in regions {
        let Some(end) = lo.checked_add(bytes.len()) else {
            continue;
        };
        if let Some(slot) = target.get_mut(*lo..end) {
            slot.copy_from_slice(bytes);
        }
    }
    Ok(())
}
