use std::path::Path;

use heterowasm_trace::{Stage, Trace};
use wasmtime::{Engine, Extern, Instance, Linker, Module, Store, Val};

use crate::error::RuntimeError;
use crate::gpu::{Compiled, GpuContext};
use crate::host::{import_arity, import_results, link_dynamic, HostState};
use crate::types::{KernelSpec, Shader};


pub struct Runtime {
    store: Store<HostState>,
    instance: Instance,
    trace: Trace,
    memory_name: String,
}

impl Runtime {

    pub fn load(dir: &Path, trace: Trace) -> Result<Self, RuntimeError> {
        let _scope = trace.stage(Stage::Runtime, "load");
        let kernels = crate::manifest::load_dir(dir)?;

        let wasm_path = dir.join("rewritten.wasm");
        let wasm = std::fs::read(&wasm_path).map_err(|source| RuntimeError::Read {
            path: wasm_path.clone(),
            source,
        })?;

        let context = GpuContext::new()?;
        let compiled: Vec<Compiled> = kernels
            .iter()
            .map(|kernel| context.compile(&shader_of(kernel)))
            .collect::<Result<_, _>>()?;

        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).map_err(|error| RuntimeError::Execute {
            reason: format!("failed to parse wasm: {error}"),
        })?;


        let (arity, results) = module
            .imports()
            .find_map(|import| {
                if import.module() != "heterowasm" || import.name() != "__hw_dispatch" {
                    return None;
                }
                let wasmtime::ExternType::Func(ty) = import.ty() else {
                    return None;
                };
                let params = ty.params().len();
                let results = ty.results().len();
                Some((params, results))
            })
            .unwrap_or_else(|| (import_arity(&kernels), import_results(&kernels)));
        let mut linker: Linker<HostState> = Linker::new(&engine);
        link_dynamic(&engine, &mut linker, arity, results)?;

        for import in module.imports() {
            if import.module() == "heterowasm" && import.name() == "__hw_dispatch" {
                continue;
            }
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
                .map_err(|error| RuntimeError::Execute {
                    reason: error.to_string(),
                })?;
        }
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
        let memory_name = memory_name.ok_or(RuntimeError::NoMemory)?;
        trace
            .info(Stage::Runtime, "memory export")
            .field("name", memory_name.as_str())
            .emit();

        let slots = kernels.len();
        let mut store = Store::new(
            &engine,
            HostState {
                kernels,
                compiled,
                context,
                workspaces: (0..slots).map(|_| None).collect(),
                calls: 0,
                total: 0,
                trace: trace.share(),
                memory_export: memory_name.clone(),
            },
        );
        let instance =
            linker
                .instantiate(&mut store, &module)
                .map_err(|error| RuntimeError::Execute {
                    reason: format!("instantiation failed: {error}"),
                })?;

        trace
            .info(Stage::Runtime, "runtime loaded")
            .field("kernels", slots)
            .field("arity", arity)
            .field("adapter", store.data().context.adapter().to_string())
            .field("wasm_bytes", wasm.len())
            .emit();

        Ok(Runtime {
            store,
            instance,

            trace: trace.share(),
            memory_name,
        })
    }


    pub fn run(&mut self, entry: &str, args: &[Val]) -> Result<Vec<Val>, RuntimeError> {
        let _scope = self.trace.stage(Stage::Runtime, "run");

        self.store.data_mut().calls = 0;
        let func = self
            .instance
            .get_func(&mut self.store, entry)
            .ok_or_else(|| RuntimeError::NoEntry {
                name: entry.to_string(),
            })?;

        let ty = func.ty(&self.store);
        let mut results = vec![Val::I32(0); ty.results().len()];
        func.call(&mut self.store, args, &mut results)
            .map_err(|error| RuntimeError::Execute {
                reason: describe(&error),
            })?;

        self.trace
            .info(Stage::Runtime, "call finished")
            .field("entry", entry)
            .field("dispatches", self.store.data().total)
            .field("results", results.len())
            .emit();
        Ok(results)
    }


    pub fn memory(&mut self) -> Result<Vec<u8>, RuntimeError> {
        let memory = match self.instance.get_export(&mut self.store, &self.memory_name) {
            Some(Extern::Memory(memory)) => memory,
            _ => return Err(RuntimeError::NoMemory),
        };
        Ok(memory.data(&self.store).to_vec())
    }


    pub fn write_memory(&mut self, offset: usize, bytes: &[u8]) -> Result<(), RuntimeError> {
        let memory = match self.instance.get_export(&mut self.store, &self.memory_name) {
            Some(Extern::Memory(memory)) => memory,
            _ => return Err(RuntimeError::NoMemory),
        };
        let target = memory.data_mut(&mut self.store);
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| RuntimeError::Execute {
                reason: "memory write offset overflow".to_string(),
            })?;
        if end > target.len() {
            return Err(RuntimeError::Execute {
                reason: format!(
                    "write [{offset}, {end}) exceeds memory of {} bytes",
                    target.len()
                ),
            });
        }
        target[offset..end].copy_from_slice(bytes);
        Ok(())
    }


    pub fn adapter(&self) -> &str {
        self.store.data().context.adapter()
    }


    pub fn dispatches(&self) -> usize {
        self.store.data().total
    }
}


fn describe(error: &wasmtime::Error) -> String {

    let mut text = error.to_string();
    for cause in error.chain().skip(1) {
        let rendered = cause.to_string();
        if !text.contains(&rendered) {
            text.push_str("\n  reason: ");
            text.push_str(&rendered);
        }
    }
    text
}


fn shader_of(kernel: &KernelSpec) -> Shader<'_> {
    Shader {
        source: &kernel.source,
        workgroup_size: kernel.workgroup_size,
        dispatch: kernel.dispatch,
        min_offset_bytes: kernel.min_offset_bytes,
        max_constant_bytes: kernel.max_constant_bytes,
        max_stride_bytes: kernel.max_stride_bytes,
        index_field: kernel.index_field,
        launch_count: kernel.launch_count,
    }
}
