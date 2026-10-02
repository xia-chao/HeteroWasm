use std::path::Path;

use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::ControlFlow;
use heterowasm_dependence::accesses_in_loop;
use heterowasm_frontend::waffle::entity::EntityRef;
use heterowasm_frontend::waffle::{FuncDecl, Module};
use heterowasm_frontend::{expand_all_bodies, load_module, normalize_bulk_memory};
use heterowasm_legality::{Disposition, LoopLegal};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use heterowasm_wgsl::{artifact, Artifact};

use crate::error::CliError;


pub(crate) fn compile(input: &Path, output: &Path, trace: Trace) -> Result<(), CliError> {
    let _scope = trace.stage(Stage::Runtime, "compile");
    let raw = std::fs::read(input).map_err(|source| CliError::Read {
        path: input.to_path_buf(),
        source,
    })?;

    let prepared = prepare_input(&raw, &trace)?;


    std::fs::create_dir_all(output).map_err(|source| CliError::CreateDir {
        path: output.to_path_buf(),
        source,
    })?;
    std::fs::write(output.join("original.wasm"), &prepared.original)
        .map_err(|source| CliError::WriteRewritten { source })?;

    let normalized = normalize_bulk_memory(&prepared.bytes, &trace);
    let mut module =
        load_module(&normalized, &trace).map_err(|err| CliError::Message(err.to_string()))?;
    expand_all_bodies(&mut module, &trace).map_err(|err| CliError::Message(err.to_string()))?;
    std::fs::create_dir_all(output).map_err(|source| CliError::CreateDir {
        path: output.to_path_buf(),
        source,
    })?;

    let fused = emit_fused(&module, output, &trace)?;

    let fused_rewrite = fused.is_some();
    let (written, skipped, emitted) = match fused {
        Some(stem) => {

            println!("fusion succeeded; skip emitting single-loop kernels (they will not be used)");
            (1_usize, 0_usize, vec![(0_usize, 0_usize, stem)])
        }
        None => emit_single_loops(&module, output, &trace)?,
    };
    println!(
        "wrote {written} artifacts, skipped {skipped} loops (failed legality, outside subset, fixed trip below the measured crossover, or fewer than 32 shader multiplies)"
    );


    if !prepared.injected_from_wat {
        println!(
            "no rewritten.wasm produced: this input cannot use auto-rewrite (needs WAT text input)"
        );
        return Ok(());
    }
    let replaced = rewrite_and_write(&mut module, output, &trace, fused_rewrite)?;


    let live: Vec<String> = if fused_rewrite {
        emitted.into_iter().map(|(_, _, stem)| stem).collect()
    } else {
        replaced
            .iter()
            .filter_map(|(func, header)| {
                emitted
                    .iter()
                    .find(|(emitted_func, emitted_header, _)| {
                        emitted_func == func && emitted_header == header
                    })
                    .map(|(_, _, stem)| stem.clone())
            })
            .collect()
    };


    if live.is_empty() {
        println!("no extractable loop — produced neither kernel nor kernels.json");
        return Ok(());
    }
    write_kernel_index(output, &live)?;
    println!("wrote kernels.json ({} kernels, call order)", live.len());
    Ok(())
}


struct Prepared {
    bytes: Vec<u8>,

    original: Vec<u8>,
    injected_from_wat: bool,
}


fn prepare_input(raw: &[u8], trace: &Trace) -> Result<Prepared, CliError> {
    let disassembled;
    let wat_input: Option<&str> = match std::str::from_utf8(raw)
        .ok()
        .filter(|text| text.trim_start().starts_with("(module"))
    {
        Some(text) => Some(text),
        None => match wasmprinter::print_bytes(raw) {
            Ok(text) => {
                disassembled = text;
                Some(disassembled.as_str())
            }
            Err(_) => None,
        },
    };

    let Some(text) = wat_input else {
        return Ok(Prepared {
            original: raw.to_vec(),
            bytes: raw.to_vec(),
            injected_from_wat: false,
        });
    };

    let probe_bytes = wat::parse_str(text).map_err(|err| CliError::Message(err.to_string()))?;

    let arity = {

        let normalized = normalize_bulk_memory(&probe_bytes, trace);
        let mut probe =
            load_module(&normalized, trace).map_err(|err| CliError::Message(err.to_string()))?;
        expand_all_bodies(&mut probe, trace).map_err(|err| CliError::Message(err.to_string()))?;
        heterowasm_wgsl::plan_gpu_arity(&probe, trace)
    };
    match arity {

        Ok(shape) => {
            let bytes = heterowasm_wgsl::inject_dispatch_import(
                &probe_bytes,
                shape.fields,
                shape.results,
                trace,
            )
            .map_err(|reason| CliError::Inject { reason })?;
            Ok(Prepared {
                original: probe_bytes,
                bytes,
                injected_from_wat: true,
            })
        }
        Err(reason) => {
            println!("no extractable loop; compile as-is (no rewrite artifact): {reason}");
            Ok(Prepared {
                original: probe_bytes.clone(),
                bytes: probe_bytes,
                injected_from_wat: false,
            })
        }
    }
}


fn emit_fused(
    module: &Module<'_>,
    output: &Path,
    trace: &Trace,
) -> Result<Option<String>, CliError> {

    match heterowasm_wgsl::plan_gpu_fusion(module, trace) {
        Ok(Some(plan)) => {
            let body = module.funcs[plan.func].body().ok_or_else(|| {
                CliError::Message("fusion target is not a function body".to_string())
            })?;
            let fused = artifact(&plan.kernel, body)
                .map_err(|error| CliError::Message(error.to_string()))?;
            let name = match &module.funcs[plan.func] {
                FuncDecl::Body(_, name, _) if !name.is_empty() => name.clone(),
                _ => "fused".to_string(),
            };
            let stem = format!("{name}-fused");
            write_artifact(output, &stem, &fused)?;
            println!("wrote {stem}.wgsl + {stem}.json (**two loops fused into one kernel**)");
            Ok(Some(stem))
        }
        Ok(None) => Ok(None),
        Err(reason) => {
            eprintln!("fusion plan failed; fall back to single-loop path: {reason}");
            Ok(None)
        }
    }
}


fn emit_single_loops(
    module: &Module<'_>,
    output: &Path,
    trace: &Trace,
) -> Result<(usize, usize, Vec<(usize, usize, String)>), CliError> {
    let mut written = 0_usize;
    let mut skipped = 0_usize;
    let mut names: Vec<(usize, usize, String)> = Vec::new();
    for func in module.funcs.iter() {
        let FuncDecl::Body(_, name, body) = &module.funcs[func] else {
            continue;
        };
        for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze(body, &evolution, trace);
            let accesses = accesses_in_loop(recovery.accesses(), &natural_loop.blocks);
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            let legality =
                LoopLegal::judge(body, &natural_loop, &accesses, &evolution, module, trace);
            if !matches!(
                legality.disposition(),
                Disposition::Gpu | Disposition::GpuAfterGuard
            ) {
                skipped += 1;
                continue;
            }
            let kernel = match heterowasm_wgsl::lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                trace,
            ) {
                Ok(kernel) => kernel,
                Err(error) => {
                    eprintln!("skip {name}: {error}");
                    skipped += 1;
                    continue;
                }
            };
            if kernel.dispatch.fixed_trip_is_slower_than_cpu()
                || kernel.shader_multiplies_still_slower_than_cpu()
            {
                skipped += 1;
                continue;
            }
            let artifact =
                artifact(&kernel, body).map_err(|error| CliError::Message(error.to_string()))?;
            let stem = if name.is_empty() {
                format!("{}-{}", func.index(), natural_loop.header.index())
            } else {
                format!("{name}-{}-{}", func.index(), natural_loop.header.index())
            };
            write_artifact(output, &stem, &artifact)?;
            println!("wrote {stem}.wgsl + {stem}.json");
            names.push((func.index(), natural_loop.header.index(), stem));
            written += 1;
        }
    }
    Ok((written, skipped, names))
}

fn write_artifact(output: &Path, stem: &str, artifact: &Artifact<'_>) -> Result<(), CliError> {
    std::fs::write(output.join(format!("{stem}.wgsl")), artifact.wgsl)
        .map_err(|source| CliError::WriteShader { source })?;
    std::fs::write(output.join(format!("{stem}.json")), &artifact.manifest)
        .map_err(|source| CliError::WriteManifest { source })?;
    Ok(())
}


fn rewrite_and_write(
    module: &mut Module<'_>,
    output: &Path,
    trace: &Trace,
    fused: bool,
) -> Result<Vec<(usize, usize)>, CliError> {
    let outcome = if fused {
        heterowasm_wgsl::rewrite_module_fused(module, trace).map(|outcome| {
            println!(
                "rewrite via fusion path: extracted {} loops, fused = {}",
                outcome.loops_removed, outcome.fused
            );
            (0..outcome.loops_removed)
                .map(|index| (0_usize, index))
                .collect()
        })
    } else {
        heterowasm_wgsl::rewrite_module_for_gpu(module, trace)
    };

    let replaced = match outcome {
        Ok(pairs) => {
            let rewritten = module
                .to_wasm_bytes()
                .map_err(|err| CliError::Message(err.to_string()))?;
            let path = output.join("rewritten.wasm");
            std::fs::write(&path, &rewritten)
                .map_err(|source| CliError::WriteRewritten { source })?;
            println!(
                "wrote rewritten.wasm ({} loops replaced with __hw_dispatch calls, {} bytes)",
                pairs.len(),
                rewritten.len()
            );
            pairs
        }
        Err(reason) => {
            println!("no extractable loop; no rewrite artifact: {reason}");
            Vec::new()
        }
    };
    Ok(replaced)
}


fn write_kernel_index(dir: &Path, names: &[String]) -> Result<(), CliError> {
    let list = names
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let text =
        format!("{{\n  \"schema\": \"heterowasm.kernels.v1\",\n  \"kernels\": [{list}]\n}}\n");
    std::fs::write(dir.join("kernels.json"), text)
        .map_err(|source| CliError::WriteManifest { source })
}
