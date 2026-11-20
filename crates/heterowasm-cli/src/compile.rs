use std::collections::HashSet;
use std::path::Path;

use heterowasm_frontend::waffle::entity::EntityRef;
use heterowasm_frontend::waffle::{Func, FuncDecl, Module};
use heterowasm_frontend::{expand_all_bodies, load_module, normalize_bulk_memory};
use heterowasm_trace::{Stage, Trace};
use heterowasm_wgsl::{artifact, Artifact, LoopPlanSet};

use crate::error::CliError;
use crate::types::KernelEmission;


pub(crate) fn compile(
    input: &Path,
    output: &Path,
    trace: Trace,
    require_offload: bool,
) -> Result<(), CliError> {
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


    let shift = usize::from(prepared.injected_from_wat);
    let shape_matches = prepared.probe_funcs + shift == module.funcs.len()
        && prepared
            .probe_shape
            .iter()
            .enumerate()
            .all(|(index, shape)| {
                let body = module
                    .funcs
                    .get(Func::new(index + shift))
                    .and_then(FuncDecl::body);
                match shape {
                    Some((blocks, values)) => body.is_some_and(|body| {
                        body.blocks.len() == *blocks && body.values.len() == *values
                    }),
                    None => body.is_none(),
                }
            });
    let plan_set = match prepared.plans {
        Some(plans) => {
            let shifted = plans.shifted(shift);
            if shape_matches && shifted.fits_module(&module, &trace) {
                shifted
            } else {
                trace
                    .info(
                        Stage::Runtime,
                        "plan reuse rejected; re-planning on the loaded module",
                    )
                    .field("function_shift", shift)
                    .field("shape_matches", shape_matches)
                    .emit();
                heterowasm_wgsl::plan_gpu_loops(&module, &trace)
            }
        }
        None => heterowasm_wgsl::plan_gpu_loops(&module, &trace),
    };
    let fusion_only: HashSet<Func> = plan_set.fusion_candidates.clone();
    let fused = if fusion_only.is_empty() {
        None
    } else {
        emit_fused(&module, output, &trace, &plan_set)?
    };

    let fused_rewrite = fused.is_some();
    let KernelEmission {
        written,
        skipped,
        emitted,
    } = match fused {
        Some(stem) => {

            println!("fusion succeeded; skip emitting single-loop kernels (they will not be used)");
            KernelEmission {
                written: 1,
                skipped: 0,
                emitted: vec![(0, 0, stem)],
            }
        }
        None => emit_single_loops(&module, &plan_set, output, &trace)?,
    };

    if fused_rewrite {
        println!("wrote {written} artifacts (fusion path; single-loop kernels are not emitted)");
    } else {
        const KINDS: [&str; 4] = ["control", "memory", "dependence", "effects"];
        let mut legality = Vec::new();
        let mut legality_total = 0;
        for (index, kind) in KINDS.iter().enumerate() {
            let count = plan_set.skipped_illegal[index] + plan_set.skipped_unknown[index];
            legality_total += count;
            if count > 0 {
                legality.push(format!("{kind} {count}"));
            }
        }

        println!(
            "wrote {written} artifacts, skipped {skipped} loops: {} with side effects (calls, bulk memory, global/table writes), {legality_total} by legality ({}), {} below the cost gate, {} not lowerable, {} without an exit",
            plan_set.skipped_side_effects,
            if legality.is_empty() {
                "none".to_string()
            } else {
                legality.join(", ")
            },
            plan_set.skipped_gate,
            plan_set.skipped_lower,
            plan_set.skipped_exit,
        );
    }

    if require_offload && written == 0 {
        return Err(CliError::Message(
            "no loop was offloaded in this run (--require-offload)".to_string(),
        ));
    }


    if !prepared.injected_from_wat {
        println!(
            "no rewritten.wasm produced: this input cannot use auto-rewrite (needs WAT text input)"
        );
        return Ok(());
    }
    let replaced = rewrite_and_write(&mut module, output, &trace, fused_rewrite, &plan_set)?;


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

    plans: Option<LoopPlanSet>,

    probe_shape: Vec<Option<(usize, usize)>>,

    probe_funcs: usize,
}


fn prepare_input(raw: &[u8], trace: &Trace) -> Result<Prepared, CliError> {

    let probe_bytes = match std::str::from_utf8(raw)
        .ok()
        .filter(|text| text.trim_start().starts_with("(module"))
    {
        Some(text) => wat::parse_str(text).map_err(|err| CliError::Message(err.to_string()))?,
        None => raw.to_vec(),
    };

    let normalized = normalize_bulk_memory(&probe_bytes, trace);
    let mut probe =
        load_module(&normalized, trace).map_err(|err| CliError::Message(err.to_string()))?;
    expand_all_bodies(&mut probe, trace).map_err(|err| CliError::Message(err.to_string()))?;


    let plans = heterowasm_wgsl::plan_gpu_loops(&probe, trace);
    let probe_funcs = probe.funcs.len();
    let probe_shape: Vec<Option<(usize, usize)>> = probe
        .funcs
        .iter()
        .map(|func| {
            probe.funcs[func]
                .body()
                .map(|body| (body.blocks.len(), body.values.len()))
        })
        .collect();


    let width = plans
        .eligible
        .iter()
        .fold((0_usize, 0_usize), |(fields, results), plan| {
            (
                fields.max(plan.kernel.fields.len()),
                results.max(plan.kernel.results.len()),
            )
        });
    let arity = if plans.eligible.is_empty() {
        heterowasm_wgsl::plan_gpu_arity(&probe, trace)
    } else {
        Ok(heterowasm_wgsl::DispatchShape {
            fields: width.0,
            results: width.1,
        })
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
                plans: Some(plans),
                probe_shape,
                probe_funcs,
            })
        }
        Err(reason) => {
            println!("no extractable loop; compile as-is (no rewrite artifact): {reason}");
            Ok(Prepared {
                original: probe_bytes.clone(),
                bytes: probe_bytes,
                injected_from_wat: false,
                plans: Some(plans),
                probe_shape,
                probe_funcs,
            })
        }
    }
}


fn emit_fused(
    module: &Module<'_>,
    output: &Path,
    trace: &Trace,
    plan_set: &LoopPlanSet,
) -> Result<Option<String>, CliError> {


    let only = Some(&plan_set.fusion_candidates);
    match heterowasm_wgsl::plan_gpu_fusion_from(module, trace, only, &plan_set.fusion_inputs) {
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
    plans: &LoopPlanSet,
    output: &Path,
    trace: &Trace,
) -> Result<KernelEmission, CliError> {
    let _ = trace;

    for (name, error) in &plans.lower_failures {
        eprintln!("skip {name}: {error}");
    }
    let mut names: Vec<(usize, usize, String)> = Vec::new();
    for plan in &plans.eligible {
        let FuncDecl::Body(_, name, body) = &module.funcs[plan.func] else {
            continue;
        };
        let artifact =
            artifact(&plan.kernel, body).map_err(|error| CliError::Message(error.to_string()))?;
        let stem = if name.is_empty() {
            format!("{}-{}", plan.func.index(), plan.header.index())
        } else {
            format!("{name}-{}-{}", plan.func.index(), plan.header.index())
        };
        write_artifact(output, &stem, &artifact)?;
        println!("wrote {stem}.wgsl + {stem}.json");
        names.push((plan.func.index(), plan.header.index(), stem));
    }
    Ok(KernelEmission {
        written: names.len(),
        skipped: plans.skipped(),
        emitted: names,
    })
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
    plan_set: &LoopPlanSet,
) -> Result<Vec<(usize, usize)>, CliError> {
    let outcome = if fused {
        heterowasm_wgsl::rewrite_module_fused_from(module, trace, &plan_set.fusion_inputs).map(
            |outcome| {
                println!(
                    "rewrite via fusion path: extracted {} loops, fused = {}",
                    outcome.loops_removed, outcome.fused
                );
                (0..outcome.loops_removed)
                    .map(|index| (0_usize, index))
                    .collect()
            },
        )
    } else {
        heterowasm_wgsl::rewrite_with_plans(module, &plan_set.rewritable, trace)
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
