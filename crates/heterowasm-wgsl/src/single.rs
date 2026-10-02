use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::ControlFlow;
use heterowasm_legality::{Disposition, LoopLegal};
use heterowasm_scev::{canonical_value, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{Block, Func, Module, Terminator, Value, ValueDef};

use crate::lower::lower;
use crate::region::{offload_region_calls, RegionCalls};
use crate::types::{Dispatch, DispatchShape, Kernel};


pub(crate) struct OffloadPlan {
    pub(crate) func: Func,
    pub(crate) header: Block,
    pub(crate) exit: Block,
    pub(crate) members: Vec<Block>,
    pub(crate) kernel: Kernel,
}


pub(crate) fn plan_first_gpu_loop(
    module: &Module<'_>,
    trace: &Trace,
) -> Result<OffloadPlan, String> {
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze(body, &evolution, trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            let legality =
                LoopLegal::judge(body, &natural_loop, &accesses, &evolution, module, trace);
            if !matches!(
                legality.disposition(),
                Disposition::Gpu | Disposition::GpuAfterGuard
            ) {
                continue;
            }
            let kernel = lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                trace,
            )
            .map_err(|error| error.to_string())?;
            let exit = loop_exit(body, &natural_loop.blocks)?;
            return Ok(OffloadPlan {
                func,
                header: natural_loop.header,
                exit,
                members: natural_loop.blocks.clone(),
                kernel,
            });
        }
    }
    Err("no extractable loop found".to_string())
}


pub(crate) fn loop_exit(body: &waffle::FunctionBody, members: &[Block]) -> Result<Block, String> {
    members
        .iter()
        .filter_map(|block| body.blocks.get(*block))
        .flat_map(|definition| match &definition.terminator {
            Terminator::Br { target } => vec![target.block],
            Terminator::CondBr {
                if_true, if_false, ..
            } => vec![if_true.block, if_false.block],
            _ => vec![],
        })
        .find(|block| !members.contains(block))
        .ok_or_else(|| "natural loop has no exit block".to_string())
}


pub(crate) fn offload_loop(
    module: &mut Module<'_>,
    func: Func,
    header: Block,
    exit: Block,
    members: &[Block],
    fields: &[Value],
    selector: u32,
    trip: Option<crate::region::RuntimeTrip>,
) -> Result<(), String> {

    let explicit: Vec<Value> = {
        let body = module.funcs[func].body().ok_or("not a function body")?;

        let inbound = crate::region::region_preheader_arguments(module, func, header, members)
            .unwrap_or_default();
        fields
            .iter()
            .map(|field| {
                let canonical = canonical_value(body, *field);
                match body.values.get(canonical) {
                    Some(ValueDef::BlockParam(block, index, _)) if *block == header => {
                        inbound.get(*index as usize).copied().unwrap_or(*field)
                    }
                    _ => *field,
                }
            })
            .collect()
    };
    offload_region_calls(
        module,
        func,
        header,
        exit,
        members,
        RegionCalls {
            header_params: &[],
            explicit: Some(&explicit),
            extra: &[],
            selector,
            extra_selectors: &[],
            trip,
        },
    )
}


pub fn rewrite_module_for_gpu(
    module: &mut Module<'_>,
    trace: &Trace,
) -> Result<Vec<(usize, usize)>, String> {

    fn runtime_trip(kernel: &crate::Kernel) -> Option<crate::region::RuntimeTrip> {
        let (limit, mask) = match kernel.dispatch {
            Dispatch::FromField(slot) => (slot, u32::MAX),
            Dispatch::FromFieldMask(slot, mask) => (slot, mask),
            _ => return None,
        };
        if let Some(start) = kernel.index_field {
            let Ok(stride) = u32::try_from(kernel.max_stride_bytes) else {
                return None;
            };
            if stride < 4 || stride % 4 != 0 {
                return None;
            }
            let guard = format!("params.p{start} + 4u * gid.x >= params.p{limit}");
            if !kernel.source.contains(&guard) {
                return None;
            }
            return Some(crate::region::RuntimeTrip {
                start_field: Some(start),
                limit_field: limit,
                stride_bytes: stride,
                mask,
            });
        }
        Some(crate::region::RuntimeTrip {
            start_field: None,
            limit_field: limit,
            stride_bytes: 1,
            mask,
        })
    }

    fn collect(module: &Module<'_>, trace: &Trace) -> Vec<OffloadPlan> {
        let mut plans = Vec::new();
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
                let recovery = AddressRecovery::analyze(body, &evolution, trace);
                let accesses: Vec<_> = recovery
                    .accesses()
                    .iter()
                    .filter(|access| natural_loop.blocks.contains(&access.block))
                    .cloned()
                    .collect();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
                let legality =
                    LoopLegal::judge(body, &natural_loop, &accesses, &evolution, module, trace);
                if !matches!(
                    legality.disposition(),
                    Disposition::Gpu | Disposition::GpuAfterGuard
                ) {
                    continue;
                }
                let Ok(kernel) = lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    trace,
                ) else {
                    continue;
                };
                if kernel.dispatch.fixed_trip_is_slower_than_cpu()
                    || kernel.shader_multiplies_still_slower_than_cpu()
                {
                    continue;
                }
                let Ok(exit) = loop_exit(body, &natural_loop.blocks) else {
                    continue;
                };
                let overlaps = plans.iter().any(|plan: &OffloadPlan| {
                    plan.func == func
                        && natural_loop
                            .blocks
                            .iter()
                            .any(|block| plan.members.contains(block))
                });
                if overlaps {
                    continue;
                }
                plans.push(OffloadPlan {
                    func,
                    header: natural_loop.header,
                    exit,
                    members: natural_loop.blocks.clone(),
                    kernel,
                });
            }
        }
        plans
    }

    let plans = collect(module, trace);
    if plans.is_empty() {
        return Err("no extractable loop found".to_string());
    }
    let mut replaced = Vec::new();
    for (index, plan) in plans.into_iter().enumerate() {
        let selector = u32::try_from(index).map_err(|_| "kernel index exceeds u32".to_string())?;
        if let Err(reason) = offload_loop(
            module,
            plan.func,
            plan.header,
            plan.exit,
            &plan.members,
            &plan.kernel.fields,
            selector,
            runtime_trip(&plan.kernel),
        ) {
            if replaced.is_empty() {
                return Err(reason);
            }
            trace
                .info(Stage::Wgsl, "rewrite stopped")
                .field("reason", reason)
                .field("replaced", replaced.len())
                .emit();
            break;
        }
        replaced.push((plan.func.index(), plan.header.index()));
    }
    Ok(replaced)
}


pub fn plan_gpu_arity(module: &Module<'_>, trace: &Trace) -> Result<DispatchShape, String> {


    let first = plan_first_gpu_loop(module, trace).ok().filter(|plan| {
        !plan.kernel.dispatch.fixed_trip_is_slower_than_cpu()
            && !plan.kernel.shader_multiplies_still_slower_than_cpu()
    });
    let mut widest = first
        .as_ref()
        .map(|plan| plan.kernel.fields.len())
        .unwrap_or(0);

    let mut widest_results = first
        .as_ref()
        .map(|plan| plan.kernel.results.len())
        .unwrap_or(0);
    let mut found = first.is_some();
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze(body, &evolution, trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            if let Ok(kernel) = lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                trace,
            ) {
                if kernel.dispatch.fixed_trip_is_slower_than_cpu()
                    || kernel.shader_multiplies_still_slower_than_cpu()
                {
                    continue;
                }
                widest = widest.max(kernel.fields.len());
                widest_results = widest_results.max(kernel.results.len());
                found = true;
            }
        }
    }
    if !found {
        return Err("module has no emittable loop".to_string());
    }
    Ok(DispatchShape {
        fields: widest,
        results: widest_results,
    })
}
