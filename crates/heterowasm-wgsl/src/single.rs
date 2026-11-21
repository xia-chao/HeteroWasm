use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::ControlFlow;
use heterowasm_legality::{Disposition, LoopLegal};
use heterowasm_scev::{canonical_value, ScalarEvolution};
use heterowasm_trace::Trace;
use waffle::{Block, Func, Module, Terminator, Value, ValueDef};

use crate::lower::lower;
use crate::plans::OffloadPlan;
use crate::region::{offload_region_calls, RegionCalls};
use crate::types::{DispatchShape, LoopRegion};


pub(crate) fn plan_first_gpu_loop(
    module: &Module<'_>,
    trace: &Trace,
) -> Result<OffloadPlan, String> {
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        let memory = AddressRecovery::memory_access_values(body);
        let legality_effects = heterowasm_legality::effects::EffectScan::scan(body);
        for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze_loop_indexed(
                body,
                &evolution,
                &natural_loop.blocks,
                trace,
                &memory,
            );
            let accesses: Vec<_> = recovery.accesses().to_vec();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            let legality = LoopLegal::judge_with_cache(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                module,
                trace,
                &legality_effects,
            );
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
    region: LoopRegion<'_>,
    fields: &[Value],
    selector: u32,
    trip: Option<crate::region::RuntimeTrip>,
) -> Result<(), String> {
    let LoopRegion {
        header,
        exit,
        members,
    } = region;

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
            entry_values: &[],
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
    let plans = crate::plans::plan_gpu_loops(module, trace);
    crate::plans::rewrite_with_plans(module, &plans.rewritable, trace)
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
        let effects = crate::plans::EffectFacts::of(body);
        let memory = AddressRecovery::memory_access_values(body);
        for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {

            let members: std::collections::HashSet<Block> =
                natural_loop.blocks.iter().copied().collect();
            if effects.cheaply_unrewardable(&members) {
                continue;
            }
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze_loop_indexed(
                body,
                &evolution,
                &natural_loop.blocks,
                trace,
                &memory,
            );
            let accesses: Vec<_> = recovery.accesses().to_vec();
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
