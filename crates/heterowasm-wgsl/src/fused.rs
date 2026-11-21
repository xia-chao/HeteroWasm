use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use waffle::{Module, Value};

use crate::fusion::{
    collect_region_uses, fusable_pairs, region_span, resolve_at_entry, FusionInput,
};
use crate::ir::header_parameter_indices;
use crate::lower::lower;
use crate::region::{offload_region_calls, region_preheader_arguments, RegionCalls};
use crate::single::rewrite_module_for_gpu;
use crate::types::{Kernel, RewriteOutcome};


struct Chain {
    producer_header: waffle::Block,
    producer_members: Vec<waffle::Block>,
    consumer_header: waffle::Block,
    consumer_exit: waffle::Block,
    consumer_members: Vec<waffle::Block>,
}

pub fn rewrite_module_fused(
    module: &mut Module<'_>,
    trace: &Trace,
) -> Result<RewriteOutcome, String> {

    let mut input = FusionInput::default();
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        input.by_func.insert(func, collect_region_uses(body, trace));
    }
    rewrite_module_fused_from(module, trace, &input)
}


pub mod from_plans {
    use super::*;


    pub fn rewrite(
        module: &mut Module<'_>,
        trace: &Trace,
        input: &FusionInput,
    ) -> Result<RewriteOutcome, String> {
        let mut pair = None;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let Some(collected) = input.loops_of(func) else {
                continue;
            };
            if let Some((region, producer, consumer)) =
                fusable_pairs(body, collected).into_iter().next()
            {
                pair = Some((func, region, producer, consumer, collected));
                break;
            }
        }
        let Some((func, _region, producer_index, consumer_index, collected)) = pair else {
            let removed = rewrite_module_for_gpu(module, trace)?.len();
            return Ok(RewriteOutcome {
                loops_removed: removed,
                fused: false,
            });
        };

        let chain = {
            let body = module.funcs[func].body().ok_or("not a function body")?;
            let (ph, _, pm) = region_span(body, &collected[producer_index].0)?;
            let (ch, ce, cm) = region_span(body, &collected[consumer_index].0)?;
            Chain {
                producer_header: ph,
                producer_members: pm,
                consumer_header: ch,
                consumer_exit: ce,
                consumer_members: cm,
            }
        };


        let mut members = chain.producer_members.clone();
        for block in &chain.consumer_members {
            if !members.contains(block) {
                members.push(*block);
            }
        }


        let producer_args =
            region_preheader_arguments(module, func, chain.producer_header, &members)?;
        if producer_args.len() < 2 {
            return Err(
                "producer needs at least two block params: upper bound + one base".to_string(),
            );
        }


        let consumer_output = {
            let body = module.funcs[func].body().ok_or("not a function body")?;
            let positions = lower_fields(body, &collected[consumer_index].0, trace)?;
            let output_position = positions
                .get(1)
                .copied()
                .ok_or("consumer has no output-base field")?;
            let output_param = body.blocks[chain.consumer_header]
                .params
                .get(output_position)
                .map(|(_, value)| *value)
                .ok_or("consumer output base is not among loop-header block parameters")?;
            resolve_at_entry(body, output_param)?
        };


        let producer_positions = {
            let body = module.funcs[func].body().ok_or("not a function body")?;
            lower_fields(body, &collected[producer_index].0, trace)?
        };
        let pick = |field: usize| -> Result<Value, String> {
            let position = producer_positions
                .get(field)
                .copied()
                .ok_or_else(|| format!("producer has no field {field}"))?;
            producer_args.get(position).copied().ok_or_else(|| {
                format!("producer predecessor does not pass a value for block parameter {position}")
            })
        };

        let mut fused_args = vec![pick(0)?, consumer_output];
        for field in 2..producer_positions.len() {
            fused_args.push(pick(field)?);
        }


        trace
            .info(Stage::Wgsl, "fusion arguments resolved")
            .subject("fusion")
            .field("producer_args", format!("{producer_args:?}"))
            .field("consumer_output", format!("{consumer_output:?}"))
            .field("fused_args", format!("{fused_args:?}"))
            .emit();

        offload_region_calls(
            module,
            func,
            chain.producer_header,
            chain.consumer_exit,
            &members,
            RegionCalls {
                header_params: &[],
                entry_values: &[],
                explicit: Some(&fused_args),
                extra: &[],
                selector: 0,
                extra_selectors: &[],
                trip: None,
            },
        )?;
        Ok(RewriteOutcome {
            loops_removed: 2,
            fused: true,
        })
    }
}


pub mod reduction_chain {
    use std::collections::HashSet;

    use waffle::{Block, Func, Value};

    use super::*;

    pub fn rewrite(
        module: &mut Module<'_>,
        trace: &Trace,
        input: &FusionInput,
    ) -> Result<RewriteOutcome, String> {
        let mut pair = None;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let Some(collected) = input.loops_of(func) else {
                continue;
            };
            if let Some((_, producer_index, consumer_index)) =
                fusable_pairs(body, collected).into_iter().next()
            {
                pair = Some((func, producer_index, consumer_index, collected));
                break;
            }
        }
        let Some((func, producer_index, consumer_index, collected)) = pair else {
            let removed = rewrite_module_for_gpu(module, trace)?.len();
            return Ok(RewriteOutcome {
                loops_removed: removed,
                fused: false,
            });
        };

        let consumed: HashSet<Func> = [func].into_iter().collect();
        let Some(plan) =
            crate::fusion::plan_gpu_fusion_reduction_from(module, trace, Some(&consumed), input)?
        else {
            return Err("fused reduction pair no longer plans".to_string());
        };

        let (producer_header, _, producer_members, consumer_exit, consumer_members, explicit) = {
            let body = module.funcs[func].body().ok_or("not a function body")?;
            let (producer_header, _, producer_members) =
                region_span(body, &collected[producer_index].0)?;
            let (_, consumer_exit, consumer_members) =
                region_span(body, &collected[consumer_index].0)?;
            let explicit = plan
                .kernel
                .fields
                .iter()
                .map(|field| resolve_at_entry(body, *field))
                .collect::<Result<Vec<Value>, String>>()?;
            (
                producer_header,
                (),
                producer_members,
                consumer_exit,
                consumer_members,
                explicit,
            )
        };
        let mut members = producer_members;
        for block in consumer_members {
            if !members.contains(&block) {
                members.push(block);
            }
        }

        let entry_values: Vec<(Value, Value)> = {
            let body = module.funcs[func].body().ok_or("not a function body")?;
            let evolution = ScalarEvolution::analyze(body, &collected[consumer_index].0, trace);
            let members_set: HashSet<Block> = members.iter().copied().collect();
            crate::live_out::live_outs(body, &members_set)
                .into_iter()
                .filter(|value| {
                    !crate::live_out::is_pass_through(body, producer_header, &members_set, *value)
                })
                .filter_map(|value| {
                    let canonical = heterowasm_scev::canonical_value(body, value);
                    if !evolution.invariant_parameters().contains(&canonical) {
                        return None;
                    }
                    resolve_at_entry(body, value)
                        .ok()
                        .map(|source| (value, source))
                })
                .collect()
        };
        offload_region_calls(
            module,
            func,
            producer_header,
            consumer_exit,
            &members,
            RegionCalls {
                header_params: &[],
                entry_values: &entry_values,
                explicit: Some(&explicit),
                extra: &[],
                selector: 0,
                extra_selectors: &[],
                trip: None,
            },
        )?;
        Ok(RewriteOutcome {
            loops_removed: 2,
            fused: true,
        })
    }
}

pub use reduction_chain::rewrite as rewrite_module_fused_reduction_from;

pub use from_plans::rewrite as rewrite_module_fused_from;


fn lower_fields(
    body: &waffle::FunctionBody,
    natural_loop: &heterowasm_cfg::NaturalLoop,
    trace: &Trace,
) -> Result<Vec<usize>, String> {
    let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
    let recovery = AddressRecovery::analyze_loop(body, &evolution, &natural_loop.blocks, trace);
    let accesses: Vec<_> = recovery.accesses().to_vec();
    let extent = LoopExtent::analyze(body, natural_loop, &evolution);
    let kernel: Kernel = lower(body, natural_loop, &accesses, &evolution, extent, 64, trace)
        .map_err(|error| error.to_string())?;
    header_parameter_indices(body, natural_loop.header, &kernel.fields)
}
