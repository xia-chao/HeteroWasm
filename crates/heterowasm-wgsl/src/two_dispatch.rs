use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use waffle::{Module, Value, ValueDef};

use crate::fusion::{collect_region_uses, fusable_pairs, region_span};
use crate::ir::header_parameter_indices;
use crate::lower::lower;
use crate::region::{offload_region_calls, region_preheader_arguments, RegionCalls};
use crate::single::rewrite_module_for_gpu;
use crate::types::{Kernel, RewriteOutcome};

pub fn rewrite_module_two_dispatches(
    module: &mut Module<'_>,
    trace: &Trace,
) -> Result<RewriteOutcome, String> {
    let mut pair = None;
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        let collected = collect_region_uses(body, trace);
        if let Some((region, producer, consumer)) =
            fusable_pairs(body, &collected).into_iter().next()
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

    let (producer_header, producer_members, consumer_header, consumer_exit, consumer_members) = {
        let body = module.funcs[func].body().ok_or("not a function body")?;
        let (ph, _, pm) = region_span(body, &collected[producer_index].0)?;
        let (ch, ce, cm) = region_span(body, &collected[consumer_index].0)?;
        (ph, pm, ch, ce, cm)
    };

    let mut members = producer_members.clone();
    for block in &consumer_members {
        if !members.contains(block) {
            members.push(*block);
        }
    }

    let producer_args = region_preheader_arguments(module, func, producer_header, &members)?;
    let producer_fields = {
        let body = module.funcs[func].body().ok_or("not a function body")?;
        let (natural_loop, _, _) = &collected[producer_index];
        lower_loop_fields(body, natural_loop, trace)?
    };
    let mut first_call = Vec::new();
    for position in &producer_fields {
        first_call.push(producer_args.get(*position).copied().ok_or_else(|| {
            format!("producer predecessor does not pass a value for block parameter {position}")
        })?);
    }

    let second_call = {
        let body = module.funcs[func].body().ok_or("not a function body")?;
        let (natural_loop, _, _) = &collected[consumer_index];
        let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
        let recovery = AddressRecovery::analyze(body, &evolution, trace);
        let accesses: Vec<_> = recovery
            .accesses()
            .iter()
            .filter(|access| natural_loop.blocks.contains(&access.block))
            .cloned()
            .collect();
        let extent = LoopExtent::analyze(body, natural_loop, &evolution);
        let kernel = lower(body, natural_loop, &accesses, &evolution, extent, 64, trace)
            .map_err(|error| error.to_string())?;
        let positions = header_parameter_indices(body, natural_loop.header, &kernel.fields)?;
        let header_params: Vec<Value> = body.blocks[consumer_header]
            .params
            .iter()
            .map(|(_, value)| *value)
            .collect();
        let mut args = Vec::new();
        for (field, position) in positions.iter().enumerate() {
            let value = header_params.get(*position).copied().ok_or_else(|| {
                format!("consumer field {field} is not among loop-header block parameters")
            })?;
            let as_producer_field = producer_fields.iter().position(|producer_position| {
                body.blocks[producer_header]
                    .params
                    .get(*producer_position)
                    .map(|(_, producer_value)| *producer_value)
                    == Some(value)
            });
            if let Some(field_index) = as_producer_field {
                args.push(first_call[field_index]);
                continue;
            }
            args.push(
                crate::fusion::resolve_at_entry(body, value).map_err(|error| {
                    format!(
                        "consumer field {field} cannot get the region-predecessor value: {error}"
                    )
                })?,
            );
        }
        args
    };


    let arity = first_call.len().max(second_call.len());
    let mut padded_first = first_call.clone();
    let mut padded_second = second_call.clone();
    let filler = producer_args.first().copied().unwrap_or(first_call[0]);
    padded_first.resize(arity, filler);
    padded_second.resize(arity, filler);

    trace
        .info(Stage::Wgsl, "two-dispatch rewrite planned")
        .subject("two_dispatch")
        .field("producer_fields", producer_fields.len())
        .field("consumer_fields", second_call.len())
        .field("arity", arity)
        .field("first_call", format!("{first_call:?}"))
        .field("second_call", format!("{second_call:?}"))
        .field("padded_first", format!("{padded_first:?}"))
        .field("padded_second", format!("{padded_second:?}"))
        .field("members", members.len())
        .emit();

    let extra_selectors = [1_u32];
    offload_region_calls(
        module,
        func,
        producer_header,
        consumer_exit,
        &members,
        RegionCalls {
            header_params: &[],
            explicit: Some(&padded_first),
            extra: &[padded_second],
            selector: 0,
            extra_selectors: &extra_selectors,
            trip: None,
        },
    )?;


    let call_sites = {
        let body = module.funcs[func].body().ok_or("not a function body")?;
        let mut total = 0_usize;
        for block in body.blocks.iter() {
            let Some(definition) = body.blocks.get(block) else {
                continue;
            };
            for inst in &definition.insts {
                if matches!(
                    body.values.get(*inst),
                    Some(ValueDef::Operator(waffle::Operator::Call { .. }, _, _))
                ) {
                    total += 1;
                }
            }
        }
        total
    };
    trace
        .info(Stage::Wgsl, "two-dispatch rewrite applied")
        .subject("two_dispatch")
        .field("call_sites", call_sites)
        .field("expected_call_sites", 2_i64)
        .field("matches", call_sites == 2)
        .emit();
    if call_sites != 2 {
        trace
            .error(
                Stage::Wgsl,
                "two-dispatch inserted the wrong number of calls",
            )
            .subject("two_dispatch")
            .field("call_sites", call_sites)
            .emit();
    }

    Ok(RewriteOutcome {
        loops_removed: 2,
        fused: false,
    })
}


pub(crate) fn lower_loop_fields(
    body: &waffle::FunctionBody,
    natural_loop: &heterowasm_cfg::NaturalLoop,
    trace: &Trace,
) -> Result<Vec<usize>, String> {
    let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
    let recovery = AddressRecovery::analyze(body, &evolution, trace);
    let accesses: Vec<_> = recovery
        .accesses()
        .iter()
        .filter(|access| natural_loop.blocks.contains(&access.block))
        .cloned()
        .collect();
    let extent = LoopExtent::analyze(body, natural_loop, &evolution);
    let kernel: Kernel = lower(body, natural_loop, &accesses, &evolution, extent, 64, trace)
        .map_err(|error| error.to_string())?;
    header_parameter_indices(body, natural_loop.header, &kernel.fields)
}
