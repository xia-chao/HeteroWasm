use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use waffle::{Module, Value};

use crate::fusion::{collect_region_uses, fusable_pairs, region_span, resolve_at_entry};
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


    let producer_args = region_preheader_arguments(module, func, chain.producer_header, &members)?;
    if producer_args.len() < 2 {
        return Err("producer needs at least two block params: upper bound + one base".to_string());
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


fn lower_fields(
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
