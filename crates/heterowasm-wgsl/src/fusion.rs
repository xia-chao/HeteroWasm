use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

use heterowasm_address::{AccessKind, AddressRecovery, NormalizedAddress};
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::{ControlFlow, NaturalLoop};
use heterowasm_scev::{canonical_value, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::{Block, FunctionBody, Module, Terminator, Value};

use crate::conformance;
use crate::lower::lower;
use crate::types::{GpuFusionPlan, Kernel};


pub(crate) type RegionUses = BTreeMap<String, Vec<(usize, bool)>>;


pub(crate) type Collected = (NaturalLoop, Option<LoopExtent>, RegionUses);

fn region_key(body: &FunctionBody, address: &NormalizedAddress) -> String {
    let terms: Vec<String> = address
        .base
        .iter()
        .map(|(value, coefficient)| {

            match conformance::field_mapping(body, &[*value]) {
                Some(parameters) => match parameters.first() {
                    Some(parameter) => format!("param{parameter}*{coefficient}"),
                    None => format!("{value:?}*{coefficient}"),
                },
                None => {
                    let canonical = canonical_value(body, *value);
                    format!("{canonical:?}*{coefficient}")
                }
            }
        })
        .collect();
    format!("[{}]@{}", terms.join("+"), address.stride)
}

pub(crate) fn collect_region_uses(body: &FunctionBody, trace: &Trace) -> Vec<Collected> {
    let mut result = Vec::new();
    for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
        let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
        let recovery = AddressRecovery::analyze(body, &evolution, trace);
        let mut uses: RegionUses = BTreeMap::new();
        for access in recovery.accesses() {
            if !natural_loop.blocks.contains(&access.block) {
                continue;
            }
            let Some(normalized) = access.normalized(&evolution) else {
                continue;
            };
            uses.entry(region_key(body, &normalized))
                .or_default()
                .push((result.len(), access.kind == AccessKind::Store));
        }
        let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
        result.push((natural_loop, extent, uses));
    }
    result
}


pub(crate) fn exclusive_intermediates(loops: &[Collected]) -> Vec<(String, usize, usize)> {
    let mut merged: BTreeMap<String, Vec<(usize, bool)>> = BTreeMap::new();
    for (index, (_, _, uses)) in loops.iter().enumerate() {
        for (region, accesses) in uses {
            for (_, is_store) in accesses {
                merged
                    .entry(region.clone())
                    .or_default()
                    .push((index, *is_store));
            }
        }
    }
    let mut found = Vec::new();
    for (region, accesses) in merged {
        let stores: Vec<usize> = accesses
            .iter()
            .filter(|(_, is_store)| *is_store)
            .map(|(index, _)| *index)
            .collect();
        let loads: Vec<usize> = accesses
            .iter()
            .filter(|(_, is_store)| !*is_store)
            .map(|(index, _)| *index)
            .collect();
        if stores.len() == 1 && loads.len() == 1 && stores[0] != loads[0] {
            found.push((region, stores[0], loads[0]));
        }
    }
    found.sort();
    found
}


pub(crate) fn same_iteration_space(
    body: &FunctionBody,
    left: &Option<LoopExtent>,
    right: &Option<LoopExtent>,
) -> bool {
    let (Some(left), Some(right)) = (left, right) else {

        return false;
    };
    let same_start = left.start == right.start;
    let same_limit = match (left.limit, right.limit) {
        (Some(a), Some(b)) => a == b,
        (None, None) => match (left.limit_value, right.limit_value) {
            (Some(a), Some(b)) => {
                if a == b {
                    true
                } else {
                    match (
                        conformance::field_mapping(body, &[a]),
                        conformance::field_mapping(body, &[b]),
                    ) {
                        (Some(left), Some(right)) => left == right,
                        _ => false,
                    }
                }
            }
            _ => false,
        },
        _ => false,
    };
    same_start && same_limit
}


pub(crate) fn fusable_pairs(
    body: &FunctionBody,
    loops: &[Collected],
) -> Vec<(String, usize, usize)> {
    let candidates = exclusive_intermediates(loops);

    let edges: Vec<(usize, usize)> = candidates
        .iter()
        .map(|(_, producer, consumer)| (*producer, *consumer))
        .collect();
    let reaches = |from: usize, to: usize| -> bool {
        let mut stack = vec![from];
        let mut seen = HashSet::new();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            for (a, b) in &edges {
                if *a == node {
                    if *b == to {
                        return true;
                    }
                    stack.push(*b);
                }
            }
        }
        false
    };

    candidates
        .into_iter()
        .filter(|(_, producer, consumer)| {
            let (_, left, _) = &loops[*producer];
            let (_, right, _) = &loops[*consumer];
            same_iteration_space(body, left, right)

                    && !reaches(*consumer, *producer)
        })
        .collect()
}


pub(crate) fn region_span(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
) -> Result<(Block, Block, Vec<Block>), String> {
    let exit = natural_loop
        .blocks
        .iter()
        .filter_map(|block| body.blocks.get(*block))
        .flat_map(|definition| match &definition.terminator {
            Terminator::Br { target } => vec![target.block],
            Terminator::CondBr {
                if_true, if_false, ..
            } => vec![if_true.block, if_false.block],
            _ => vec![],
        })
        .find(|block| !natural_loop.blocks.contains(block))
        .ok_or("natural loop has no exit block")?;
    Ok((natural_loop.header, exit, natural_loop.blocks.clone()))
}


pub(crate) fn resolve_at_entry(body: &FunctionBody, value: Value) -> Result<Value, String> {
    let parameter = conformance::field_mapping(body, &[value])
        .and_then(|mapping| mapping.first().copied())
        .ok_or("value does not trace back to a function parameter — fusion precondition unmet")?;
    body.blocks
        .get(body.entry)
        .and_then(|entry| entry.params.get(parameter as usize))
        .map(|(_, value)| *value)
        .ok_or_else(|| format!("function entry has no parameter {parameter}"))
}


pub fn plan_gpu_fusion(
    module: &Module<'_>,
    trace: &Trace,
) -> Result<Option<GpuFusionPlan>, String> {
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        let collected = collect_region_uses(body, trace);
        let Some((region, producer_index, consumer_index)) =
            fusable_pairs(body, &collected).into_iter().next()
        else {
            continue;
        };

        let lower_one = |index: usize| -> Result<Kernel, String> {
            let (natural_loop, _, _) = &collected[index];
            let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
            let recovery = AddressRecovery::analyze(body, &evolution, trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            lower(body, natural_loop, &accesses, &evolution, extent, 64, trace)
                .map_err(|error| error.to_string())
        };
        let producer = lower_one(producer_index)?;
        let consumer = lower_one(consumer_index)?;
        let plan = fuse_two_chain(&producer, &consumer, producer.fields.len(), &region)?;
        trace
            .info(Stage::Wgsl, "gpu fusion planned")
            .subject("gpu_fusion")
            .field("removed_intermediate", plan.removed_intermediate.clone())
            .field("producer_fields", producer.fields.len())
            .field("consumer_fields", consumer.fields.len())
            .emit();
        return Ok(Some(GpuFusionPlan {
            kernel: plan.fused,
            func,
        }));
    }
    Ok(None)
}

pub(crate) struct FusionPlan {
    pub(crate) fused: Kernel,

    pub(crate) removed_intermediate: String,
}

fn split_store(source: &str) -> Result<(String, String), String> {
    let line = source
        .lines()
        .find(|line| line.trim_start().starts_with("mem[") && line.contains(" = "))
        .ok_or("emit result has no assignment statement")?;
    let (target, value) = line
        .split_once(" = ")
        .ok_or("assignment statement format changed")?;
    Ok((
        target.trim().to_string(),
        value.trim().trim_end_matches(';').trim().to_string(),
    ))
}

fn renumber(expression: &str, mapping: &[usize]) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = expression;
    while let Some(at) = rest.find("params.p") {
        out.push_str(&rest[..at]);
        let tail = &rest[at + "params.p".len()..];
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        let slot: usize = digits.parse().map_err(|_| "failed to parse field number")?;
        let mapped = mapping.get(slot).copied().ok_or_else(|| {
            format!("field p{slot} has no renumbering plan — fusion precondition unmet")
        })?;
        if mapped == usize::MAX {
            return Err(format!(
                "field p{slot} should not appear here — fusion precondition unmet"
            ));
        }

        write!(out, "params.p{mapped}").map_err(|_| "write failed")?;
        rest = &tail[digits.len()..];
    }
    out.push_str(rest);
    Ok(out)
}


pub(crate) fn fuse_two_chain(
    producer: &Kernel,
    consumer: &Kernel,
    producer_fields: usize,
    removed_intermediate: &str,
) -> Result<FusionPlan, String> {
    let (_, producer_value) = split_store(&producer.source)?;
    let (consumer_target, consumer_value) = split_store(&consumer.source)?;


    let input_count = producer_fields.saturating_sub(2);
    if input_count == 0 {
        return Err("producer has no input base; cannot fuse".to_string());
    }


    const POISON: usize = usize::MAX;
    let mut producer_mapping: Vec<usize> = Vec::with_capacity(producer_fields);
    producer_mapping.push(0);
    producer_mapping.push(POISON);
    for index in 0..input_count {
        producer_mapping.push(2 + index);
    }
    let fused_value = renumber(&producer_value, &producer_mapping)?;


    let consumer_expression = {
        let needle = "mem[params.p2 / 4u + i]";
        if !consumer_value.contains(needle) {
            return Err(format!(
                "consumer values have no load of the intermediate (looking for `{needle}`) — fusion precondition unmet"
            ));
        }
        consumer_value.replacen(needle, &format!("({fused_value})"), 1)
    };


    let source = consumer
        .source
        .replacen(&consumer_target, "mem[params.p1 / 4u + i]", 1);
    let source = source.replacen(&consumer_value, &consumer_expression, 1);


    let fused_fields = 2 + input_count;
    let source = adjust_params_struct(&source, fused_fields)?;

    Ok(FusionPlan {
        fused: Kernel {
            source,
            workgroup_size: consumer.workgroup_size,
            dispatch: consumer.dispatch,

            min_constant_bytes: producer.min_constant_bytes.min(consumer.min_constant_bytes),

            max_constant_bytes: producer.max_constant_bytes.max(consumer.max_constant_bytes),
            max_stride_bytes: producer.max_stride_bytes.max(consumer.max_stride_bytes),
            index_field: consumer.index_field,
            launch_count: consumer.launch_count,

            results: Vec::new(),

            fields: producer.fields.clone(),
        },
        removed_intermediate: removed_intermediate.to_string(),
    })
}

fn adjust_params_struct(source: &str, fields: usize) -> Result<String, String> {
    let start = source
        .find("struct Params {")
        .ok_or("Params struct not found")?;
    let end = source[start..]
        .find('}')
        .map(|offset| start + offset + 1)
        .ok_or("Params struct is not closed")?;
    let mut replacement = String::from("struct Params {\n");
    for index in 0..fields {
        writeln!(replacement, "  p{index} : u32,").map_err(|_| "write failed")?;
    }
    replacement.push('}');
    Ok(format!(
        "{}{}{}",
        &source[..start],
        replacement,
        &source[end..]
    ))
}
