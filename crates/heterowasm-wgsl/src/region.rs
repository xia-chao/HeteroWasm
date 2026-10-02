use heterowasm_cfg::ControlFlow;
use waffle::entity::EntityRef;
use waffle::{Block, BlockTarget, Func, Module, Terminator, Value, ValueDef};

use heterowasm_scev::canonical_value;

use crate::ir::{remap_terminator_arguments, targets_block};


#[derive(Clone, Copy)]
pub(crate) struct RuntimeTrip {
    pub(crate) start_field: Option<usize>,
    pub(crate) limit_field: usize,
    pub(crate) stride_bytes: u32,
    pub(crate) mask: u32,
}


pub(crate) struct RegionCalls<'a> {
    pub(crate) header_params: &'a [usize],
    pub(crate) explicit: Option<&'a [Value]>,
    pub(crate) extra: &'a [Vec<Value>],

    pub(crate) selector: u32,
    pub(crate) extra_selectors: &'a [u32],
    pub(crate) trip: Option<RuntimeTrip>,
}


pub(crate) fn offload_region_calls(
    module: &mut Module<'_>,
    func: Func,
    header: Block,
    exit: Block,
    members: &[Block],
    calls: RegionCalls<'_>,
) -> Result<(), String> {

    let imported = module
        .imports
        .iter()
        .find_map(|import| match &import.kind {
            waffle::ImportKind::Func(func)
                if import.module == "heterowasm" && import.name == "__hw_dispatch" =>
            {
                Some(*func)
            }
            _ => None,
        })
        .ok_or("module does not declare `heterowasm.__hw_dispatch` import")?;


    let signature = module.funcs[imported].sig();
    let param_count = module.signatures[signature].params.len();
    let return_count = module.signatures[signature].returns.len();

    let body = module.funcs[func].body_mut().ok_or("not a function body")?;


    let entries: Vec<Block> = body
        .blocks
        .iter()
        .filter(|block| *block != header && !members.contains(block))
        .filter(|block| {
            body.blocks
                .get(*block)
                .is_some_and(|definition| targets_block(&definition.terminator, header))
        })
        .collect();
    if entries.len() != 1 {
        return Err(format!(
            "loop header has {} external entries; auto-rewrite does not support this yet",
            entries.len()
        ));
    }
    let preheader = entries[0];


    let target = {
        let definition = body
            .blocks
            .get(preheader)
            .ok_or("predecessor block does not exist")?;
        match &definition.terminator {
            Terminator::Br { target } if target.block == header => target.clone(),
            Terminator::CondBr {
                if_true, if_false, ..
            } => {
                if if_true.block == header {
                    if_true.clone()
                } else {
                    if_false.clone()
                }
            }
            _ => return Err("predecessor terminator does not target the loop header".to_string()),
        }
    };
    let mut arguments: Vec<Value> = match calls.explicit {
        Some(args) => args.to_vec(),
        None => calls
            .header_params
            .iter()
            .map(|index| {
                target.args.get(*index).copied().ok_or_else(|| {
                    format!("predecessor does not pass a value for loop-header parameter {index}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    let field_slots = param_count.saturating_sub(1);
    if arguments.len() > field_slots {
        return Err("loop passes more arguments than the dispatch import".to_string());
    }
    while arguments.len() < field_slots {
        let zero = body.add_op(
            preheader,
            waffle::Operator::I32Const { value: 0 },
            &[],
            &[waffle::Type::I32],
        );
        arguments.push(zero);
    }
    let selector = body.add_op(
        preheader,
        waffle::Operator::I32Const {
            value: calls.selector,
        },
        &[],
        &[waffle::Type::I32],
    );
    arguments.push(selector);


    let header_args = target.args.clone();


    let member_set: std::collections::HashSet<Block> = members.iter().copied().collect();
    let exit_params_before = body
        .blocks
        .get(exit)
        .ok_or("exit block does not exist")?
        .params
        .iter()
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    let edge_args = crate::live_out::exit_edge_arguments(body, &member_set, exit)?;
    if edge_args.len() != exit_params_before.len() {
        return Err(format!(
            "exit block has {} params but the loop edge passes {} values",
            exit_params_before.len(),
            edge_args.len()
        ));
    }


    let live_outs: Vec<Value> = crate::live_out::live_outs(body, &member_set);


    let mut exit_args: Vec<Value> = vec![Value::invalid(); exit_params_before.len()];
    let mut from_results: Vec<usize> = Vec::new();

    let mut cpu_exit_tail: Vec<Value> = Vec::new();
    let mut replacements: Vec<(Value, Value)> = Vec::new();
    let mut edge_filled = vec![false; exit_params_before.len()];

    for value in &live_outs {
        let canonical = canonical_value(body, *value);
        let slot = edge_args
            .iter()
            .position(|arg| canonical_value(body, *arg) == canonical);
        let pass = crate::live_out::is_pass_through(body, header, &member_set, *value);
        let source = if pass {
            let Some(ValueDef::BlockParam(_, index, _)) = body.values.get(canonical) else {
                return Err(
                    "pass-through live-out is not a block parameter after classification"
                        .to_string(),
                );
            };
            Some(header_args.get(*index as usize).copied().ok_or_else(|| {
                format!("loop-header parameter {index} has no argument on the predecessor side; cannot supply live-out")
            })?)
        } else {
            None
        };
        if let Some(slot) = slot {
            if let Some(source) = source {
                exit_args[slot] = source;
            } else {
                from_results.push(slot);
            }
            edge_filled[slot] = true;
            replacements.push((*value, exit_params_before[slot]));
        } else {
            let ty = match body.values.get(canonical) {
                Some(ValueDef::BlockParam(_, _, ty)) => *ty,
                _ => waffle::Type::I32,
            };
            let new_param = body.add_blockparam(exit, ty);

            cpu_exit_tail.push(canonical);
            let slot = exit_args.len();
            if let Some(source) = source {
                exit_args.push(source);
            } else {
                from_results.push(slot);
                exit_args.push(Value::invalid());
            }
            replacements.push((*value, new_param));
        }
    }
    for (slot, arg) in edge_args.iter().enumerate() {
        if edge_filled[slot] {
            continue;
        }
        let canonical = canonical_value(body, *arg);
        let defined_inside = member_set.iter().any(|block| {
            body.blocks.get(*block).is_some_and(|definition| {
                definition
                    .params
                    .iter()
                    .any(|(_, value)| *value == canonical)
                    || definition.insts.contains(&canonical)
            })
        });
        if defined_inside {
            return Err(format!(
                "exit-edge value {arg:?} is defined inside the loop but was not classified as a live-out"
            ));
        }
        exit_args[slot] = *arg;
    }


    let flow = ControlFlow::analyze(body, &heterowasm_trace::Trace::silent());

    for block in body.blocks.iter().collect::<Vec<_>>() {
        if members.contains(&block) || !flow.dominates(exit, block) {
            continue;
        }

        let count = body
            .blocks
            .get(block)
            .map_or(0, |definition| definition.insts.len());
        let substitute = |value: Value, replacements: &[(Value, Value)]| {
            replacements
                .iter()
                .find(|(from, _)| *from == value)
                .map_or(value, |(_, to)| *to)
        };
        for position in 0..count {
            let Some(inst) = body
                .blocks
                .get(block)
                .and_then(|definition| definition.insts.get(position).copied())
            else {
                continue;
            };
            let Some(definition) = body.values.get(inst).cloned() else {
                continue;
            };
            match definition {
                ValueDef::Operator(operator, args, tys) => {
                    let operands: Vec<Value> = body.arg_pool[args].to_vec();
                    let mapped: Vec<Value> = operands
                        .iter()
                        .map(|operand| substitute(*operand, &replacements))
                        .collect();
                    body.values[inst] = ValueDef::Operator(
                        operator,
                        body.arg_pool.from_iter(mapped.into_iter()),
                        tys,
                    );
                }

                ValueDef::Alias(inner) => {
                    body.values[inst] = ValueDef::Alias(substitute(inner, &replacements));
                }
                ValueDef::PickOutput(inner, index, ty) => {
                    body.values[inst] =
                        ValueDef::PickOutput(substitute(inner, &replacements), index, ty);
                }
                _ => {}
            }
        }

        let mut terminator = body
            .blocks
            .get_mut(block)
            .map(|definition| std::mem::replace(&mut definition.terminator, Terminator::None))
            .unwrap_or(Terminator::None);
        remap_terminator_arguments(&mut terminator, &replacements);
        body.blocks
            .get_mut(block)
            .ok_or("block does not exist")?
            .terminator = terminator;
    }


    let entry = body
        .blocks
        .get(preheader)
        .map(|definition| definition.terminator.clone());
    let exit_edges: Vec<Block> = members
        .iter()
        .copied()
        .filter(|block| {
            body.blocks
                .get(*block)
                .is_some_and(|definition| targets_block(&definition.terminator, exit))
        })
        .collect();
    let tail_in_exit_block = cpu_exit_tail.is_empty()
        || (exit_edges.len() == 1
            && cpu_exit_tail.iter().all(|value| {
                let home = exit_edges[0];
                match body.values.get(*value) {
                    Some(ValueDef::BlockParam(block, _, _)) => *block == home,
                    Some(ValueDef::Operator(..) | ValueDef::PickOutput(..)) => body
                        .blocks
                        .get(home)
                        .is_some_and(|definition| definition.insts.contains(value)),
                    _ => false,
                }
            }));
    let call_block = if let (Some(trip), Some(Terminator::Br { target })) = (calls.trip, entry) {
        if target.block == header
            && tail_in_exit_block
            && trip.limit_field < field_slots
            && trip
                .start_field
                .map(|slot| slot < field_slots)
                .unwrap_or(true)
            && trip.stride_bytes > 0
        {
            let limit_value = arguments[trip.limit_field];
            let masked = if trip.mask == u32::MAX {
                limit_value
            } else {
                let mask_bits = body.add_op(
                    preheader,
                    waffle::Operator::I32Const { value: trip.mask },
                    &[],
                    &[waffle::Type::I32],
                );
                body.add_op(
                    preheader,
                    waffle::Operator::I32And,
                    &[limit_value, mask_bits],
                    &[waffle::Type::I32],
                )
            };
            let trips = if let Some(start) = trip.start_field {
                let delta = body.add_op(
                    preheader,
                    waffle::Operator::I32Sub,
                    &[masked, arguments[start]],
                    &[waffle::Type::I32],
                );
                let step = body.add_op(
                    preheader,
                    waffle::Operator::I32Const {
                        value: trip.stride_bytes,
                    },
                    &[],
                    &[waffle::Type::I32],
                );
                body.add_op(
                    preheader,
                    waffle::Operator::I32DivU,
                    &[delta, step],
                    &[waffle::Type::I32],
                )
            } else {
                masked
            };
            let floor = body.add_op(
                preheader,
                waffle::Operator::I32Const {
                    value: crate::Dispatch::TRIPS_SLOWER_THAN_CPU_BELOW,
                },
                &[],
                &[waffle::Type::I32],
            );
            let take_gpu = body.add_op(
                preheader,
                waffle::Operator::I32GeU,
                &[trips, floor],
                &[waffle::Type::I32],
            );
            let gpu_block = body.add_block();
            body.blocks
                .get_mut(preheader)
                .ok_or("predecessor block does not exist")?
                .terminator = Terminator::CondBr {
                cond: take_gpu,
                if_true: BlockTarget {
                    block: gpu_block,
                    args: Vec::new(),
                },
                if_false: BlockTarget {
                    block: header,
                    args: target.args,
                },
            };
            gpu_block
        } else {
            preheader
        }
    } else {
        preheader
    };


    if from_results.len() > return_count {
        return Err("loop returns more values than the dispatch import".to_string());
    }
    let result_types = vec![waffle::Type::I32; return_count];
    let call = body.add_op(
        call_block,
        waffle::Operator::Call {
            function_index: imported,
        },
        &arguments,
        &result_types,
    );

    for k in 0..return_count {
        let picked = body.add_value(ValueDef::PickOutput(
            call,
            u32::try_from(k).map_err(|_| "return-value count exceeds u32")?,
            waffle::Type::I32,
        ));

        body.append_to_block(call_block, picked);
        if let Some(position) = from_results.get(k) {
            exit_args[*position] = picked;
        }
    }

    for (ordinal, extra) in calls.extra.iter().enumerate() {
        let mut extra_args = extra.clone();
        while extra_args.len() < field_slots {
            let zero = body.add_op(
                call_block,
                waffle::Operator::I32Const { value: 0 },
                &[],
                &[waffle::Type::I32],
            );
            extra_args.push(zero);
        }
        let selector_bits = calls.extra_selectors.get(ordinal).copied().unwrap_or(0);
        let selector = body.add_op(
            call_block,
            waffle::Operator::I32Const {
                value: selector_bits,
            },
            &[],
            &[waffle::Type::I32],
        );
        extra_args.push(selector);
        body.add_op(
            call_block,
            waffle::Operator::Call {
                function_index: imported,
            },
            &extra_args,
            &[],
        );
    }
    if call_block != preheader {
        for block in members {
            let Some(definition) = body.blocks.get_mut(*block) else {
                continue;
            };
            let extend = |target: &mut BlockTarget| {
                if target.block == exit {
                    target.args.extend_from_slice(&cpu_exit_tail);
                }
            };
            match &mut definition.terminator {
                Terminator::Br { target } => extend(target),
                Terminator::CondBr {
                    if_true, if_false, ..
                } => {
                    extend(if_true);
                    extend(if_false);
                }
                _ => {}
            }
        }
    }
    body.blocks
        .get_mut(call_block)
        .ok_or("dispatch block does not exist")?
        .terminator = Terminator::Br {
        target: BlockTarget {
            block: exit,
            args: exit_args,
        },
    };

    body.recompute_edges();
    Ok(())
}


pub(crate) fn region_preheader_arguments(
    module: &Module<'_>,
    func: Func,
    header: Block,
    members: &[Block],
) -> Result<Vec<Value>, String> {
    let body = module.funcs[func].body().ok_or("not a function body")?;
    let entries: Vec<Block> = body
        .blocks
        .iter()
        .filter(|block| *block != header && !members.contains(block))
        .filter(|block| {
            body.blocks
                .get(*block)
                .is_some_and(|definition| targets_block(&definition.terminator, header))
        })
        .collect();
    if entries.len() != 1 {
        return Err(format!(
            "region entry has {} external predecessors",
            entries.len()
        ));
    }
    let definition = body
        .blocks
        .get(entries[0])
        .ok_or("predecessor block does not exist")?;
    match &definition.terminator {
        Terminator::Br { target } if target.block == header => Ok(target.args.clone()),
        Terminator::CondBr {
            if_true, if_false, ..
        } => {
            if if_true.block == header {
                Ok(if_true.args.clone())
            } else {
                Ok(if_false.args.clone())
            }
        }
        _ => Err("predecessor terminator does not target the region entry".to_string()),
    }
}
