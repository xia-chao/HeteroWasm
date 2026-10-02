use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody, Operator, Value, ValueDef};

use crate::canonical::{canonical_value, constant_value};
use crate::invariance::LoopInvariance;
use crate::memory_recurrence::find_memory_recurrences;
use crate::types::{InductionVariable, ScalarEvolution};

impl ScalarEvolution {

    pub fn analyze(body: &FunctionBody, natural_loop: &NaturalLoop, trace: &Trace) -> Self {
        let _scope = trace.stage(Stage::ScalarEvolution, "loop");
        let Some(header_def) = body.blocks.get(natural_loop.header) else {
            return Self::default();
        };
        let in_loop: HashSet<Block> = natural_loop.blocks.iter().copied().collect();

        let mut induction = Vec::new();
        let mut invariant_params: HashSet<Value> = HashSet::new();
        let mut carried_non_affine: Vec<Value> = Vec::new();
        for (position, (_, parameter)) in header_def.params.iter().enumerate() {
            let parameter = *parameter;
            let mut step: Option<i64> = None;
            let mut initial: Option<Value> = None;
            let mut not_affine = false;
            let mut inner_predecessors = 0_usize;
            let mut inner_update: Option<Value> = None;

            for &pred in &header_def.preds {
                let Some(args) = LoopInvariance::branch_args_to(body, pred, natural_loop.header)
                else {
                    continue;
                };
                let Some(&argument) = args.get(position) else {
                    continue;
                };

                if in_loop.contains(&pred) {
                    inner_predecessors += 1;
                    if inner_update.is_none() {
                        inner_update = Some(argument);
                    }
                    match delta_from_add(body, argument, parameter) {

                        Some(0) => {}
                        Some(step_value) => step = Some(step_value),
                        None => not_affine = true,
                    }
                } else if argument != parameter {
                    initial = Some(argument);
                }
            }


            if inner_predecessors > 0 && !not_affine && step.is_none() {
                invariant_params.insert(parameter);
            }


            if inner_predecessors > 0 && not_affine {
                carried_non_affine.push(parameter);
            }

            let accepted = !not_affine && step.is_some() && initial.is_some();


            let mut event = trace
                .debug(
                    Stage::ScalarEvolution,
                    "loop-header block-parameter decision",
                )
                .field("position", position)
                .field("parameter", parameter.index())
                .field("inner_predecessors", inner_predecessors)
                .field("has_step", step.is_some())
                .field("has_initial", initial.is_some())
                .field("not_affine_update", not_affine);
            if let Some(update) = inner_update {
                event = event
                    .field("update", update.index())
                    .field("update_is_parameter", update == parameter);
            }
            if let Some(step_value) = step {
                event = event.field("step", step_value);
            }
            event.field("accepted", accepted).emit();

            if accepted {
                if let (Some(step_value), Some(initial_value)) = (step, initial) {
                    induction.push(InductionVariable {
                        value: parameter,
                        initial: initial_value,
                        step: step_value,
                    });
                }
            }
        }


        let invariance = LoopInvariance::new(body, natural_loop, &invariant_params);
        carried_non_affine.retain(|parameter| !invariance.value_is_invariant(*parameter));


        let memory_recurrences = if induction.is_empty() {
            find_memory_recurrences(body, natural_loop, &invariant_params)
        } else {
            Vec::new()
        };
        let recovered_from_memory = memory_recurrences.len();
        if recovered_from_memory > 0 {
            trace
                .info(
                    Stage::ScalarEvolution,
                    "recovered a recurrence from a memory stack slot",
                )
                .field("header", natural_loop.header.index())
                .field("recovered", recovered_from_memory)
                .emit();
        }
        induction.extend(memory_recurrences);

        let recurrence_in_memory = induction.is_empty() && !header_def.params.is_empty();
        if recurrence_in_memory {
            trace
                .warn(
                    Stage::ScalarEvolution,
                    "recurrence stays in a memory stack slot (usually a debug artifact); outside Phase 0 committed coverage",
                )
                .field("header", natural_loop.header.index())
                .field("parameters", header_def.params.len())
                .field("outside_coverage", true)
                .emit();
        }

        trace
            .info(
                Stage::ScalarEvolution,
                "induction variable recognition complete",
            )
            .field("induction", induction.len())
            .field("recovered_from_memory", recovered_from_memory)
            .field("header", natural_loop.header.index())
            .field("outside_coverage", recurrence_in_memory)
            .emit();

        if !carried_non_affine.is_empty() {
            trace
                .warn(
                    Stage::ScalarEvolution,
                    "loop-carried scalar with non-constant update — iterations are not independent; must not parallelize",
                )
                .field("header", natural_loop.header.index())
                .field("carried", carried_non_affine.len())
                .emit();
        }

        Self {
            induction,
            recurrence_in_memory,
            recovered_from_memory,
            invariant_params,
            carried_non_affine,
        }
    }
}


fn delta_from_add(body: &FunctionBody, value: Value, parameter: Value) -> Option<i64> {
    let parameter = canonical_value(body, parameter);
    let value = canonical_value(body, value);
    if value == parameter {
        return Some(0);
    }
    let ValueDef::Operator(operator, args, _) = body.values.get(value)? else {
        return None;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    let left = canonical_value(body, *operands.first()?);
    let right = canonical_value(body, *operands.get(1)?);
    match operator {
        Operator::I32Add => {
            if left == parameter {
                constant_value(body, right)
            } else if right == parameter {
                constant_value(body, left)
            } else {
                None
            }
        }
        Operator::I32Sub if left == parameter => {
            constant_value(body, right).map(|constant| constant.saturating_neg())
        }
        _ => None,
    }
}
