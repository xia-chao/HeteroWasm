use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody, Operator, Value, ValueDef};

use crate::canonical::{canonical_value, constant_value};
use crate::invariance::LoopInvariance;
use crate::types::InductionVariable;


#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecurrenceReport {
    pub stores_in_loop: usize,
    pub not_add_of_load: usize,
    pub add_nonconstant: usize,
    pub accumulator_like: usize,
    pub copy_load: usize,
    pub other_operator: usize,
    pub address_mismatch: usize,
    pub memory_mismatch: usize,
    pub address_not_invariant: usize,
    pub duplicate: usize,
    pub multiple_stores: usize,
    pub recovered: usize,
}


pub(crate) fn find_memory_recurrences(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
    invariant_params: &HashSet<Value>,
) -> (Vec<InductionVariable>, Vec<Value>, RecurrenceReport) {
    let mut report = RecurrenceReport::default();
    let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();
    let mut found: Vec<InductionVariable> = Vec::new();

    let mut addresses: Vec<Value> = Vec::new();

    let invariance = LoopInvariance::new(body, natural_loop, invariant_params);

    for value in body.values.iter() {
        let Some(ValueDef::Operator(store_operator @ Operator::I32Store { .. }, args, _)) =
            body.values.get(value)
        else {
            continue;
        };
        if !members.contains(&body.value_blocks[value]) {
            continue;
        }
        report.stores_in_loop += 1;
        let operands: &[Value] = &body.arg_pool[*args];
        let (Some(&address), Some(&stored)) = (operands.first(), operands.get(1)) else {
            continue;
        };

        let Some((loaded, load_address, load_operator, step)) = add_of_load(body, stored) else {
            report.not_add_of_load += 1;
            match body.values.get(canonical_value(body, stored)) {
                Some(ValueDef::Operator(Operator::I32Add, _, _)) => {
                    report.add_nonconstant += 1;
                    if add_of_load_any(body, stored).is_some_and(|(load_address, _)| {
                        same_memory_arg(
                            load_operator_from(body, stored).unwrap_or(store_operator),
                            store_operator,
                        ) && same_address(body, load_address, address, 8)
                            && invariance.value_is_invariant(address)
                            && store_count_for(body, &members, address) == 1
                    }) {
                        report.accumulator_like += 1;
                    }
                }
                Some(ValueDef::Operator(Operator::I32Load { .. }, _, _)) => report.copy_load += 1,
                Some(ValueDef::Operator(_, _, _)) => report.other_operator += 1,
                _ => {}
            }
            continue;
        };


        if !same_address(body, address, load_address, 8) {
            report.address_mismatch += 1;
            continue;
        }
        if !same_memory_arg(load_operator, store_operator) {
            report.memory_mismatch += 1;
            continue;
        }
        if !invariance.value_is_invariant(address) {
            report.address_not_invariant += 1;
            continue;
        }

        if store_count_for(body, &members, address) > 1 {
            report.multiple_stores += 1;
            continue;
        }
        if found.iter().any(|existing| existing.value == loaded) {
            report.duplicate += 1;
            continue;
        }

        found.push(InductionVariable {
            value: loaded,
            initial: recurrence_initial(body, &members, address).unwrap_or(loaded),
            step,
        });
        addresses.push(canonical_value(body, address));
        report.recovered += 1;
    }

    (found, addresses, report)
}


fn store_count_for(body: &FunctionBody, members: &HashSet<Block>, address: Value) -> usize {
    let mut count = 0_usize;
    for value in body.values.iter() {
        let Some(ValueDef::Operator(Operator::I32Store { .. }, args, _)) = body.values.get(value)
        else {
            continue;
        };
        if !members.contains(&body.value_blocks[value]) {
            continue;
        }
        let operands: &[Value] = &body.arg_pool[*args];
        let Some(&stored_address) = operands.first() else {
            continue;
        };
        if same_address(body, stored_address, address, 8) {
            count += 1;
        }
    }
    count
}


fn add_of_load_any(body: &FunctionBody, stored: Value) -> Option<(Value, Value)> {
    let Some(ValueDef::Operator(Operator::I32Add, args, _)) =
        body.values.get(canonical_value(body, stored))
    else {
        return None;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    let left = *operands.first()?;
    let right = *operands.get(1)?;
    for (candidate, other) in [(left, right), (right, left)] {
        let Some(ValueDef::Operator(Operator::I32Load { .. }, load_args, _)) =
            body.values.get(canonical_value(body, candidate))
        else {
            continue;
        };
        let load_operands: &[Value] = &body.arg_pool[*load_args];
        return Some((*load_operands.first()?, other));
    }
    None
}


fn load_operator_from(body: &FunctionBody, stored: Value) -> Option<&Operator> {
    let ValueDef::Operator(Operator::I32Add, args, _) =
        body.values.get(canonical_value(body, stored))?
    else {
        return None;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    for candidate in operands {
        if let Some(ValueDef::Operator(operator @ Operator::I32Load { .. }, _, _)) =
            body.values.get(canonical_value(body, *candidate))
        {
            return Some(operator);
        }
    }
    None
}


fn add_of_load(body: &FunctionBody, stored: Value) -> Option<(Value, Value, &Operator, i64)> {
    let Some(ValueDef::Operator(Operator::I32Add, args, _)) =
        body.values.get(canonical_value(body, stored))
    else {
        return None;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    let left = *operands.first()?;
    let right = *operands.get(1)?;


    let (loaded, step) = match (constant_value(body, left), constant_value(body, right)) {
        (Some(step), None) => (right, step),
        (None, Some(step)) => (left, step),
        _ => return None,
    };

    let canonical_loaded = canonical_value(body, loaded);
    let Some(ValueDef::Operator(load_operator @ Operator::I32Load { .. }, load_args, _)) =
        body.values.get(canonical_loaded)
    else {
        return None;
    };
    let load_operands: &[Value] = &body.arg_pool[*load_args];

    Some((
        canonical_loaded,
        *load_operands.first()?,
        load_operator,
        step,
    ))
}


pub(crate) fn same_address(body: &FunctionBody, left: Value, right: Value, depth: usize) -> bool {
    let left = canonical_value(body, left);
    let right = canonical_value(body, right);
    if left == right {
        return true;
    }
    if depth == 0 {
        return false;
    }

    let (
        Some(ValueDef::Operator(left_operator, left_args, _)),
        Some(ValueDef::Operator(right_operator, right_args, _)),
    ) = (body.values.get(left), body.values.get(right))
    else {
        return false;
    };

    if left_operator != right_operator {
        return false;
    }

    let left_operands: &[Value] = &body.arg_pool[*left_args];
    let right_operands: &[Value] = &body.arg_pool[*right_args];
    left_operands.len() == right_operands.len()
        && left_operands
            .iter()
            .zip(right_operands.iter())
            .all(|(left, right)| same_address(body, *left, *right, depth - 1))
}

fn same_memory_arg(load: &Operator, store: &Operator) -> bool {
    match (load, store) {
        (Operator::I32Load { memory: load }, Operator::I32Store { memory: store }) => {
            load.offset == store.offset && load.memory.index() == store.memory.index()
        }
        _ => false,
    }
}


fn recurrence_initial(
    body: &FunctionBody,
    members: &HashSet<Block>,
    address: Value,
) -> Option<Value> {
    let target = canonical_value(body, address);

    for value in body.values.iter() {
        let Some(ValueDef::Operator(Operator::I32Store { .. }, args, _)) = body.values.get(value)
        else {
            continue;
        };
        if members.contains(&body.value_blocks[value]) {
            continue;
        }
        let operands: &[Value] = &body.arg_pool[*args];
        let Some(&stored_address) = operands.first() else {
            continue;
        };
        if canonical_value(body, stored_address) == target {
            return operands.get(1).copied();
        }
    }

    None
}
