use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody, Operator, Value, ValueDef};

use crate::canonical::{canonical_value, constant_value};
use crate::invariance::LoopInvariance;
use crate::types::InductionVariable;


pub(crate) fn find_memory_recurrences(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
    invariant_params: &HashSet<Value>,
) -> Vec<InductionVariable> {
    let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();
    let mut found: Vec<InductionVariable> = Vec::new();

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
        let operands: &[Value] = &body.arg_pool[*args];
        let (Some(&address), Some(&stored)) = (operands.first(), operands.get(1)) else {
            continue;
        };

        let Some((loaded, load_address, load_operator, step)) = add_of_load(body, stored) else {
            continue;
        };


        if !same_address(body, address, load_address, 8) {
            continue;
        }
        if !same_memory_arg(load_operator, store_operator) {
            continue;
        }
        if !invariance.value_is_invariant(address) {
            continue;
        }

        if store_count_for(body, &members, address) > 1 {
            continue;
        }
        if found.iter().any(|existing| existing.value == loaded) {
            continue;
        }

        found.push(InductionVariable {
            value: loaded,
            initial: recurrence_initial(body, &members, address).unwrap_or(loaded),
            step,
        });
    }

    found
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


fn same_address(body: &FunctionBody, left: Value, right: Value, depth: usize) -> bool {
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
    if std::mem::discriminant(left_operator) != std::mem::discriminant(right_operator) {
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
