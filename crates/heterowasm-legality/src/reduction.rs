use heterowasm_scev::{canonical_value, ScalarEvolution};
use waffle::{FunctionBody, Operator, Value, ValueDef};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Add,
    And,
    Or,
    Xor,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
    pub accumulator: Value,
    pub update: Value,
    pub op: ReductionOp,
}


pub fn recognize(
    body: &FunctionBody,
    natural_loop: &heterowasm_cfg::NaturalLoop,
    evolution: &ScalarEvolution,
) -> Option<Reduction> {

    let carried = evolution.carried_non_affine();
    if carried.len() != 1 {
        return None;
    }
    let accumulator = carried[0];
    let header = body.blocks.get(natural_loop.header)?;
    let (position, _) = header.params.iter().enumerate().find(|(_, (_, value))| {
        canonical_value(body, *value) == canonical_value(body, accumulator)
    })?;


    let mut update: Option<Value> = None;
    for &latch in &natural_loop.latches {
        let block = body.blocks.get(latch)?;
        let waffle::Terminator::Br { target } = &block.terminator else {
            return None;
        };
        if target.block != natural_loop.header {
            return None;
        }
        let candidate = *target.args.get(position)?;
        if let Some(previous) = update {
            if canonical_value(body, previous) != canonical_value(body, candidate) {
                return None;
            }
        }
        update = Some(candidate);
    }
    let update = update?;
    let definition = body.values.get(canonical_value(body, update))?;
    let ValueDef::Operator(operator, args, _) = definition else {
        return None;
    };
    let op = match operator {
        Operator::I32Add => ReductionOp::Add,
        Operator::I32And => ReductionOp::And,
        Operator::I32Or => ReductionOp::Or,
        Operator::I32Xor => ReductionOp::Xor,
        _ => return None,
    };
    let operands: &[Value] = &body.arg_pool[*args];
    let left = *operands.first()?;
    let right = *operands.get(1)?;
    let is_accumulator =
        |value: Value| canonical_value(body, value) == canonical_value(body, accumulator);
    match (is_accumulator(left), is_accumulator(right)) {
        (true, false) | (false, true) => {}

        _ => return None,
    }
    Some(Reduction {
        accumulator,
        update,
        op,
    })
}
