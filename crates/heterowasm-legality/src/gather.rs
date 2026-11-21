use heterowasm_address::AddressRecovery;
use heterowasm_scev::{canonical_value, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{FunctionBody, Operator, Value, ValueDef};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexBound {

    Unbounded,

    Masked(u32),

    Rem(u32),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gather {

    pub address: Value,

    pub base: Option<Value>,
    pub stride: i64,
    pub offset: i64,

    pub index_value: Value,

    pub index_load: Option<Value>,

    pub index_affine: bool,
    pub bound: IndexBound,
}


pub fn recognize(
    body: &FunctionBody,
    natural_loop: &heterowasm_cfg::NaturalLoop,
    evolution: &ScalarEvolution,
) -> Vec<Gather> {
    let mut found = Vec::new();
    for &block in &natural_loop.blocks {
        let Some(definition) = body.blocks.get(block) else {
            continue;
        };
        for inst in &definition.insts {
            let Some(ValueDef::Operator(Operator::I32Load { memory }, args, _)) =
                body.values.get(*inst)
            else {
                continue;
            };
            let operands: &[Value] = &body.arg_pool[*args];
            let Some(&address) = operands.first() else {
                continue;
            };

            if AddressRecovery::form_of(body, evolution, address).is_some() {
                continue;
            }
            let Some(parsed) = parse(body, evolution, address, memory.offset) else {
                continue;
            };
            let (bound, index_load) = bound_of(body, parsed.index_value);
            let index_affine = index_load.is_some_and(|index_load| {
                let Some(ValueDef::Operator(Operator::I32Load { .. }, index_args, _)) =
                    body.values.get(canonical_value(body, index_load))
                else {
                    return false;
                };
                let index_operands: &[Value] = &body.arg_pool[*index_args];
                index_operands.first().is_some_and(|index_address| {
                    AddressRecovery::form_of(body, evolution, *index_address).is_some()
                })
            });
            found.push(Gather {
                address,
                base: parsed.base,
                stride: parsed.stride,
                offset: parsed.offset,
                index_value: parsed.index_value,
                index_load,
                index_affine,
                bound,
            });
        }
    }
    found
}


pub fn report(
    body: &FunctionBody,
    natural_loop: &heterowasm_cfg::NaturalLoop,
    evolution: &ScalarEvolution,
    trace: &Trace,
) {
    let gathers = recognize(body, natural_loop, evolution);
    let stride_four = gathers.iter().filter(|gather| gather.stride == 4).count();
    let unbounded = gathers
        .iter()
        .filter(|gather| gather.bound == IndexBound::Unbounded)
        .count();
    let masked = gathers
        .iter()
        .filter(|gather| matches!(gather.bound, IndexBound::Masked(_)))
        .count();
    let rem = gathers
        .iter()
        .filter(|gather| matches!(gather.bound, IndexBound::Rem(_)))
        .count();

    let mut index_terms = 0_usize;
    let mut masked_terms = 0_usize;
    let mut scaled_terms = 0_usize;
    let mut pointer_terms = 0_usize;
    for &block in &natural_loop.blocks {
        let Some(definition) = body.blocks.get(block) else {
            continue;
        };
        for inst in &definition.insts {
            let Some(ValueDef::Operator(Operator::I32Load { .. }, args, _)) =
                body.values.get(*inst)
            else {
                continue;
            };
            let operands: &[Value] = &body.arg_pool[*args];
            let Some(&address) = operands.first() else {
                continue;
            };
            if AddressRecovery::form_of(body, evolution, address).is_some() {
                continue;
            }
            let mut visited = std::collections::HashSet::new();
            if let Some((scale, masked)) =
                find_index_load(body, evolution, address, 1, &mut visited, 0)
            {
                index_terms += 1;
                if masked.is_some() {
                    masked_terms += 1;
                }
                if scale.abs() >= 2 {
                    scaled_terms += 1;
                } else {
                    pointer_terms += 1;
                }
            }
        }
    }

    let mut pinned_accesses = 0_usize;
    let mut pinned_terms = 0_usize;
    let mut pinned_distinct: std::collections::HashSet<Value> = std::collections::HashSet::new();
    let mut loop_has_store = false;
    for &block in &natural_loop.blocks {
        let Some(definition) = body.blocks.get(block) else {
            continue;
        };
        for inst in &definition.insts {
            let Some(ValueDef::Operator(operator, args, _)) = body.values.get(*inst) else {
                continue;
            };
            let operands: &[Value] = &body.arg_pool[*args];
            let Some(&address) = operands.first() else {
                continue;
            };
            if matches!(operator, Operator::I32Store { .. }) {
                loop_has_store = true;
            }
            if !matches!(
                operator,
                Operator::I32Load { .. } | Operator::I32Store { .. }
            ) {
                continue;
            }

            let mut visited = std::collections::HashSet::new();
            if let Some(split) = split_address(body, evolution, address, 1, &mut visited, 0) {
                if split.loads.is_empty() {
                    continue;
                }
                pinned_accesses += 1;
                pinned_terms += split.loads.len();
                for load in split.loads {
                    pinned_distinct.insert(canonical_value(body, load));
                }
            }
        }
    }
    if gathers.is_empty() && pinned_accesses == 0 {
        return;
    }
    trace
        .info(Stage::Legality, "gather shape classification")
        .field("pinned_accesses", pinned_accesses)
        .field("pinned_terms", pinned_terms)
        .field("pinned_distinct", pinned_distinct.len())
        .field("pinned_with_stores", usize::from(loop_has_store))
        .subject(format!("loop@{}", natural_loop.header.index()).as_str())
        .field("gather_loads", gathers.len())
        .field("index_load_terms", index_terms)
        .field("masked_index_terms", masked_terms)
        .field("scaled_index_terms", scaled_terms)
        .field("pointer_load_terms", pointer_terms)
        .field("stride_bytes_4", stride_four)
        .field("unbounded", unbounded)
        .field("masked", masked)
        .field("rem", rem)
        .field(
            "index_loads",
            gathers
                .iter()
                .filter(|gather| gather.index_load.is_some())
                .count(),
        )
        .field(
            "index_affine",
            gathers.iter().filter(|gather| gather.index_affine).count(),
        )
        .emit();
}

struct Parsed {
    base: Option<Value>,
    stride: i64,
    offset: i64,
    index_value: Value,
}


fn parse(
    body: &FunctionBody,
    evolution: &ScalarEvolution,
    address: Value,
    instruction_offset: u32,
) -> Option<Parsed> {
    let mut terms: Vec<(Value, i64)> = Vec::new();
    let mut constant = i64::from(instruction_offset);
    flatten(body, address, 1, &mut terms, &mut constant, 0)?;
    let mut base: Option<Value> = None;
    let mut stride: i64 = 0;
    let mut index_value: Option<Value> = None;
    for (term, sign) in terms {
        if let Some(form) = evolution.affine_of(body, term) {
            if !form.induction.is_empty() {
                return None;
            }
            if form.offset != 0 || form.invariants.len() != 1 {
                return None;
            }
            let (value, coefficient) = form.invariants[0];
            if coefficient != sign {
                return None;
            }
            base = Some(value);
            continue;
        }

        let (stride_here, index) = index_term(body, term)?;
        if index_value.is_some() && stride != 0 {
            return None;
        }
        stride = sign.checked_mul(stride_here)?;
        index_value = Some(index);
    }
    Some(Parsed {
        base,
        stride,
        offset: constant,
        index_value: index_value?,
    })
}


fn index_term(body: &FunctionBody, value: Value) -> Option<(i64, Value)> {
    match body.values.get(canonical_value(body, value))? {
        ValueDef::Operator(Operator::I32Mul, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first()?;
            let right = *operands.get(1)?;
            if let Some(constant) = constant_of(body, right) {
                return Some((constant, left));
            }
            if let Some(constant) = constant_of(body, left) {
                return Some((constant, right));
            }
            None
        }
        ValueDef::Operator(Operator::I32Shl, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first()?;
            let shift = constant_of(body, *operands.get(1)?)?;
            let shift = u32::try_from(shift).ok()?;
            Some((1_i64.checked_shl(shift)?, left))
        }
        _ => Some((1, value)),
    }
}

fn flatten(
    body: &FunctionBody,
    value: Value,
    sign: i64,
    terms: &mut Vec<(Value, i64)>,
    constant: &mut i64,
    depth: u32,
) -> Option<()> {
    if depth > 16 {
        return None;
    }
    match body.values.get(canonical_value(body, value))? {
        ValueDef::Operator(Operator::I32Add, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            flatten(body, *operands.first()?, sign, terms, constant, depth + 1)?;
            flatten(body, *operands.get(1)?, sign, terms, constant, depth + 1)?;
        }
        ValueDef::Operator(Operator::I32Sub, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            flatten(body, *operands.first()?, sign, terms, constant, depth + 1)?;
            flatten(
                body,
                *operands.get(1)?,
                sign.checked_neg()?,
                terms,
                constant,
                depth + 1,
            )?;
        }
        _ => {
            if let Some(value) = constant_of(body, value) {
                *constant = constant.checked_add(sign.checked_mul(value)?)?;
            } else {
                terms.push((value, sign));
            }
        }
    }
    Some(())
}

fn constant_of(body: &FunctionBody, value: Value) -> Option<i64> {
    let ValueDef::Operator(Operator::I32Const { value }, _, _) =
        body.values.get(canonical_value(body, value))?
    else {
        return None;
    };
    Some(i64::from(*value))
}


fn bound_of(body: &FunctionBody, value: Value) -> (IndexBound, Option<Value>) {
    let Some(definition) = body.values.get(canonical_value(body, value)) else {
        return (IndexBound::Unbounded, None);
    };
    let (wrapping, args) = match definition {
        ValueDef::Operator(Operator::I32And, args, _) => (true, args),
        ValueDef::Operator(Operator::I32RemU, args, _) => (false, args),
        ValueDef::Operator(Operator::I32Load { .. }, _, _) => {
            return (IndexBound::Unbounded, Some(value));
        }
        _ => return (IndexBound::Unbounded, None),
    };
    let operands: &[Value] = &body.arg_pool[*args];
    let Some(left) = operands.first() else {
        return (IndexBound::Unbounded, None);
    };
    let (inner, load) = bound_of(body, *left);
    if inner != IndexBound::Unbounded {
        return (inner, load);
    }
    let Some(constant) = operands.get(1).and_then(|right| constant_of(body, *right)) else {
        return (IndexBound::Unbounded, load);
    };
    let bound = if wrapping {
        IndexBound::Masked(u32::try_from(constant).unwrap_or(u32::MAX))
    } else {
        IndexBound::Rem(u32::try_from(constant).unwrap_or(0))
    };
    (bound, load)
}


fn find_index_load(
    body: &FunctionBody,
    evolution: &ScalarEvolution,
    value: Value,
    scale: i64,
    visited: &mut std::collections::HashSet<Value>,
    depth: u32,
) -> Option<(i64, Option<u32>)> {
    if depth > 16 || !visited.insert(value) {
        return None;
    }
    let definition = body.values.get(canonical_value(body, value))?;
    match definition {
        ValueDef::Operator(Operator::I32Load { .. }, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let address = operands.first()?;
            if AddressRecovery::form_of(body, evolution, *address).is_some() {
                return Some((scale, None));
            }
            None
        }
        ValueDef::Operator(Operator::I32And, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = operands.first()?;
            let mask = operands
                .get(1)
                .and_then(|right| constant_of(body, *right))
                .and_then(|mask| u32::try_from(mask).ok());
            let found = find_index_load(body, evolution, *left, scale, visited, depth + 1)?;
            Some((found.0, mask.or(found.1)))
        }
        ValueDef::Operator(Operator::I32Mul, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first()?;
            let right = *operands.get(1)?;
            if let Some(factor) = constant_of(body, right) {
                return Some((
                    scale.checked_mul(factor)?,
                    find_index_load(
                        body,
                        evolution,
                        left,
                        scale.checked_mul(factor)?,
                        visited,
                        depth + 1,
                    )?
                    .1,
                ));
            }
            if let Some(factor) = constant_of(body, left) {
                return Some((
                    scale.checked_mul(factor)?,
                    find_index_load(
                        body,
                        evolution,
                        right,
                        scale.checked_mul(factor)?,
                        visited,
                        depth + 1,
                    )?
                    .1,
                ));
            }
            None
        }
        ValueDef::Operator(Operator::I32Shl, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first()?;
            let shift = u32::try_from(constant_of(body, *operands.get(1)?)?).ok()?;
            let factor = 1_i64.checked_shl(shift)?;
            Some((
                scale.checked_mul(factor)?,
                find_index_load(
                    body,
                    evolution,
                    left,
                    scale.checked_mul(factor)?,
                    visited,
                    depth + 1,
                )?
                .1,
            ))
        }
        ValueDef::Operator(_, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            for operand in operands {
                if let Some(found) =
                    find_index_load(body, evolution, *operand, scale, visited, depth + 1)
                {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}


struct Split {
    loads: Vec<Value>,
    induction: bool,
}


fn split_address(
    body: &FunctionBody,
    evolution: &ScalarEvolution,
    value: Value,
    scale: i64,
    visited: &mut std::collections::HashSet<Value>,
    depth: u32,
) -> Option<Split> {
    if depth > 24 || !visited.insert(value) {
        return None;
    }
    let definition = body.values.get(canonical_value(body, value))?;
    match definition {
        ValueDef::Operator(Operator::I32Const { .. }, _, _) => Some(Split {
            loads: vec![],
            induction: false,
        }),
        ValueDef::Operator(Operator::I32Add, args, _)
        | ValueDef::Operator(Operator::I32Sub, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let flip = matches!(definition, ValueDef::Operator(Operator::I32Sub, _, _));
            let mut loads = Vec::new();
            let mut induction = false;
            for (index, operand) in operands.iter().enumerate() {
                let sign = if flip && index == 1 { -1 } else { 1 };
                let split = split_address(
                    body,
                    evolution,
                    *operand,
                    scale.checked_mul(sign)?,
                    visited,
                    depth + 1,
                )?;
                loads.extend(split.loads);
                induction |= split.induction;
            }
            Some(Split { loads, induction })
        }
        ValueDef::Operator(Operator::I32Mul, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first()?;
            let right = *operands.get(1)?;
            let (constant, other) = match (constant_of(body, right), constant_of(body, left)) {
                (Some(constant), _) => (constant, left),
                (None, Some(constant)) => (constant, right),
                (None, None) => return None,
            };
            split_address(
                body,
                evolution,
                other,
                scale.checked_mul(constant)?,
                visited,
                depth + 1,
            )
        }
        ValueDef::Operator(Operator::I32Shl, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let shift = u32::try_from(constant_of(body, *operands.get(1)?)?).ok()?;
            split_address(
                body,
                evolution,
                *operands.first()?,
                scale.checked_mul(1_i64.checked_shl(shift)?)?,
                visited,
                depth + 1,
            )
        }
        ValueDef::Operator(Operator::I32Load { .. }, args, _) => {

            let operands: &[Value] = &body.arg_pool[*args];
            let load_address = *operands.first()?;
            let mut inner_visited = visited.clone();
            let inner = split_address(
                body,
                evolution,
                load_address,
                scale,
                &mut inner_visited,
                depth + 1,
            )?;

            if inner.induction {
                return None;
            }
            Some(Split {
                loads: vec![value],
                induction: false,
            })
        }
        ValueDef::Operator(_, _, _) | ValueDef::Alias(_) | ValueDef::PickOutput(_, _, _) => {
            let affine = evolution.affine_of(body, canonical_value(body, value))?;
            Some(Split {
                loads: vec![],
                induction: !affine.induction.is_empty(),
            })
        }
        _ => {

            let affine = evolution.affine_of(body, canonical_value(body, value))?;
            Some(Split {
                loads: vec![],
                induction: !affine.induction.is_empty(),
            })
        }
    }
}
