use std::collections::HashSet;

use heterowasm_address::{AccessKind, AddressRecovery, MemoryAccess, NormalizedAddress};
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::NaturalLoop;
use heterowasm_scev::{canonical_value, constant_value, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::{Block, FunctionBody, Operator, Value, ValueDef};

use crate::error::LowerError;
use crate::types::{Dispatch, Kernel};


pub(crate) struct Emitter {
    pub(crate) fields: Vec<Value>,
    pub(crate) statements: Vec<String>,

    pub(crate) min_constant_bytes: i64,

    pub(crate) max_constant_bytes: i64,

    pub(crate) max_stride_bytes: i64,

    trip: String,
}


pub fn lower(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
    accesses: &[MemoryAccess],
    evolution: &ScalarEvolution,
    extent: Option<LoopExtent>,
    workgroup_size: u32,
    trace: &Trace,
) -> Result<Kernel, LowerError> {
    let _scope = trace.stage(Stage::Wgsl, "loop");

    let extent = extent.ok_or(LowerError::NoExtent)?;

    let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();
    let mut emitter = Emitter {
        fields: Vec::new(),
        statements: Vec::new(),
        min_constant_bytes: 0,
        max_constant_bytes: 0,
        max_stride_bytes: 0,
        trip: "i".to_string(),
    };


    fn scaled_start(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<StartOffset> {
        let form = evolution.affine_of(body, value)?;
        if !form.induction.is_empty() || form.offset != 0 || form.invariants.len() != 2 {
            return None;
        }
        let mut slots = [(0_usize, 0_i64); 2];
        for (index, (piece, coefficient)) in form.invariants.iter().enumerate() {
            if *coefficient == 0 {
                return None;
            }
            let canonical = canonical_value(body, *piece);
            let ValueDef::BlockParam(block, _, _) = body.values.get(canonical)? else {
                return None;
            };
            if members.contains(block) {
                return None;
            }
            let slot = field_of(body, canonical, evolution, members, emitter).ok()?;
            slots[index] = (slot, *coefficient);
        }
        Some(StartOffset::Scaled(
            slots[0].0, slots[0].1, slots[1].0, slots[1].1,
        ))
    }

    let start = match extent.start {
        Some(0) => None,
        Some(value) => Some(StartOffset::Constant(value)),
        None => {
            let value = extent.start_value.ok_or(LowerError::NoConstantStart)?;

            let summed = evolution.affine_of(body, value).and_then(|form| {
                if !form.induction.is_empty()
                    || form.offset != 0
                    || form.invariants.len() != 2
                    || form
                        .invariants
                        .iter()
                        .any(|(_, coefficient)| *coefficient != 1)
                {
                    return None;
                }
                enum Piece {
                    Base(usize),
                    Shifted(usize, i64),
                }
                let mut classify = |piece: waffle::Value| -> Option<Piece> {
                    let canonical = canonical_value(body, piece);
                    match body.values.get(canonical)? {
                        waffle::ValueDef::BlockParam(block, _, _) if !members.contains(block) => {
                            field_of(body, canonical, evolution, &members, &mut emitter)
                                .ok()
                                .map(Piece::Base)
                        }
                        waffle::ValueDef::Operator(waffle::Operator::I32Shl, args, _) => {
                            let operands = &body.arg_pool[*args];
                            let base = *operands.first()?;
                            let shift = *operands.get(1)?;
                            let base = canonical_value(body, base);
                            let shift = canonical_value(body, shift);
                            let waffle::ValueDef::BlockParam(block, _, _) =
                                body.values.get(base)?
                            else {
                                return None;
                            };
                            if members.contains(block) {
                                return None;
                            }
                            let amount = constant_value(body, shift)?;
                            if amount < 0 {
                                return None;
                            }
                            let slot =
                                field_of(body, base, evolution, &members, &mut emitter).ok()?;
                            Some(Piece::Shifted(slot, amount))
                        }
                        _ => None,
                    }
                };
                let left = classify(form.invariants[0].0)?;
                let right = classify(form.invariants[1].0)?;
                match (left, right) {
                    (Piece::Base(base), Piece::Shifted(shifted, amount))
                    | (Piece::Shifted(shifted, amount), Piece::Base(base)) => {
                        Some(StartOffset::SumShift(base, shifted, amount))
                    }
                    _ => None,
                }
            });
            Some(if let Some(offset) = summed {
                offset
            } else if let Some(offset) =
                scaled_start(body, value, evolution, &members, &mut emitter)
            {
                offset
            } else if let Some(offset) =
                StartOffset::from_outside_load(body, value, evolution, &members, &mut emitter)
            {
                offset
            } else {
                StartOffset::Field(field_of(body, value, evolution, &members, &mut emitter)?)
            })
        }
    };


    fn outside_and_mask(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<(usize, u32)> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32And, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        let operands = &body.arg_pool[*args];
        let left = canonical_value(body, *operands.first()?);
        let right = canonical_value(body, *operands.get(1)?);
        let (base, mask) = if let Some(mask) = constant_value(body, right) {
            (left, mask)
        } else if let Some(mask) = constant_value(body, left) {
            (right, mask)
        } else {
            return None;
        };
        let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(base)? else {
            return None;
        };
        if members.contains(block) {
            return None;
        }
        let bits = i32::try_from(mask).ok()?;
        let slot = field_of(body, base, evolution, members, emitter).ok()?;
        Some((slot, u32::from_ne_bytes(bits.to_ne_bytes())))
    }


    fn outside_add_limit(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<Dispatch> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32Add, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        enum Piece {
            Base(Value),
            Shifted(Value, u32),
        }
        let classify = |operand: Value| -> Option<Piece> {
            let canonical = canonical_value(body, operand);
            match body.values.get(canonical)? {
                waffle::ValueDef::BlockParam(block, _, _) if !members.contains(block) => {
                    Some(Piece::Base(canonical))
                }
                waffle::ValueDef::Operator(waffle::Operator::I32Shl, inner, _) => {
                    let operands = &body.arg_pool[*inner];
                    let base = canonical_value(body, *operands.first()?);
                    let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(base)? else {
                        return None;
                    };
                    if members.contains(block) {
                        return None;
                    }
                    let amount = constant_value(body, canonical_value(body, *operands.get(1)?))?;
                    let amount = u32::try_from(amount).ok()?;
                    if amount >= 32 {
                        return None;
                    }
                    Some(Piece::Shifted(base, amount))
                }
                _ => None,
            }
        };
        let operands = &body.arg_pool[*args];
        let left = classify(*operands.first()?)?;
        let right = classify(*operands.get(1)?)?;
        let mut slot = |operand: Value| field_of(body, operand, evolution, members, emitter).ok();
        match (left, right) {
            (Piece::Base(left), Piece::Base(right)) => {
                Some(Dispatch::FromSum(slot(left)?, slot(right)?))
            }
            (Piece::Base(base), Piece::Shifted(shifted, amount))
            | (Piece::Shifted(shifted, amount), Piece::Base(base)) => {
                Some(Dispatch::FromSumShift(slot(base)?, slot(shifted)?, amount))
            }
            _ => None,
        }
    }


    fn loaded_shift_mask(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<Dispatch> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32And, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        let operands = &body.arg_pool[*args];
        let left = canonical_value(body, *operands.first()?);
        let right = canonical_value(body, *operands.get(1)?);
        let (load, mask) = if let Some(mask) = constant_value(body, right) {
            (left, mask)
        } else if let Some(mask) = constant_value(body, left) {
            (right, mask)
        } else {
            return None;
        };
        let mask = u32::try_from(mask).ok()?;
        let waffle::ValueDef::Operator(waffle::Operator::I32Load { memory }, load_args, _) =
            body.values.get(load)?
        else {
            return None;
        };
        if memory.offset != 0 {
            return None;
        }
        let address = canonical_value(body, *body.arg_pool[*load_args].first()?);
        let waffle::ValueDef::Operator(waffle::Operator::I32Add, add_args, _) =
            body.values.get(address)?
        else {
            return None;
        };
        enum Piece {
            Base(Value),
            Shifted(Value, u32),
        }
        let classify = |operand: Value| -> Option<Piece> {
            let canonical = canonical_value(body, operand);
            match body.values.get(canonical)? {
                waffle::ValueDef::BlockParam(block, _, _) if !members.contains(block) => {
                    Some(Piece::Base(canonical))
                }
                waffle::ValueDef::Operator(waffle::Operator::I32Shl, inner, _) => {
                    let inner_operands = &body.arg_pool[*inner];
                    let base = canonical_value(body, *inner_operands.first()?);
                    let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(base)? else {
                        return None;
                    };
                    if members.contains(block) {
                        return None;
                    }
                    let amount =
                        constant_value(body, canonical_value(body, *inner_operands.get(1)?))?;
                    let amount = u32::try_from(amount).ok()?;
                    if amount >= 32 {
                        return None;
                    }
                    Some(Piece::Shifted(base, amount))
                }
                _ => None,
            }
        };
        let add_operands = &body.arg_pool[*add_args];
        let (base, shifted, amount) = match (
            classify(*add_operands.first()?)?,
            classify(*add_operands.get(1)?)?,
        ) {
            (Piece::Base(base), Piece::Shifted(shifted, amount))
            | (Piece::Shifted(shifted, amount), Piece::Base(base)) => (base, shifted, amount),
            _ => return None,
        };
        let base = field_of(body, base, evolution, members, emitter).ok()?;
        let shifted = field_of(body, shifted, evolution, members, emitter).ok()?;
        Some(Dispatch::FromLoadedShiftMask(base, shifted, amount, mask))
    }


    fn masked_add_limit(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<Dispatch> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32Add, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        let operands = &body.arg_pool[*args];
        let left = canonical_value(body, *operands.first()?);
        let right = canonical_value(body, *operands.get(1)?);

        let bits = |operand: Value| -> Option<u32> {
            let constant = constant_value(body, operand)?;
            if let Ok(bits) = u32::try_from(constant) {
                return Some(bits);
            }
            let narrow = i32::try_from(constant).ok()?;
            Some(u32::from_ne_bytes(narrow.to_ne_bytes()))
        };
        let (anded, addend) = if let Some(addend) = bits(right) {
            (left, addend)
        } else if let Some(addend) = bits(left) {
            (right, addend)
        } else {
            return None;
        };
        let waffle::ValueDef::Operator(waffle::Operator::I32And, and_args, _) =
            body.values.get(anded)?
        else {
            return None;
        };
        let and_operands = &body.arg_pool[*and_args];
        let and_left = canonical_value(body, *and_operands.first()?);
        let and_right = canonical_value(body, *and_operands.get(1)?);
        let (base, mask) = if let Some(mask) = bits(and_right) {
            (and_left, mask)
        } else if let Some(mask) = bits(and_left) {
            (and_right, mask)
        } else {
            return None;
        };
        let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(base)? else {
            return None;
        };
        if members.contains(block) {
            return None;
        }
        let slot = field_of(body, base, evolution, members, emitter).ok()?;
        Some(Dispatch::FromFieldMaskAdd(slot, mask, addend))
    }


    fn diff_shift_limit(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<Dispatch> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32Add, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        let operands = &body.arg_pool[*args];
        let left = canonical_value(body, *operands.first()?);
        let right = canonical_value(body, *operands.get(1)?);
        let outside = |operand: Value| -> Option<Value> {
            let canonical = canonical_value(body, operand);
            match body.values.get(canonical)? {
                waffle::ValueDef::BlockParam(block, _, _) if !members.contains(block) => {
                    Some(canonical)
                }
                _ => None,
            }
        };
        let shifted = |operand: Value| -> Option<(Value, Value, u32)> {
            let canonical = canonical_value(body, operand);
            let waffle::ValueDef::Operator(waffle::Operator::I32Shl, inner, _) =
                body.values.get(canonical)?
            else {
                return None;
            };
            let inner_operands = &body.arg_pool[*inner];
            let difference = canonical_value(body, *inner_operands.first()?);
            let amount = constant_value(body, canonical_value(body, *inner_operands.get(1)?))?;
            let amount = u32::try_from(amount).ok()?;
            if amount >= 32 {
                return None;
            }
            let waffle::ValueDef::Operator(waffle::Operator::I32Sub, sub_args, _) =
                body.values.get(difference)?
            else {
                return None;
            };
            let sub_operands = &body.arg_pool[*sub_args];
            let minuend = outside(*sub_operands.first()?)?;
            let subtrahend = outside(*sub_operands.get(1)?)?;
            Some((minuend, subtrahend, amount))
        };
        let (base, minuend, subtrahend, amount) = if let Some(base) = outside(left) {
            let (minuend, subtrahend, amount) = shifted(right)?;
            (base, minuend, subtrahend, amount)
        } else if let Some(base) = outside(right) {
            let (minuend, subtrahend, amount) = shifted(left)?;
            (base, minuend, subtrahend, amount)
        } else {
            return None;
        };
        let base = field_of(body, base, evolution, members, emitter).ok()?;
        let minuend = field_of(body, minuend, evolution, members, emitter).ok()?;
        let subtrahend = field_of(body, subtrahend, evolution, members, emitter).ok()?;
        Some(Dispatch::FromSubShift(base, minuend, subtrahend, amount))
    }

    let dispatch = match extent.limit {
        Some(limit) => {
            Dispatch::Fixed(u32::try_from(limit).map_err(|_| LowerError::LimitTooLarge)?)
        }
        None => {
            let value = extent.limit_value.ok_or(LowerError::NoConstantLimit)?;
            if let Some((slot, mask)) =
                outside_and_mask(body, value, evolution, &members, &mut emitter)
            {
                Dispatch::FromFieldMask(slot, mask)
            } else if let Some(dispatch) =
                outside_add_limit(body, value, evolution, &members, &mut emitter)
            {
                dispatch
            } else if let Some(dispatch) =
                diff_shift_limit(body, value, evolution, &members, &mut emitter)
            {
                dispatch
            } else if let Some(dispatch) =
                loaded_shift_mask(body, value, evolution, &members, &mut emitter)
            {
                dispatch
            } else if let Some(dispatch) =
                masked_add_limit(body, value, evolution, &members, &mut emitter)
            {
                dispatch
            } else if let Some(StartOffset::Loaded(slot, offset)) =
                StartOffset::from_outside_load(body, value, evolution, &members, &mut emitter)
            {
                match u32::try_from(offset) {
                    Ok(offset) => Dispatch::FromLoaded(slot, offset),
                    Err(_) => Dispatch::FromField(field_of(
                        body,
                        value,
                        evolution,
                        &members,
                        &mut emitter,
                    )?),
                }
            } else {
                Dispatch::FromField(field_of(body, value, evolution, &members, &mut emitter)?)
            }
        }
    };
    emitter.trip = match start {
        None => "i".to_string(),
        Some(_) => "gid.x".to_string(),
    };

    for access in accesses {

        if access.kind != AccessKind::Store {
            continue;
        }
        let normalized = access
            .normalized(evolution)
            .ok_or(LowerError::UnsupportedBase)?;
        let index = word_index(&normalized, body, evolution, &members, &mut emitter)?;
        let stored = stored_operand(body, access.value)?;
        let value = value_expression(body, stored, evolution, &members, &mut emitter, 256)?;
        emitter.statements.push(render_store(&index, &value.text));
    }


    if emitter.statements.is_empty() {
        return Err(LowerError::NothingToEmit);
    }


    let mut results: Vec<String> = Vec::new();
    for live_out in crate::live_out::live_outs(body, &members) {

        if crate::live_out::is_pass_through(body, natural_loop.header, &members, live_out) {
            continue;
        }
        let canonical = canonical_value(body, live_out);
        let mut rendered =
            value_expression(body, canonical, evolution, &members, &mut emitter, 256)?;

        if let Some(induction) = evolution
            .induction_variables()
            .iter()
            .find(|variable| canonical_value(body, variable.value) == canonical)
        {
            rendered = rendered.plus_constant(induction.step);
        }
        results.push(rendered.text);
    }


    let induction_step = natural_loop
        .blocks
        .iter()
        .find_map(|&block| {
            let def = body.blocks.get(block)?;
            let waffle::Terminator::CondBr {
                cond,
                if_true,
                if_false,
            } = &def.terminator
            else {
                return None;
            };
            if members.contains(&if_true.block) && members.contains(&if_false.block) {
                return None;
            }
            let mut stack = vec![*cond];
            let mut found: Option<i64> = None;
            let mut ambiguous = false;
            let mut seen = 0_usize;
            while let Some(value) = stack.pop() {
                seen += 1;
                if seen > 64 {
                    break;
                }
                let canonical = canonical_value(body, value);
                if let Some(variable) = evolution
                    .induction_variables()
                    .iter()
                    .find(|variable| canonical_value(body, variable.value) == canonical)
                {
                    match found {
                        None => found = Some(variable.step),
                        Some(step) if step != variable.step => ambiguous = true,
                        Some(_) => {}
                    }
                    continue;
                }
                if let Some(ValueDef::Operator(_, args, _)) = body.values.get(canonical) {
                    stack.extend(body.arg_pool[*args].iter().copied());
                }
            }
            if ambiguous {
                None
            } else {
                found
            }
        })
        .filter(|step| *step != 1)
        .unwrap_or(1);


    let launch_count = match (&start, dispatch) {
        (Some(StartOffset::Constant(origin)), Dispatch::Fixed(limit)) if induction_step < 0 => {
            let magnitude = u32::try_from(induction_step.unsigned_abs()).unwrap_or(0);
            let origin = u32::try_from(*origin).unwrap_or(0);
            if magnitude > 0 && origin > limit {

                let span = origin - limit;
                let mut trips = span / magnitude;
                if span % magnitude != 0 {
                    trips = trips.saturating_add(1);
                }
                if trips > limit {
                    Some(trips)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    };

    let source = render_source(
        dispatch,
        start,
        induction_step,
        workgroup_size,
        &emitter.fields,
        &emitter.statements,
        &results,
    );
    let kernel = Kernel {
        source,
        workgroup_size,
        dispatch,
        fields: emitter.fields.clone(),
        min_constant_bytes: emitter.min_constant_bytes,
        max_constant_bytes: emitter.max_constant_bytes,
        max_stride_bytes: emitter.max_stride_bytes,
        index_field: match start {
            Some(StartOffset::Field(position)) => Some(position),
            Some(StartOffset::SumShift(_, _, _)) | Some(StartOffset::Scaled(_, _, _, _)) => None,
            _ => None,
        },
        launch_count,
        results,
    };

    trace
        .info(Stage::Wgsl, "kernel emit complete")
        .field("stores", emitter.statements.len())
        .field("uniform_fields", emitter.fields.len())
        .field(
            "dispatch_is_fixed",
            matches!(kernel.dispatch, Dispatch::Fixed(_)),
        )
        .emit();

    Ok(kernel)
}


fn word_index(
    normalized: &NormalizedAddress,
    body: &FunctionBody,
    evolution: &ScalarEvolution,
    members: &HashSet<Block>,
    emitter: &mut Emitter,
) -> Result<String, LowerError> {
    if normalized.stride.rem_euclid(4) != 0 || normalized.offset.rem_euclid(4) != 0 {
        return Err(LowerError::Unaligned);
    }

    let mut terms: Vec<String> = Vec::new();
    let mut constant_words = normalized.offset / 4;


    fn shift_sum(
        body: &FunctionBody,
        value: waffle::Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<String> {
        let form = evolution.affine_of(body, value)?;
        if !form.induction.is_empty()
            || form.offset != 0
            || form.invariants.len() != 2
            || form
                .invariants
                .iter()
                .any(|(_, coefficient)| *coefficient != 1)
        {
            return None;
        }
        enum Piece {
            Base(usize),
            Shifted(usize, i64),
        }
        let mut classify = |piece: waffle::Value| -> Option<Piece> {
            let canonical = canonical_value(body, piece);
            match body.values.get(canonical)? {
                waffle::ValueDef::BlockParam(block, _, _) if !members.contains(block) => {
                    field_of(body, canonical, evolution, members, emitter)
                        .ok()
                        .map(Piece::Base)
                }
                waffle::ValueDef::Operator(waffle::Operator::I32Shl, args, _) => {
                    let operands = &body.arg_pool[*args];
                    let base = canonical_value(body, *operands.first()?);
                    let shift = canonical_value(body, *operands.get(1)?);
                    let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(base)? else {
                        return None;
                    };
                    if members.contains(block) {
                        return None;
                    }
                    let amount = constant_value(body, shift)?;
                    if amount < 0 {
                        return None;
                    }
                    let slot = field_of(body, base, evolution, members, emitter).ok()?;
                    Some(Piece::Shifted(slot, amount))
                }
                _ => None,
            }
        };
        let left = classify(form.invariants[0].0)?;
        let right = classify(form.invariants[1].0)?;
        match (left, right) {
            (Piece::Base(base), Piece::Shifted(shifted, amount))
            | (Piece::Shifted(shifted, amount), Piece::Base(base)) => Some(format!(
                "{} + ({} << {}u)",
                field_reference(base),
                field_reference(shifted),
                amount.unsigned_abs()
            )),
            _ => None,
        }
    }


    fn affine_words(
        body: &FunctionBody,
        value: waffle::Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<String> {
        let form = evolution.affine_of(body, value)?;
        if !form.induction.is_empty()
            || form.invariants.is_empty()
            || form.offset.rem_euclid(4) != 0
        {
            return None;
        }
        let mut parts = Vec::new();
        for (piece, coefficient) in &form.invariants {
            if *coefficient == 0 {
                continue;
            }
            let slot = field_of(body, *piece, evolution, members, emitter).ok()?;
            let name = field_reference(slot);
            if *coefficient == 1 {
                parts.push(format!("{name} / 4u"));
                continue;
            }
            if coefficient.rem_euclid(4) != 0 {
                return None;
            }
            let words = coefficient / 4;
            if words == 1 {
                parts.push(name);
            } else if words > 1 {
                parts.push(format!("{words}u * {name}"));
            } else {
                parts.push(format!("0u - {}u * {name}", words.unsigned_abs()));
            }
        }
        if parts.is_empty() {
            return None;
        }
        if form.offset > 0 {
            parts.push(format!("{}u", form.offset / 4));
        } else if form.offset < 0 {
            parts.push(format!("0u - {}u", (form.offset / 4).unsigned_abs()));
        }
        Some(parts.join(" + "))
    }

    for (value, coefficient) in &normalized.base {
        if let Some(constant) = constant_value(body, *value) {

            let bytes = constant.saturating_mul(*coefficient);
            if bytes.rem_euclid(4) != 0 {
                return Err(LowerError::Unaligned);
            }
            constant_words = constant_words.saturating_add(bytes / 4);
            continue;
        }
        if *coefficient == 0 {
            continue;
        }
        let name = match field_of(body, *value, evolution, members, emitter) {
            Ok(field) => field_reference(field),
            Err(error) => {
                if *coefficient == 1 {
                    if let Some(text) =
                        StartOffset::from_outside_load(body, *value, evolution, members, emitter)
                    {
                        terms.push(format!("({}) / 4u", text.text()));
                        continue;
                    }
                    if let Some(text) = affine_words(body, *value, evolution, members, emitter) {
                        terms.push(text);
                        continue;
                    }
                }
                if let Some(text) = shift_sum(body, *value, evolution, members, emitter) {
                    if *coefficient != 1 {
                        return Err(error);
                    }
                    terms.push(format!("({text}) / 4u"));
                    continue;
                }
                return Err(error);
            }
        };
        if *coefficient == 1 {
            terms.push(format!("{name} / 4u"));
            continue;
        }
        if coefficient.rem_euclid(4) != 0 {
            return Err(LowerError::Unaligned);
        }
        let words = coefficient / 4;
        if words == 1 {
            terms.push(name);
        } else if words > 1 {
            terms.push(format!("{words}u * {name}"));
        } else {
            terms.push(format!("0u - {}u * {name}", words.unsigned_abs()));
        }
    }

    let stride_words = normalized.stride / 4;
    if stride_words != 0 {

        let trip = if normalized.initial_in_base {
            emitter.trip.clone()
        } else {
            "i".to_string()
        };
        terms.push(scaled_index(stride_words, &trip));
    }
    emitter.max_stride_bytes = emitter.max_stride_bytes.max(normalized.stride.abs());


    let constant_bytes = constant_words.saturating_mul(4);
    emitter.min_constant_bytes = emitter.min_constant_bytes.min(constant_bytes);
    emitter.max_constant_bytes = emitter.max_constant_bytes.max(constant_bytes);

    let mut out = terms.join(" + ");
    if constant_words > 0 {
        if out.is_empty() {
            out = constant_index(constant_words);
        } else {
            out.push_str(" + ");
            out.push_str(&constant_index(constant_words));
        }
    } else if constant_words < 0 {

        if out.is_empty() {
            return Err(LowerError::UnsupportedBase);
        }
        out.push_str(" - ");
        out.push_str(&constant_index(
            i64::try_from(constant_words.unsigned_abs())
                .map_err(|_| LowerError::UnsupportedBase)?,
        ));
    }

    if out.is_empty() {
        return Ok("0u".to_string());
    }
    Ok(out)
}


fn stored_operand(body: &FunctionBody, instruction: Value) -> Result<Value, LowerError> {
    let canonical = canonical_value(body, instruction);
    let Some(ValueDef::Operator(Operator::I32Store { .. }, args, _)) = body.values.get(canonical)
    else {
        return Err(LowerError::UnsupportedStoreShape);
    };
    let operands: &[Value] = &body.arg_pool[*args];
    operands.get(1).copied().ok_or(LowerError::MissingOperand)
}


fn value_expression(
    body: &FunctionBody,
    value: Value,
    evolution: &ScalarEvolution,
    members: &HashSet<Block>,
    emitter: &mut Emitter,
    depth: usize,
) -> Result<Rendered, LowerError> {

    if depth == 0 {
        return Err(LowerError::ExpressionTooDeep);
    }
    let canonical = canonical_value(body, value);

    if let Some(constant) = constant_value(body, canonical) {
        let narrowed = i32::try_from(constant).unwrap_or(0);
        return Ok(Rendered::atom(constant_index(i64::from(narrowed as u32))));
    }

    let Some(definition) = body.values.get(canonical) else {
        return Err(LowerError::MissingValueDefinition);
    };

    match definition {
        ValueDef::BlockParam(..) => match field_of(body, canonical, evolution, members, emitter) {
            Ok(field) => Ok(Rendered::atom(field_reference(field))),

            Err(LowerError::UnsupportedFieldAffine) => {
                Rendered::affine(body, canonical, evolution, members, emitter, depth)
            }
            Err(other) => Err(other),
        },

        ValueDef::Operator(Operator::I32Shl, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first().ok_or(LowerError::MissingOperand)?;
            let right = *operands.get(1).ok_or(LowerError::MissingOperand)?;
            let amount = constant_value(body, canonical_value(body, right))
                .filter(|amount| *amount >= 0)
                .ok_or(LowerError::UnsupportedOperator)?;
            let left = value_expression(body, left, evolution, members, emitter, depth - 1)?;
            Ok(Rendered::atom(format!(
                "({} << {}u)",
                left.text,
                amount.unsigned_abs()
            )))
        }
        ValueDef::Operator(Operator::I32Load { memory }, args, _) => {
            let operands: &[Value] = &body.arg_pool[*args];
            let address = *operands.first().ok_or(LowerError::MissingOperand)?;
            let form = AddressRecovery::form_of(body, evolution, address)
                .ok_or(LowerError::UnsupportedBase)?;
            let normalized = NormalizedAddress::of(&form, i64::from(memory.offset), evolution)
                .ok_or(LowerError::UnsupportedBase)?;
            let index = word_index(&normalized, body, evolution, members, emitter)?;
            Ok(Rendered::atom(load_expression(&index)))
        }
        ValueDef::Operator(operator, args, _) => {
            let text = binary_operator(operator).ok_or(LowerError::UnsupportedOperator)?;
            let operands: &[Value] = &body.arg_pool[*args];
            let left = *operands.first().ok_or(LowerError::MissingOperand)?;
            let right = *operands.get(1).ok_or(LowerError::MissingOperand)?;
            let left = value_expression(body, left, evolution, members, emitter, depth - 1)?;
            let right = value_expression(body, right, evolution, members, emitter, depth - 1)?;
            Ok(binary_expression(&left, text, &right))
        }
        _ => Err(LowerError::UnsupportedValueKind),
    }
}


fn load_expression(index: &str) -> String {
    format!("mem[{index}]")
}

fn binary_operator(operator: &Operator) -> Option<&'static str> {
    match operator {
        Operator::I32Add => Some("+"),
        Operator::I32Sub => Some("-"),
        Operator::I32Mul => Some("*"),
        _ => None,
    }
}

fn binary_expression(left: &Rendered, operator: &str, right: &Rendered) -> Rendered {
    let precedence = operator_precedence(operator);

    let same = right.precedence == precedence;
    let wrap_right = right.precedence < precedence || (operator == "-" && same);
    let left_text = if left.precedence < precedence {
        format!("({})", left.text)
    } else {
        left.text.clone()
    };
    let right_text = if wrap_right {
        format!("({})", right.text)
    } else {
        right.text.clone()
    };
    Rendered {
        text: format!("{left_text} {operator} {right_text}"),
        precedence,
    }
}


const ATOM: u8 = 3;

fn operator_precedence(operator: &str) -> u8 {
    match operator {
        "*" => 2,
        "+" | "-" => 1,
        _ => ATOM,
    }
}


struct Rendered {
    text: String,
    precedence: u8,
}

impl Rendered {
    fn atom(text: String) -> Self {
        Rendered {
            text,
            precedence: ATOM,
        }
    }


    fn scaled(self, coefficient: i64) -> Self {
        match coefficient {
            0 => Rendered::atom("0u".to_string()),
            1 => self,
            value if value > 0 => {
                binary_expression(&self, "*", &Rendered::atom(format!("{value}u")))
            }

            value => binary_expression(
                &Rendered::atom("0u".to_string()),
                "-",
                &binary_expression(
                    &self,
                    "*",
                    &Rendered::atom(format!("{}u", value.unsigned_abs())),
                ),
            ),
        }
    }


    fn plus_constant(self, constant: i64) -> Self {
        if constant == 0 {
            return self;
        }
        let magnitude = Rendered::atom(format!("{}u", constant.unsigned_abs()));
        if constant > 0 {
            binary_expression(&self, "+", &magnitude)
        } else {
            binary_expression(&self, "-", &magnitude)
        }
    }


    fn sum(terms: Vec<Rendered>) -> Option<Self> {
        let mut iter = terms.into_iter();
        let mut total = iter.next()?;
        for term in iter {
            total = binary_expression(&total, "+", &term);
        }
        Some(total)
    }


    fn affine(
        body: &FunctionBody,
        canonical: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
        depth: usize,
    ) -> Result<Self, LowerError> {
        let form = evolution
            .affine_of(body, canonical)
            .ok_or(LowerError::UnsupportedFieldOpaque)?;
        let mut terms: Vec<Rendered> = Vec::new();
        let mut induction_coefficient = 0_i64;

        for (value, coefficient) in &form.invariants {
            let operand = value_expression(
                body,
                *value,
                evolution,
                members,
                emitter,
                depth.saturating_sub(1),
            )?;
            terms.push(operand.scaled(*coefficient));
        }

        for (value, coefficient) in &form.induction {
            let target = canonical_value(body, *value);
            let variable = evolution
                .induction_variables()
                .iter()
                .find(|variable| canonical_value(body, variable.value) == target)
                .ok_or(LowerError::UnsupportedFieldOpaque)?;
            let initial = value_expression(
                body,
                variable.initial,
                evolution,
                members,
                emitter,
                depth.saturating_sub(1),
            )?;
            terms.push(initial.scaled(*coefficient));
            induction_coefficient =
                induction_coefficient.saturating_add(coefficient.saturating_mul(variable.step));
        }

        let mut rendered = Rendered::sum(terms)
            .unwrap_or_else(|| Rendered::atom("0u".to_string()))
            .plus_constant(form.offset);
        if induction_coefficient != 0 {

            rendered = binary_expression(
                &rendered,
                "+",
                &Rendered::atom(emitter.trip.clone()).scaled(induction_coefficient),
            );
        }
        Ok(rendered)
    }
}

fn field_reference(field: usize) -> String {
    format!("params.p{field}")
}


fn field_of(
    body: &FunctionBody,
    value: Value,
    evolution: &ScalarEvolution,
    members: &HashSet<Block>,
    emitter: &mut Emitter,
) -> Result<usize, LowerError> {
    let canonical = canonical_value(body, value);
    match body.values.get(canonical) {
        Some(ValueDef::BlockParam(block, _, _)) => {
            if members.contains(block) && !evolution.invariant_parameters().contains(&canonical) {

                return Err(if evolution.affine_of(body, canonical).is_some() {
                    LowerError::UnsupportedFieldAffine
                } else {
                    LowerError::UnsupportedFieldOpaque
                });
            }
        }
        _ => {

            return Err(if evolution.affine_of(body, canonical).is_some() {
                LowerError::UnsupportedFieldAffine
            } else {
                LowerError::UnsupportedFieldOpaque
            });
        }
    }

    if let Some(position) = emitter.fields.iter().position(|field| *field == canonical) {
        return Ok(position);
    }
    emitter.fields.push(canonical);
    Ok(emitter.fields.len() - 1)
}

fn constant_index(words: i64) -> String {
    format!("{words}u")
}

fn scaled_index(words: i64, trip: &str) -> String {
    if words == 1 {
        trip.to_string()
    } else if words > 1 {
        format!("{words}u * {trip}")
    } else {
        format!("0u - {}u * {trip}", words.unsigned_abs())
    }
}

fn render_store(index: &str, value: &str) -> String {
    format!("  mem[{index}] = {value};")
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartOffset {
    Constant(i64),
    Field(usize),
    SumShift(usize, usize, i64),
    Scaled(usize, i64, usize, i64),
    Loaded(usize, i64),
}

impl StartOffset {

    fn text(self) -> String {
        match self {
            StartOffset::Constant(value) => constant_index(value),
            StartOffset::Field(position) => field_reference(position),
            StartOffset::SumShift(base, shifted, amount) => format!(
                "{} + ({} << {}u)",
                field_reference(base),
                field_reference(shifted),
                amount.unsigned_abs()
            ),
            StartOffset::Scaled(left, left_coefficient, right, right_coefficient) => format!(
                "{} + {}",
                Self::scaled_term(left, left_coefficient),
                Self::scaled_term(right, right_coefficient)
            ),
            StartOffset::Loaded(slot, offset) => {
                let name = field_reference(slot);
                if offset == 0 {
                    format!("mem[{name} / 4u]")
                } else {
                    format!("mem[({name} + {offset}u) / 4u]")
                }
            }
        }
    }


    fn from_outside_load(
        body: &FunctionBody,
        value: Value,
        evolution: &ScalarEvolution,
        members: &HashSet<Block>,
        emitter: &mut Emitter,
    ) -> Option<Self> {
        let canonical = canonical_value(body, value);
        let waffle::ValueDef::Operator(waffle::Operator::I32Load { memory }, args, _) =
            body.values.get(canonical)?
        else {
            return None;
        };
        let offset = i64::from(memory.offset);
        if offset < 0 || offset.rem_euclid(4) != 0 {
            return None;
        }
        let address = canonical_value(body, *body.arg_pool[*args].first()?);
        let waffle::ValueDef::BlockParam(block, _, _) = body.values.get(address)? else {
            return None;
        };
        if members.contains(block) {
            return None;
        }
        let slot = field_of(body, address, evolution, members, emitter).ok()?;
        Some(Self::Loaded(slot, offset))
    }


    fn scaled_term(slot: usize, coefficient: i64) -> String {
        let name = field_reference(slot);
        if coefficient == 1 {
            name
        } else if coefficient > 1 {
            format!("{coefficient}u * {name}")
        } else if coefficient == -1 {
            format!("0u - {name}")
        } else {
            format!("0u - {}u * {name}", coefficient.unsigned_abs())
        }
    }
}

fn render_source(
    dispatch: Dispatch,
    start: Option<StartOffset>,
    induction_step: i64,
    workgroup_size: u32,
    fields: &[Value],
    statements: &[String],
    results: &[String],
) -> String {
    let mut out = String::new();

    out.push_str("struct Params {\n");
    if fields.is_empty() {
        out.push_str("  _unused : u32,\n");
    } else {
        for position in 0..fields.len() {
            out.push_str(&field_declaration(position));
        }
    }
    out.push_str("};\n\n");

    out.push_str("@group(0) @binding(0) var<storage, read_write> mem : array<u32>;\n");
    out.push_str("@group(0) @binding(1) var<uniform> params : Params;\n");

    if !results.is_empty() {
        out.push_str("@group(0) @binding(2) var<storage, read_write> results : array<u32>;\n");
    }
    out.push('\n');
    out.push_str(&compute_header(workgroup_size));

    match start {
        None => {
            if induction_step < 0 {
                out.push_str(&format!(
                    "  let i : u32 = 0u - {}u * gid.x;\n",
                    induction_step.unsigned_abs()
                ));
            } else {
                out.push_str("  let i : u32 = gid.x;\n");
            }
        }
        Some(StartOffset::Constant(value)) => {
            if induction_step < 0 {
                out.push_str(&format!(
                    "  let i : u32 = {} - {}u * gid.x;\n",
                    constant_index(value),
                    induction_step.unsigned_abs()
                ));
            } else {
                out.push_str(&format!(
                    "  let i : u32 = {} + gid.x;\n",
                    constant_index(value)
                ));
            }
        }
        Some(offset @ StartOffset::Field(_))
        | Some(offset @ StartOffset::SumShift(_, _, _))
        | Some(offset @ StartOffset::Scaled(_, _, _, _))
        | Some(offset @ StartOffset::Loaded(_, _)) => {
            let origin = offset.text();
            if induction_step < 0 {
                out.push_str(&format!(
                    "  let i : u32 = {origin} - {}u * gid.x;\n",
                    induction_step.unsigned_abs()
                ));
            } else {
                out.push_str(&format!("  let i : u32 = {origin} + gid.x;\n"));
            }
        }
    }
    out.push_str(&guard(dispatch, induction_step, &start));
    for statement in statements {
        out.push_str(statement);
        out.push('\n');
    }

    if !results.is_empty() {
        let bound = bound_text(dispatch);
        let predicate = if induction_step == 1 {
            format!("i + 1u == {bound}")
        } else if induction_step < 0 {

            let magnitude = induction_step.unsigned_abs();
            let origin = match &start {
                None => "0u".to_string(),
                Some(offset) => offset.text(),
            };
            format!("{origin} - {magnitude}u * gid.x - {magnitude}u <= {bound}")
        } else {
            let current = match &start {
                None => format!("{induction_step}u * gid.x"),
                Some(offset) => {
                    format!("{} + {induction_step}u * gid.x", offset.text())
                }
            };
            format!("{current} + {induction_step}u >= {bound}")
        };
        out.push_str(&format!("  if ({predicate}) {{\n"));
        for (position, expression) in results.iter().enumerate() {
            out.push_str(&format!("    results[{position}] = {expression};\n"));
        }
        out.push_str("  }\n");
    }
    out.push_str("}\n");

    out
}

fn field_declaration(position: usize) -> String {
    format!("  p{position} : u32,\n")
}

fn compute_header(workgroup_size: u32) -> String {
    format!(
        "@compute @workgroup_size({workgroup_size})\nfn main(@builtin(global_invocation_id) gid : vec3<u32>) {{\n"
    )
}

fn bound_text(dispatch: Dispatch) -> String {
    match dispatch {
        Dispatch::Fixed(limit) => constant_index(i64::from(limit)),
        Dispatch::FromField(field) => field_reference(field),
        Dispatch::FromFieldMask(field, mask) => {
            format!("({} & {mask}u)", field_reference(field))
        }
        Dispatch::FromFieldMaskAdd(field, mask, addend) => {
            format!("(({} & {mask}u) + {addend}u)", field_reference(field))
        }
        Dispatch::FromSum(left, right) => {
            format!("({} + {})", field_reference(left), field_reference(right))
        }
        Dispatch::FromSumShift(base, shifted, amount) => format!(
            "({} + ({} << {amount}u))",
            field_reference(base),
            field_reference(shifted)
        ),
        Dispatch::FromSubShift(base, minuend, subtrahend, amount) => format!(
            "({} + (({} - {}) << {amount}u))",
            field_reference(base),
            field_reference(minuend),
            field_reference(subtrahend)
        ),
        Dispatch::FromLoadedShiftMask(base, shifted, amount, mask) => format!(
            "(mem[({} + ({} << {amount}u)) / 4u] & {mask}u)",
            field_reference(base),
            field_reference(shifted)
        ),
        Dispatch::FromLoaded(slot, offset) => {
            let name = field_reference(slot);
            if offset == 0 {
                format!("mem[{name} / 4u]")
            } else {
                format!("mem[({name} + {offset}u) / 4u]")
            }
        }
    }
}

fn guard(dispatch: Dispatch, step: i64, start: &Option<StartOffset>) -> String {
    let bound = bound_text(dispatch);
    if step == 1 {
        return format!("  if (i >= {bound}) {{\n    return;\n  }}\n");
    }

    if step < 0 {
        let magnitude = step.unsigned_abs();
        let origin = match start {
            None => "0u".to_string(),
            Some(offset) => offset.text(),
        };
        let cursor = format!("{origin} - {magnitude}u * gid.x");

        return format!(
            "  if ({origin} < {magnitude}u * gid.x || {cursor} <= {bound}) {{\n    return;\n  }}\n"
        );
    }
    let cursor = match start {
        None => format!("{step}u * gid.x"),
        Some(offset) => format!("{} + {step}u * gid.x", offset.text()),
    };
    format!("  if ({cursor} >= {bound}) {{\n    return;\n  }}\n")
}
