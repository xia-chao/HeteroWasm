use heterowasm_scev::{canonical_value, constant_value, AffineForm, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{FunctionBody, MemoryArg, Operator, Value, ValueDef};

use crate::consts::ADDRESS_NOT_AFFINE;
use crate::types::{AccessKind, AddressConclusion, MemoryAccess};


#[derive(Debug, Clone, Default)]
pub struct AddressRecovery {
    accesses: Vec<MemoryAccess>,
}

impl AddressRecovery {

    pub fn analyze(body: &FunctionBody, evolution: &ScalarEvolution, trace: &Trace) -> Self {
        let _scope = trace.stage(Stage::AddressRecovery, "function");
        let mut accesses = Vec::new();

        for value in body.values.iter() {
            let Some(ValueDef::Operator(operator, args, _)) = body.values.get(value) else {
                continue;
            };
            let Some((kind, bytes, memory)) = memory_shape(operator) else {
                continue;
            };
            let operands: &[Value] = &body.arg_pool[*args];
            let Some(&address) = operands.first() else {
                continue;
            };

            let conclusion = match AddressRecovery::form_of(body, evolution, address) {
                Some(form) => AddressConclusion::Affine(form),
                None => AddressConclusion::Unknown {
                    reason: ADDRESS_NOT_AFFINE,
                },
            };

            accesses.push(MemoryAccess {
                value,
                block: body.value_blocks[value],
                kind,
                bytes,
                instruction_offset: i64::from(memory.offset),
                conclusion,
            });
        }

        let affine = accesses.iter().filter(|access| access.is_affine()).count();
        trace
            .info(Stage::AddressRecovery, "address recovery complete")
            .field("accesses", accesses.len())
            .field("affine", affine)
            .field("unknown", accesses.len().saturating_sub(affine))
            .emit();


        for access in &accesses {
            if let AddressConclusion::Unknown { reason } = access.conclusion {
                trace
                    .warn(
                        Stage::AddressRecovery,
                        "address cannot be recovered as affine form",
                    )
                    .field("reason", reason)
                    .field("block", access.block.index())
                    .emit();
            }
        }

        Self { accesses }
    }


    pub fn accesses(&self) -> &[MemoryAccess] {
        &self.accesses
    }


    pub fn affine_count(&self) -> usize {
        self.accesses
            .iter()
            .filter(|access| access.is_affine())
            .count()
    }


    pub fn unknown_count(&self) -> usize {
        self.accesses.len().saturating_sub(self.affine_count())
    }


    pub fn form_of(
        body: &FunctionBody,
        evolution: &ScalarEvolution,
        address: Value,
    ) -> Option<AffineForm> {
        evolution
            .affine_of(body, address)
            .or_else(|| Self::shifted_address(body, evolution, address))
            .or_else(|| Self::constant_shift(body, evolution, address))
    }


    fn shifted_address(
        body: &FunctionBody,
        evolution: &ScalarEvolution,
        address: Value,
    ) -> Option<AffineForm> {
        let ValueDef::Operator(Operator::I32Add, args, _) =
            body.values.get(canonical_value(body, address))?
        else {
            return None;
        };
        let operands: &[Value] = &body.arg_pool[*args];
        enum Piece {
            Base(Value),
            Shifted(Value, u32),
        }
        let classify = |operand: Value| {
            let canonical = canonical_value(body, operand);
            match body.values.get(canonical)? {
                ValueDef::BlockParam(..) if !evolution.is_induction(body, canonical) => {
                    Some(Piece::Base(canonical))
                }
                ValueDef::Operator(Operator::I32Shl, inner, _) => {
                    let inner: &[Value] = &body.arg_pool[*inner];
                    let base = canonical_value(body, *inner.first()?);
                    let amount =
                        u32::try_from(constant_value(body, canonical_value(body, *inner.get(1)?))?)
                            .ok()?;
                    if amount < 2 || amount >= 32 {
                        return None;
                    }
                    let variable = evolution
                        .induction_variables()
                        .iter()
                        .find(|variable| canonical_value(body, variable.value) == base)?;
                    if variable.step != 1 {
                        return None;
                    }
                    Some(Piece::Shifted(base, amount))
                }
                _ => None,
            }
        };
        let (base, induction, amount) =
            match (classify(*operands.first()?)?, classify(*operands.get(1)?)?) {
                (Piece::Base(base), Piece::Shifted(induction, amount))
                | (Piece::Shifted(induction, amount), Piece::Base(base)) => {
                    (base, induction, amount)
                }
                _ => return None,
            };
        Some(AffineForm {
            invariants: vec![(base, 1)],
            induction: vec![(induction, 1_i64 << amount)],
            offset: 0,
        })
    }


    fn constant_shift(
        body: &FunctionBody,
        evolution: &ScalarEvolution,
        address: Value,
    ) -> Option<AffineForm> {
        let ValueDef::Operator(Operator::I32Add, args, _) =
            body.values.get(canonical_value(body, address))?
        else {
            return None;
        };
        let operands: &[Value] = &body.arg_pool[*args];
        let left = *operands.first()?;
        let right = *operands.get(1)?;
        let shift_of = |operand: Value| -> Option<(Value, u32)> {
            let ValueDef::Operator(Operator::I32Shl, inner, _) =
                body.values.get(canonical_value(body, operand))?
            else {
                return None;
            };
            let inner: &[Value] = &body.arg_pool[*inner];
            let base = canonical_value(body, *inner.first()?);
            let amount =
                u32::try_from(constant_value(body, canonical_value(body, *inner.get(1)?))?).ok()?;
            if !(2..32).contains(&amount) {
                return None;
            }
            let variable = evolution
                .induction_variables()
                .iter()
                .find(|variable| canonical_value(body, variable.value) == base)?;
            if variable.step != 1 {
                return None;
            }
            Some((variable.value, amount))
        };
        let constant_of = |operand: Value| constant_value(body, canonical_value(body, operand));
        let (induction, amount, offset) = if let (Some((induction, amount)), Some(offset)) =
            (shift_of(left), constant_of(right))
        {
            (induction, amount, offset)
        } else if let (Some((induction, amount)), Some(offset)) =
            (shift_of(right), constant_of(left))
        {
            (induction, amount, offset)
        } else {
            return None;
        };
        Some(AffineForm {
            invariants: Vec::new(),
            induction: vec![(induction, 1_i64.checked_shl(amount)?)],
            offset,
        })
    }
}


fn memory_shape(operator: &Operator) -> Option<(AccessKind, u32, &MemoryArg)> {
    match operator {
        Operator::I32Load { memory } | Operator::F32Load { memory } => {
            Some((AccessKind::Load, 4, memory))
        }
        Operator::I64Load { memory } | Operator::F64Load { memory } => {
            Some((AccessKind::Load, 8, memory))
        }
        Operator::I32Load8S { memory }
        | Operator::I32Load8U { memory }
        | Operator::I64Load8S { memory }
        | Operator::I64Load8U { memory } => Some((AccessKind::Load, 1, memory)),
        Operator::I32Load16S { memory }
        | Operator::I32Load16U { memory }
        | Operator::I64Load16S { memory }
        | Operator::I64Load16U { memory } => Some((AccessKind::Load, 2, memory)),
        Operator::I64Load32S { memory } | Operator::I64Load32U { memory } => {
            Some((AccessKind::Load, 4, memory))
        }
        Operator::I32Store { memory } | Operator::F32Store { memory } => {
            Some((AccessKind::Store, 4, memory))
        }
        Operator::I64Store { memory } | Operator::F64Store { memory } => {
            Some((AccessKind::Store, 8, memory))
        }
        Operator::I32Store8 { memory } | Operator::I64Store8 { memory } => {
            Some((AccessKind::Store, 1, memory))
        }
        Operator::I32Store16 { memory } | Operator::I64Store16 { memory } => {
            Some((AccessKind::Store, 2, memory))
        }
        Operator::I64Store32 { memory } => Some((AccessKind::Store, 4, memory)),
        _ => None,
    }
}
