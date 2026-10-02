use std::collections::HashSet;

use waffle::{FunctionBody, Operator, Value, ValueDef};

use crate::canonical::{constant_value, i32_bits_to_i64};
use crate::types::{AffineForm, ScalarEvolution};

impl ScalarEvolution {

    pub fn affine_of(&self, body: &FunctionBody, value: Value) -> Option<AffineForm> {
        let mut visiting: HashSet<Value> = HashSet::new();
        self.affine_inner(body, value, &mut visiting)
    }

    fn affine_inner(
        &self,
        body: &FunctionBody,
        value: Value,
        visiting: &mut HashSet<Value>,
    ) -> Option<AffineForm> {

        if !visiting.insert(value) {
            return None;
        }
        let result = self.affine_of_value(body, value, visiting);
        visiting.remove(&value);
        result.or_else(|| self.symbolic_invariant(body, value))
    }


    fn symbolic_invariant(&self, body: &FunctionBody, value: Value) -> Option<AffineForm> {
        if self.depends_on_induction(body, value, 8) {
            return None;
        }
        Some(AffineForm {
            invariants: vec![(value, 1)],
            induction: Vec::new(),
            offset: 0,
        })
    }

    fn affine_of_value(
        &self,
        body: &FunctionBody,
        value: Value,
        visiting: &mut HashSet<Value>,
    ) -> Option<AffineForm> {
        match body.values.get(value)? {
            ValueDef::Operator(Operator::I32Const { value: raw }, _, _) => Some(AffineForm {
                invariants: Vec::new(),
                induction: Vec::new(),
                offset: i32_bits_to_i64(*raw),
            }),
            ValueDef::Operator(Operator::I64Const { value: raw }, _, _) => Some(AffineForm {
                invariants: Vec::new(),
                induction: Vec::new(),
                offset: i64::try_from(*raw).ok()?,
            }),
            ValueDef::Operator(Operator::I32Add, args, _) => {
                let operands: &[Value] = &body.arg_pool[*args];
                let mut total = self.affine_inner(body, *operands.first()?, visiting)?;
                let second = self.affine_inner(body, *operands.get(1)?, visiting)?;
                total.add_assign(&second);
                Some(total)
            }
            ValueDef::Operator(Operator::I32Sub, args, _) => {
                let operands: &[Value] = &body.arg_pool[*args];
                let mut total = self.affine_inner(body, *operands.first()?, visiting)?;
                let second = self.affine_inner(body, *operands.get(1)?, visiting)?;
                total.sub_assign(&second);
                Some(total)
            }
            ValueDef::Operator(Operator::I32Mul, args, _) => {
                let operands: &[Value] = &body.arg_pool[*args];
                let left = *operands.first()?;
                let right = *operands.get(1)?;

                if let Some(factor) = constant_value(body, right) {
                    let mut scaled = self.affine_inner(body, left, visiting)?;
                    scaled.scale(factor);
                    Some(scaled)
                } else if let Some(factor) = constant_value(body, left) {
                    let mut scaled = self.affine_inner(body, right, visiting)?;
                    scaled.scale(factor);
                    Some(scaled)
                } else {
                    None
                }
            }
            ValueDef::BlockParam(..) => {
                if self.is_induction(body, value) {
                    Some(AffineForm {
                        invariants: Vec::new(),
                        induction: vec![(value, 1)],
                        offset: 0,
                    })
                } else {
                    Some(AffineForm {
                        invariants: vec![(value, 1)],
                        induction: Vec::new(),
                        offset: 0,
                    })
                }
            }
            ValueDef::PickOutput(inner, 0, _) | ValueDef::Alias(inner) => {
                self.affine_inner(body, *inner, visiting)
            }
            _ => None,
        }
    }
}
