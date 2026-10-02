use std::collections::HashSet;

use waffle::{FunctionBody, Value, ValueDef};

use crate::canonical::canonical_value;


#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AffineForm {
    pub invariants: Vec<(Value, i64)>,
    pub induction: Vec<(Value, i64)>,
    pub offset: i64,
}

impl AffineForm {
    pub(crate) fn add_assign(&mut self, other: &AffineForm) {
        for (value, coefficient) in &other.invariants {
            accumulate(&mut self.invariants, *value, *coefficient);
        }
        for (value, coefficient) in &other.induction {
            accumulate(&mut self.induction, *value, *coefficient);
        }
        self.offset = self.offset.saturating_add(other.offset);
        self.normalize();
    }

    pub(crate) fn sub_assign(&mut self, other: &AffineForm) {
        for (value, coefficient) in &other.invariants {
            accumulate(&mut self.invariants, *value, coefficient.saturating_neg());
        }
        for (value, coefficient) in &other.induction {
            accumulate(&mut self.induction, *value, coefficient.saturating_neg());
        }
        self.offset = self.offset.saturating_sub(other.offset);
        self.normalize();
    }

    pub(crate) fn scale(&mut self, factor: i64) {
        for (_, coefficient) in &mut self.invariants {
            *coefficient = coefficient.saturating_mul(factor);
        }
        for (_, coefficient) in &mut self.induction {
            *coefficient = coefficient.saturating_mul(factor);
        }
        self.offset = self.offset.saturating_mul(factor);
        self.normalize();
    }


    fn normalize(&mut self) {
        self.invariants.retain(|(_, coefficient)| *coefficient != 0);
        self.induction.retain(|(_, coefficient)| *coefficient != 0);
    }
}


fn accumulate(terms: &mut Vec<(Value, i64)>, value: Value, coefficient: i64) {
    if let Some(entry) = terms.iter_mut().find(|(existing, _)| *existing == value) {
        entry.1 = entry.1.saturating_add(coefficient);
    } else if coefficient != 0 {
        terms.push((value, coefficient));
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InductionVariable {
    pub value: Value,
    pub initial: Value,
    pub step: i64,
}


#[derive(Debug, Clone, Default)]
pub struct ScalarEvolution {
    pub(crate) induction: Vec<InductionVariable>,

    pub(crate) recurrence_in_memory: bool,

    pub(crate) recovered_from_memory: usize,

    pub(crate) invariant_params: HashSet<Value>,

    pub(crate) carried_non_affine: Vec<Value>,
}

impl ScalarEvolution {

    pub fn invariant_parameters(&self) -> &HashSet<Value> {
        &self.invariant_params
    }


    pub fn recovered_from_memory(&self) -> usize {
        self.recovered_from_memory
    }


    pub fn carried_non_affine(&self) -> &[Value] {
        &self.carried_non_affine
    }


    pub fn induction_variables(&self) -> &[InductionVariable] {
        &self.induction
    }


    pub fn is_induction(&self, body: &FunctionBody, value: Value) -> bool {
        let canonical = canonical_value(body, value);
        self.induction
            .iter()
            .any(|variable| variable.value == canonical)
    }


    pub fn depends_on_induction(&self, body: &FunctionBody, value: Value, depth: usize) -> bool {
        if depth == 0 {
            return false;
        }
        let canonical = canonical_value(body, value);
        if self.is_induction(body, canonical) {
            return true;
        }
        let Some(ValueDef::Operator(_, args, _)) = body.values.get(canonical) else {
            return false;
        };
        let operands: &[Value] = &body.arg_pool[*args];
        operands
            .iter()
            .any(|operand| self.depends_on_induction(body, *operand, depth - 1))
    }


    pub fn outside_coverage(&self) -> bool {
        self.recurrence_in_memory
    }
}
