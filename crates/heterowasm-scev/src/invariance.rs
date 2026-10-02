use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use waffle::{Block, BlockTarget, FunctionBody, Operator, Terminator, Value, ValueDef};

use crate::canonical::canonical_value;


const EXPANSION_BUDGET: usize = 4096;


#[derive(Default)]
struct Expansion {
    visiting: HashSet<Value>,
    budget: usize,
}


pub struct LoopInvariance<'a> {
    body: &'a FunctionBody,
    members: HashSet<Block>,
    proven: &'a HashSet<Value>,
}

impl<'a> LoopInvariance<'a> {

    pub fn new(
        body: &'a FunctionBody,
        natural_loop: &NaturalLoop,
        proven: &'a HashSet<Value>,
    ) -> Self {
        Self {
            body,
            members: natural_loop.blocks.iter().copied().collect(),
            proven,
        }
    }


    pub fn value_is_invariant(&self, value: Value) -> bool {
        let mut expansion = Expansion {
            visiting: HashSet::new(),
            budget: EXPANSION_BUDGET,
        };
        match self.origins(value, &mut expansion) {
            Some(origins) => origins.len() == 1,
            None => false,
        }
    }


    fn origins(&self, value: Value, expansion: &mut Expansion) -> Option<HashSet<Value>> {

        if expansion.budget == 0 {
            return None;
        }
        expansion.budget -= 1;
        let canonical = canonical_value(self.body, value);
        if self.proven.contains(&canonical) {
            return Some(HashSet::from([canonical]));
        }

        match self.body.values.get(canonical)? {
            ValueDef::BlockParam(block, index, _) => {
                if !self.members.contains(block) {

                    return Some(HashSet::from([canonical]));
                }

                if expansion.visiting.contains(&canonical) {
                    return Some(HashSet::new());
                }
                expansion.visiting.insert(canonical);
                let origins = self.origins_of_param(*block, *index, expansion);
                expansion.visiting.remove(&canonical);
                origins
            }
            ValueDef::Operator(operator, args, _) => {

                if !self.members.contains(&self.body.value_blocks[canonical]) {
                    return Some(HashSet::from([canonical]));
                }

                if matches!(
                    operator,
                    Operator::I32Load { .. }
                        | Operator::I32Load8S { .. }
                        | Operator::I32Load8U { .. }
                        | Operator::I32Load16S { .. }
                        | Operator::I32Load16U { .. }
                        | Operator::I64Load { .. }
                ) {
                    return None;
                }

                let operands: &[Value] = &self.body.arg_pool[*args];
                if operands.is_empty() {
                    return None;
                }
                for operand in operands {
                    if self.origins(*operand, expansion)?.len() != 1 {
                        return None;
                    }
                }
                Some(HashSet::from([canonical]))
            }

            ValueDef::Alias(_) | ValueDef::Placeholder(_) | ValueDef::None => None,
            _ => Some(HashSet::from([canonical])),
        }
    }


    fn origins_of_param(
        &self,
        block: Block,
        index: u32,
        expansion: &mut Expansion,
    ) -> Option<HashSet<Value>> {
        let predecessors = self.body.blocks.get(block)?.preds.clone();
        if predecessors.is_empty() {
            return None;
        }
        let mut origins: HashSet<Value> = HashSet::new();
        for predecessor in predecessors {
            let args = Self::branch_args_to(self.body, predecessor, block)?;
            let argument = *args.get(index as usize)?;
            for origin in self.origins(argument, expansion)? {
                origins.insert(origin);
            }
        }
        Some(origins)
    }


    pub fn branch_args_to(
        body: &'a FunctionBody,
        pred: Block,
        target: Block,
    ) -> Option<&'a [Value]> {
        let def = body.blocks.get(pred)?;
        match &def.terminator {
            Terminator::Br { target: only } if only.block == target => Some(&only.args),
            Terminator::CondBr {
                if_true, if_false, ..
            } => {
                if if_true.block == target {
                    Some(&if_true.args)
                } else if if_false.block == target {
                    Some(&if_false.args)
                } else {
                    None
                }
            }
            Terminator::Select {
                targets, default, ..
            } => targets
                .iter()
                .chain(std::iter::once(default))
                .find(|candidate: &&BlockTarget| candidate.block == target)
                .map(|candidate| candidate.args.as_slice()),
            _ => None,
        }
    }
}
