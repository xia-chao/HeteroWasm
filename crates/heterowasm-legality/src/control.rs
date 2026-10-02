use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use heterowasm_ir::Verdict;
use heterowasm_scev::ScalarEvolution;
use waffle::{Block, FunctionBody, Terminator, Value, ValueDef};

use crate::types::{Obligation, ObligationKind};


pub(crate) fn control_obligation(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
    evolution: &ScalarEvolution,
) -> Obligation {
    if natural_loop.latches.is_empty() {
        return Obligation {
            kind: ObligationKind::Control,
            verdict: Verdict::ProvenIllegal,
            detail: "natural loop has no back-edge",
        };
    }
    if evolution.induction_variables().is_empty() {
        return Obligation {
            kind: ObligationKind::Control,
            verdict: Verdict::Unknown,
            detail: "no induction variable recognized; trip bound cannot be determined",
        };
    }
    if !branch_tests_induction(body, natural_loop, evolution) {
        return Obligation {
            kind: ObligationKind::Control,
            verdict: Verdict::Unknown,
            detail: "loop exit condition does not test the induction variable; trip bound cannot be determined",
        };
    }
    Obligation {
        kind: ObligationKind::Control,
        verdict: Verdict::ProvenLegal,
        detail: "counted loop: header / latch / exit condition on induction var are all present",
    }
}


fn branch_tests_induction(
    body: &FunctionBody,
    natural_loop: &NaturalLoop,
    evolution: &ScalarEvolution,
) -> bool {
    let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();

    for &block in &natural_loop.blocks {
        let Some(def) = body.blocks.get(block) else {
            continue;
        };
        let Terminator::CondBr {
            cond,
            if_true,
            if_false,
        } = &def.terminator
        else {
            continue;
        };
        let is_exit = !members.contains(&if_true.block) || !members.contains(&if_false.block);
        if !is_exit {
            continue;
        }
        if condition_tests_induction(body, *cond, evolution) {
            return true;
        }
    }
    false
}

fn condition_tests_induction(
    body: &FunctionBody,
    condition: Value,
    evolution: &ScalarEvolution,
) -> bool {
    let Some(ValueDef::Operator(_, args, _)) = body.values.get(condition) else {
        return false;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    operands
        .iter()
        .any(|operand| depends_on_induction(body, *operand, evolution, 8))
}


fn depends_on_induction(
    body: &FunctionBody,
    value: Value,
    evolution: &ScalarEvolution,
    depth: usize,
) -> bool {
    if depth == 0 {
        return false;
    }
    let canonical = heterowasm_scev::canonical_value(body, value);
    if evolution.is_induction(body, canonical) {
        return true;
    }
    let Some(ValueDef::Operator(_, args, _)) = body.values.get(canonical) else {
        return false;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    operands
        .iter()
        .any(|operand| depends_on_induction(body, *operand, evolution, depth - 1))
}
