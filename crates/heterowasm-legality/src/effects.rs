use std::collections::HashSet;

use heterowasm_cfg::NaturalLoop;
use heterowasm_ir::Verdict;
use waffle::{Block, FunctionBody, Operator, ValueDef};

use crate::types::{Obligation, ObligationKind};


pub(crate) fn effects_obligation(body: &FunctionBody, natural_loop: &NaturalLoop) -> Obligation {
    let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();

    for value in body.values.iter() {
        let Some(ValueDef::Operator(operator, _, _)) = body.values.get(value) else {
            continue;
        };
        let Some(reason) = side_effect_reason(operator) else {
            continue;
        };
        if !members.contains(&body.value_blocks[value]) {
            continue;
        }
        return Obligation {
            kind: ObligationKind::Effects,
            verdict: Verdict::Unknown,
            detail: reason,
        };
    }

    Obligation {
        kind: ObligationKind::Effects,
        verdict: Verdict::ProvenLegal,
        detail: "loop body has no side-effect ops; effects are limited to loads/stores visible in this function",
    }
}


fn side_effect_reason(operator: &Operator) -> Option<&'static str> {
    match operator {
        Operator::Call { .. } | Operator::CallIndirect { .. } => {
            Some("loop body contains a call; callee access set and side effects cannot be bounded statically")
        }
        Operator::MemoryCopy { .. } | Operator::MemoryFill { .. } => {
            Some("loop body contains bulk memory ops whose access set bypasses load/store")
        }
        Operator::MemoryGrow { .. } => Some("loop body contains memory.grow; memory size may change across iterations"),
        Operator::GlobalSet { .. } => Some("loop body writes a global"),
        Operator::TableSet { .. } | Operator::TableGrow { .. } => Some("loop body mutates a table"),
        Operator::Unreachable => Some("loop body contains unreachable; execution may be incomplete"),
        _ => None,
    }
}
