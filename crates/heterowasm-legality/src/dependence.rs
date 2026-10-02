use heterowasm_dependence::{DependenceAnalysis, LoopLegality};
use heterowasm_ir::Verdict;
use heterowasm_scev::ScalarEvolution;

use crate::types::{Obligation, ObligationKind};


pub(crate) fn dependence_obligation(
    analysis: DependenceAnalysis,
    evolution: &ScalarEvolution,
) -> Obligation {

    let carried = evolution.carried_non_affine();
    if !carried.is_empty() {
        return Obligation {
            kind: ObligationKind::Dependence,
            verdict: Verdict::ProvenIllegal,
            detail: "loop-carried scalar with non-constant update (e.g. reduction accumulator) — iterations are not independent",
        };
    }

    match analysis.legality() {
        LoopLegality::Parallelizable => Obligation {
            kind: ObligationKind::Dependence,
            verdict: Verdict::ProvenLegal,
            detail: "all access pairs pass the GCD test; no intra-loop overlap",
        },
        LoopLegality::RequiresAliasGuard { .. } => Obligation {
            kind: ObligationKind::Dependence,
            verdict: Verdict::Unknown,
            detail: "access pairs with different bases cannot statically exclude aliasing",
        },
        LoopLegality::SerialOnly { reason } => Obligation {
            kind: ObligationKind::Dependence,
            verdict: Verdict::ProvenIllegal,
            detail: reason,
        },
        LoopLegality::Unknown { reason } => Obligation {
            kind: ObligationKind::Dependence,
            verdict: Verdict::Unknown,
            detail: reason,
        },
    }
}
