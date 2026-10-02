use heterowasm_corpus::Expectation;
use heterowasm_dependence::LoopLegality;
use heterowasm_legality::Disposition;

use crate::types::{CaseClass, LoopVerdict};


pub(crate) fn classify(expectation: Expectation, loops: &[LoopVerdict]) -> CaseClass {
    let released = loops.iter().any(|verdict| {
        matches!(
            verdict.disposition,
            Disposition::Gpu | Disposition::GpuAfterGuard
        )
    });
    let guarded = loops
        .iter()
        .any(|verdict| matches!(verdict.disposition, Disposition::GpuAfterGuard));

    match expectation {
        Expectation::MustReject { .. } => {
            if released {
                CaseClass::Unsafe {
                    reason: "expected must-reject, but analysis accepted the loop",
                }
            } else {
                CaseClass::Agree
            }
        }
        Expectation::Parallelizable => {
            if loops.is_empty() {
                CaseClass::Conservative {
                    reason: "expected parallelizable, but no loop was recognized",
                }
            } else if loops
                .iter()
                .all(|verdict| matches!(verdict.dependence, LoopLegality::Parallelizable))
            {

                CaseClass::Agree
            } else if guarded {
                CaseClass::Conservative {
                    reason: "expected parallelizable, but aliasing cannot be excluded statically; needs runtime check",
                }
            } else if released {
                CaseClass::Agree
            } else {
                CaseClass::Conservative {
                    reason: "expected parallelizable, but analysis fell back to CPU",
                }
            }
        }
        Expectation::RequiresGuard { .. } => {
            if released {
                CaseClass::Agree
            } else {
                CaseClass::Conservative {
                    reason: "expected parallel after guard, but analysis fell back to CPU",
                }
            }
        }
        Expectation::NoCandidate { .. } => {
            if loops.is_empty() {
                CaseClass::Agree
            } else {
                CaseClass::Unsafe {
                    reason: "expected no parallel candidate, but a loop was recognized",
                }
            }
        }

        Expectation::Undetermined => CaseClass::NotApplicable,
    }
}
