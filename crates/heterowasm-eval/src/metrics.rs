use heterowasm_dependence::LoopLegality;

use crate::types::{CaseClass, CaseOutcome, Metrics};


pub(crate) fn accumulate(outcomes: &[CaseOutcome]) -> Metrics {
    let mut metrics = Metrics::default();

    for outcome in outcomes {
        match outcome.class {
            CaseClass::Agree => metrics.agree += 1,
            CaseClass::Conservative { .. } => metrics.conservative += 1,
            CaseClass::Unsafe { .. } => metrics.unsafe_count += 1,
            CaseClass::NotApplicable => {}
        }

        for verdict in &outcome.loops {
            if verdict.outside_coverage {
                metrics.outside_coverage += 1;
            }

            if !outcome.promised || verdict.outside_coverage {
                continue;
            }

            metrics.loops += 1;
            if verdict.induction_variables > 0 {
                metrics.loops_with_induction += 1;
            }
            metrics.accesses += verdict.accesses;
            metrics.affine_accesses += verdict.affine;
            metrics.bounds_proven += verdict.bounds_proven;

            match verdict.dependence {
                LoopLegality::Parallelizable => metrics.dependence_parallelizable += 1,
                LoopLegality::RequiresAliasGuard { .. } => metrics.dependence_guard += 1,
                LoopLegality::SerialOnly { .. } => metrics.dependence_serial += 1,
                LoopLegality::Unknown { .. } => metrics.dependence_unknown += 1,
            }
        }
    }

    metrics
}
