use heterowasm_dependence::LoopLegality;
use heterowasm_legality::Disposition;

use super::{CaseClass, CaseOutcome, Decision, LoopVerdict, Metrics, GATE_1_MIN_LOOPS};
use crate::classify::classify;
use crate::decide::decide;
use crate::metrics::accumulate;
use heterowasm_corpus::Expectation;
use heterowasm_trace::Trace;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn verdict(disposition: Disposition, dependence: LoopLegality) -> LoopVerdict {
    LoopVerdict {
        outside_coverage: false,
        induction_variables: 1,
        accesses: 4,
        affine: 4,
        bounds_proven: 0,
        disposition,
        dependence,
    }
}


#[test]
fn must_reject_is_unsafe_when_any_loop_is_released() {
    let loops = vec![
        verdict(Disposition::Cpu, LoopLegality::SerialOnly { reason: "" }),
        verdict(Disposition::Gpu, LoopLegality::Parallelizable),
    ];
    let class = classify(
        Expectation::MustReject {
            reason: "loop-carried RAW",
        },
        &loops,
    );
    assert!(
        matches!(class, CaseClass::Unsafe { .. }),
        "actual: {class:?}"
    );
}

#[test]
fn must_reject_is_agree_when_all_loops_stay_on_cpu() {
    let loops = vec![verdict(Disposition::Cpu, LoopLegality::Parallelizable)];
    let class = classify(Expectation::MustReject { reason: "x" }, &loops);
    assert_eq!(class, CaseClass::Agree);
}


#[test]
fn parallelizable_expectation_with_guard_is_conservative_not_unsafe() {
    let loops = vec![verdict(
        Disposition::GpuAfterGuard,
        LoopLegality::RequiresAliasGuard { pairs: 2 },
    )];
    let class = classify(Expectation::Parallelizable, &loops);
    assert!(
        matches!(class, CaseClass::Conservative { .. }),
        "actual: {class:?}"
    );
}


#[test]
fn undetermined_is_never_counted_as_pass_or_reject() {
    let loops = vec![verdict(
        Disposition::Cpu,
        LoopLegality::Unknown { reason: "x" },
    )];
    assert_eq!(
        classify(Expectation::Undetermined, &loops),
        CaseClass::NotApplicable
    );
}

fn outcome(name: &str, promised: bool, loops: Vec<LoopVerdict>) -> CaseOutcome {
    CaseOutcome {
        name: name.to_string(),
        group: name.to_string(),
        promised,
        expectation: None,
        loops,
        class: CaseClass::NotApplicable,
    }
}


#[test]
fn unpromised_products_should_stay_out_of_the_denominator() {
    let outcomes = vec![
        outcome(
            "c-o0",
            false,
            vec![verdict(
                Disposition::Cpu,
                LoopLegality::Unknown { reason: "x" },
            )],
        ),
        outcome(
            "c-o1",
            true,
            vec![verdict(Disposition::Cpu, LoopLegality::Parallelizable)],
        ),
    ];
    let metrics = accumulate(&outcomes);
    assert_eq!(
        metrics.loops, 1,
        "only the in-commitment loop should be counted"
    );
    assert_eq!(metrics.accesses, 4);
}


#[test]
fn outside_coverage_loops_should_be_reported_but_not_counted() {
    let mut debug_loop = verdict(Disposition::Cpu, LoopLegality::Unknown { reason: "x" });
    debug_loop.outside_coverage = true;
    let outcomes = vec![outcome("rust-o1", true, vec![debug_loop])];

    let metrics = accumulate(&outcomes);
    assert_eq!(metrics.loops, 0);
    assert_eq!(metrics.outside_coverage, 1);
}

#[test]
fn unsafe_false_negative_should_force_no_go_regardless_of_other_metrics() {
    let metrics = Metrics {
        loops: 100,
        loops_with_induction: 100,
        accesses: 100,
        affine_accesses: 100,
        unsafe_count: 1,
        ..Metrics::default()
    };
    assert!(matches!(decide(&metrics), Decision::NoGo { .. }));
}


#[test]
fn small_sample_should_be_inconclusive_not_go() {
    let metrics = Metrics {
        loops: GATE_1_MIN_LOOPS - 1,
        loops_with_induction: GATE_1_MIN_LOOPS - 1,
        accesses: 40,
        affine_accesses: 40,
        ..Metrics::default()
    };
    assert!(matches!(decide(&metrics), Decision::Inconclusive { .. }));
}


#[test]
fn healthy_metrics_should_go() {
    let metrics = Metrics {
        loops: 40,
        loops_with_induction: 40,
        accesses: 100,
        affine_accesses: 95,
        dependence_parallelizable: 10,
        dependence_guard: 25,
        dependence_unknown: 3,
        ..Metrics::default()
    };
    assert_eq!(decide(&metrics), Decision::Go);
}

#[test]
fn low_e1_should_force_no_go() {
    let metrics = Metrics {
        loops: 40,
        loops_with_induction: 20,
        accesses: 100,
        affine_accesses: 95,
        ..Metrics::default()
    };
    assert!(matches!(decide(&metrics), Decision::NoGo { .. }));
}


#[test]
fn evaluation_should_reach_a_decision() -> TestResult {
    let trace = Trace::silent();
    let report = super::evaluate(&trace)?;
    assert!(!report.outcomes.is_empty(), "corpus must not be empty");
    assert!(
        report.metrics.loops > 0,
        "commitment range should have at least one loop; actual outcomes={}",
        report.outcomes.len()
    );
    Ok(())
}
