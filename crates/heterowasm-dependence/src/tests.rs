use heterowasm_address::AddressRecovery;
use heterowasm_cfg::{ControlFlow, NaturalLoop};
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Level, Stage, Trace};
use waffle::{FuncDecl, FunctionBody};

use super::{accesses_in_loop, DependenceAnalysis, LoopLegality};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn with_first_loop<F>(case_name: &str, inspect: F) -> TestResult
where
    F: FnOnce(&FunctionBody, &NaturalLoop, &Trace),
{
    let case = CASES
        .iter()
        .find(|case| case.name == case_name)
        .ok_or_else(|| format!("case {case_name} is not registered"))?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let first = loops
        .first()
        .ok_or("case must contain at least one natural loop")?;
    inspect(body, first, &trace);
    Ok(())
}

fn legality_of(body: &FunctionBody, natural_loop: &NaturalLoop, trace: &Trace) -> LoopLegality {
    let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
    let recovery = AddressRecovery::analyze(body, &evolution, trace);
    let accesses = accesses_in_loop(recovery.accesses(), &natural_loop.blocks);
    DependenceAnalysis::analyze(&accesses, &evolution, trace).legality()
}


#[test]
fn vector_add_should_require_alias_guard() -> TestResult {
    with_first_loop("pointwise/vector_add", |body, natural_loop, trace| {
        let legality = legality_of(body, natural_loop, trace);
        assert!(
            matches!(legality, LoopLegality::RequiresAliasGuard { .. }),
            "A/B/C bases cannot statically exclude aliasing; expect needs-guard, got: {legality:?}"
        );
    })
}


#[test]
fn loop_carried_raw_must_be_serial_only() -> TestResult {
    with_first_loop("loop_carried_raw", |body, natural_loop, trace| {
        let legality = legality_of(body, natural_loop, trace);
        assert!(
            matches!(legality, LoopLegality::SerialOnly { .. }),
            "A[i] = A[i-1] + 1 has cross-iteration RAW; must serialize, got: {legality:?}"
        );
    })
}


#[test]
fn indirect_index_must_be_unknown() -> TestResult {
    with_first_loop("indirect_index", |body, natural_loop, trace| {
        let legality = legality_of(body, natural_loop, trace);
        assert!(
            matches!(legality, LoopLegality::Unknown { .. }),
            "A[B[i]] has a non-affine address; must be Unknown, got: {legality:?}"
        );
    })
}


#[test]
fn straight_line_code_has_no_loop_to_judge() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "control/no_loop")
        .ok_or("case control/no_loop is not registered")?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    assert!(flow.natural_loops(body, &trace).is_empty());
    Ok(())
}


#[test]
fn verdicts_should_not_be_constant_across_cases() -> TestResult {
    let mut guard = 0_usize;
    let mut serial = 0_usize;
    let mut unknown = 0_usize;

    for case_name in ["pointwise/vector_add", "loop_carried_raw", "indirect_index"] {
        let case = CASES
            .iter()
            .find(|case| case.name == case_name)
            .ok_or("case is not registered")?;
        let bytes = case.to_wasm()?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let body = module
            .funcs
            .values()
            .find_map(|decl| match decl {
                FuncDecl::Body(_, _, body) => Some(body),
                _ => None,
            })
            .ok_or("case must contain one expanded function body")?;

        let trace = Trace::silent();
        let flow = ControlFlow::analyze(body, &trace);
        let loops = flow.natural_loops(body, &trace);
        let first = loops
            .first()
            .ok_or("case must contain at least one natural loop")?;
        match legality_of(body, first, &trace) {
            LoopLegality::RequiresAliasGuard { .. } => guard += 1,
            LoopLegality::SerialOnly { .. } => serial += 1,
            LoopLegality::Unknown { .. } => unknown += 1,
            LoopLegality::Parallelizable => {}
        }
    }

    assert_eq!(guard, 1, "exactly one case should need alias checking");
    assert_eq!(serial, 1, "exactly one case should require serialization");
    assert_eq!(unknown, 1, "exactly one case should be undecidable");
    Ok(())
}


#[test]
fn every_pair_decision_should_leave_an_event() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "loop_carried_raw")
        .ok_or("case is not registered")?;
    let bytes = case.to_wasm()?;
    let (trace, sink) = Trace::to_memory(Level::Debug);
    let mut module = load_module(&bytes, &trace)?;
    expand_all_bodies(&mut module, &trace)?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let first = loops
        .first()
        .ok_or("case must contain at least one natural loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);
    let accesses = accesses_in_loop(recovery.accesses(), &first.blocks);
    let analysis = DependenceAnalysis::analyze(&accesses, &evolution, &trace);

    assert!(
        analysis.pairs_checked() > 0,
        "this case must have comparable access pairs"
    );
    assert!(
        sink.count(Stage::Dependence, Level::Debug) > 0,
        "each access-pair decision must leave an event"
    );
    assert!(sink.contains_message("loop dependence decision complete"));

    let overlap_events = sink
        .events()
        .into_iter()
        .filter(|event| event.field("verdict").is_some())
        .count();
    assert_eq!(
        overlap_events,
        analysis.pairs_checked(),
        "event count must match the number of access pairs actually compared"
    );
    Ok(())
}


#[test]
fn out_of_coverage_loop_must_be_unknown() -> TestResult {
    let compiled = heterowasm_corpus::compiled_cases()?;
    let debug_products: Vec<_> = compiled
        .iter()
        .filter(|case| case.opt_level == "0")
        .collect();
    if debug_products.is_empty() {
        eprintln!("note: no -O0 artifact found; run bash corpus/source-compiled/build.sh first");
        return Ok(());
    }

    let trace = Trace::silent();
    let mut checked = 0_usize;
    for product in debug_products {
        let mut module = load_module(&product.bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for decl in module.funcs.values() {
            let FuncDecl::Body(_, _, body) = decl else {
                continue;
            };
            let flow = ControlFlow::analyze(body, &trace);
            for natural_loop in flow.natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                if !evolution.outside_coverage() {
                    continue;
                }
                let recovery = AddressRecovery::analyze(body, &evolution, &trace);
                let accesses = accesses_in_loop(recovery.accesses(), &natural_loop.blocks);
                let analysis = DependenceAnalysis::analyze(&accesses, &evolution, &trace);

                assert_eq!(
                    analysis.legality(),
                    LoopLegality::Unknown {
                        reason: super::OUTSIDE_COVERAGE
                    },
                    "out-of-coverage loops must be Unknown; no definite conclusion allowed"
                );
                assert_eq!(
                    analysis.pairs_checked(),
                    0,
                    "must not compare any access pairs when out of coverage"
                );
                checked += 1;
            }
        }
    }

    assert!(checked > 0, "must check at least one out-of-coverage loop");
    Ok(())
}


#[test]
fn same_address_read_modify_write_should_be_parallelizable() -> TestResult {
    with_first_loop("integer/inplace_scale", |body, natural_loop, trace| {
        let legality = legality_of(body, natural_loop, trace);
        assert_eq!(
            legality,
            LoopLegality::Parallelizable,
            "A[i] = A[i] * 2 reads/writes in the same iteration; not loop-carried"
        );
    })
}


#[test]
fn loop_carried_delta_must_stay_serial() -> TestResult {
    with_first_loop("loop_carried_raw", |body, natural_loop, trace| {
        let legality = legality_of(body, natural_loop, trace);
        assert!(
            matches!(legality, LoopLegality::SerialOnly { .. }),
            "A[i] = A[i-1] + 1 solves to i - j = -1; must serialize, got: {legality:?}"
        );
    })
}


#[test]
fn nested_loop_inner_may_be_parallelizable_but_outer_must_not_be() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "unsupported/nested_loop")
        .ok_or("case unsupported/nested_loop is not registered")?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    assert_eq!(
        loops.len(),
        2,
        "nested case should identify inner and outer natural loops"
    );

    let mut parallelizable = 0_usize;
    let mut unknown = 0_usize;
    for natural_loop in &loops {
        match legality_of(body, natural_loop, &trace) {
            LoopLegality::Parallelizable => parallelizable += 1,
            LoopLegality::Unknown { .. } => unknown += 1,
            other => panic!("nested loops must not yield a third verdict kind: {other:?}"),
        }
    }

    assert_eq!(parallelizable, 1, "inner loop should be parallelizable");
    assert_eq!(
        unknown, 1,
        "outer loop mentions inner variables; must be undecidable"
    );
    Ok(())
}
