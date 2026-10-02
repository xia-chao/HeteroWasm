use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Level, Stage, Trace};
use waffle::{FuncDecl, FunctionBody};

use super::{AccessKind, AddressConclusion, AddressRecovery};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn with_body<F>(case_name: &str, inspect: F) -> TestResult
where
    F: FnOnce(&FunctionBody) -> TestResult,
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
    inspect(body)
}


fn recovery_of(body: &FunctionBody) -> AddressRecovery {
    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let evolution = match loops.first() {
        Some(natural_loop) => ScalarEvolution::analyze(body, natural_loop, &trace),
        None => ScalarEvolution::default(),
    };
    AddressRecovery::analyze(body, &evolution, &trace)
}

#[test]
fn vector_add_should_recover_all_three_accesses_as_affine() -> TestResult {
    with_body("pointwise/vector_add", |body| {
        let recovery = recovery_of(body);
        assert_eq!(
            recovery.accesses().len(),
            3,
            "C[i] = A[i] + B[i] has two loads and one store"
        );
        assert_eq!(
            recovery.unknown_count(),
            0,
            "this case has no non-affine addresses"
        );
        Ok(())
    })
}

#[test]
fn vector_add_store_should_be_affine_with_stride_four() -> TestResult {
    with_body("pointwise/vector_add", |body| {
        let recovery = recovery_of(body);
        let store = recovery
            .accesses()
            .iter()
            .find(|access| access.kind == AccessKind::Store)
            .ok_or("case must contain one store")?;
        let affine = store
            .affine()
            .ok_or("address of C[i] must be recoverable")?;
        assert_eq!(affine.induction.len(), 1);
        assert_eq!(affine.induction[0].1, 4);
        assert_eq!(affine.offset, 0);
        Ok(())
    })
}

#[test]
fn strided_copy_load_should_report_stride_eight() -> TestResult {
    with_body("stride/strided_copy", |body| {
        let recovery = recovery_of(body);
        let load = recovery
            .accesses()
            .iter()
            .find(|access| access.kind == AccessKind::Load)
            .ok_or("case must contain one load")?;
        let affine = load
            .affine()
            .ok_or("address of C[2i] must be recoverable")?;
        assert_eq!(affine.induction[0].1, 8);
        Ok(())
    })
}

#[test]
fn indirect_index_should_leave_at_least_one_access_unknown() -> TestResult {
    with_body("indirect_index", |body| {
        let recovery = recovery_of(body);
        assert!(
            recovery.unknown_count() >= 1,
            "A[i] = B[C[i]] has an address depending on a load; some access must be unrecovered"
        );
        Ok(())
    })
}


#[test]
fn unknown_access_should_carry_a_reason() -> TestResult {
    with_body("indirect_index", |body| {
        let recovery = recovery_of(body);
        let unknown = recovery
            .accesses()
            .iter()
            .find(|access| !access.is_affine())
            .ok_or("this case must contain an unrecovered access")?;
        match &unknown.conclusion {
            AddressConclusion::Unknown { reason } => {
                assert!(!reason.trim().is_empty(), "Unknown must carry a reason");
            }
            AddressConclusion::Affine(_) => {
                return Err("filter guarantees this is Unknown".into());
            }
        }
        Ok(())
    })
}


#[test]
fn unknown_access_should_emit_a_warning_event() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "indirect_index")
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
        .ok_or("no function body")?;

    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let first = loops.first().ok_or("case must contain a loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);

    assert!(recovery.unknown_count() >= 1);
    assert!(
        sink.count(Stage::AddressRecovery, Level::Warn) >= 1,
        "every unrecovered address must leave a WARN event, else recovery rate cannot be counted from the event stream"
    );
    assert!(sink.contains_message("address recovery complete"));
    Ok(())
}
