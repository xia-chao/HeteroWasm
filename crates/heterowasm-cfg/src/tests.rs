use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_trace::Trace;
use waffle::{FuncDecl, FunctionBody};

use super::ControlFlow;

type TestResult = Result<(), Box<dyn std::error::Error>>;


fn with_first_body<F>(case_name: &str, inspect: F) -> TestResult
where
    F: FnOnce(&FunctionBody, &Trace),
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
    inspect(body, &trace);
    Ok(())
}

#[test]
fn entry_should_dominate_every_reachable_block() -> TestResult {
    with_first_body("pointwise/vector_add", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        for &block in flow.reverse_postorder() {
            assert!(
                flow.dominates(body.entry, block),
                "entry must dominate every block reachable from it"
            );
        }
    })
}

#[test]
fn every_block_should_dominate_itself() -> TestResult {
    with_first_body("pointwise/vector_add", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        for &block in flow.reverse_postorder() {
            assert!(flow.dominates(block, block));
        }
    })
}

#[test]
fn vector_add_should_have_exactly_one_natural_loop() -> TestResult {
    with_first_body("pointwise/vector_add", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        let loops = flow.natural_loops(body, trace);
        assert_eq!(loops.len(), 1, "pointwise loop should identify exactly one");
        assert!(
            !loops[0].latches.is_empty(),
            "a natural loop must have at least one back-edge, else it should not be recognized as a loop"
        );
    })
}

#[test]
fn straight_line_code_should_have_no_natural_loop() -> TestResult {
    with_first_body("control/no_loop", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        assert!(
            flow.natural_loops(body, trace).is_empty(),
            "straight-line code has no back-edges; must not report any loop"
        );
    })
}

#[test]
fn two_separate_loops_should_be_reported_as_two() -> TestResult {
    with_first_body("control/two_loops", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        assert_eq!(flow.natural_loops(body, trace).len(), 2);
    })
}

#[test]
fn loop_header_should_dominate_every_block_in_its_loop() -> TestResult {
    with_first_body("pointwise/vector_add", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        for natural_loop in flow.natural_loops(body, trace) {
            for &block in &natural_loop.blocks {
                assert!(
                    flow.dominates(natural_loop.header, block),
                    "natural loop header must dominate every block in the loop body"
                );
            }
        }
    })
}

#[test]
fn nested_loops_should_both_be_detected_although_nesting_is_out_of_scope() -> TestResult {
    with_first_body("unsupported/nested_loop", |body, trace| {
        let flow = ControlFlow::analyze(body, trace);
        let loops = flow.natural_loops(body, trace);
        assert_eq!(
            loops.len(),
            2,
            "nested loops should be recognized as two natural loops"
        );


        let outer = loops[0].header;
        let inner = loops[1].header;
        assert!(
            flow.dominates(outer, inner),
            "outer loop header should dominate inner loop header"
        );
    })
}
