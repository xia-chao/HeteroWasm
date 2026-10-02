use std::collections::HashSet;

use super::{compiled_cases, Case, Category, Expectation, CASES};

type TestResult = Result<(), Box<dyn std::error::Error>>;


fn compile_checked(case: &Case) -> Result<(), Box<dyn std::error::Error>> {
    case.to_wasm()
        .map_err(|err| format!("case {} failed to compile: {err}", case.name))?;
    Ok(())
}

#[test]
fn every_case_should_compile_to_wasm() -> TestResult {
    for case in CASES {
        compile_checked(case)?;
    }
    Ok(())
}

#[test]
fn case_names_should_be_unique() {
    let mut seen = HashSet::new();
    for case in CASES {
        assert!(seen.insert(case.name), "duplicate case name: {}", case.name);
    }
}

#[test]
fn every_negative_case_should_state_a_rejection_or_guard_reason() {
    for case in CASES.iter().filter(|c| c.category == Category::Negative) {
        let stated = match case.expectation {
            Expectation::MustReject { reason } => !reason.trim().is_empty(),
            Expectation::RequiresGuard { guard } => !guard.trim().is_empty(),
            _ => false,
        };
        assert!(
            stated,
            "negative case {} has no reject/guard reason; if you cannot write one, it is not really negative",
            case.name
        );
    }
}

#[test]
fn parallelizable_cases_should_all_contain_a_loop() -> TestResult {

    for case in CASES
        .iter()
        .filter(|c| c.expectation == Expectation::Parallelizable)
    {
        let text = case.wat;
        assert!(
            text.contains("(loop"),
            "case {} expects parallelizable but contains no loop structure",
            case.name
        );
    }
    Ok(())
}

#[test]
fn compiled_corpus_should_be_reported_even_when_absent() -> TestResult {
    let cases = compiled_cases()?;
    if cases.is_empty() {
        eprintln!("note: source-compiled artifacts are empty; run bash corpus/source-compiled/build.sh first");
    }
    assert!(
        cases.len() <= 6,
        "source-compiled case count exceeds expectation"
    );
    Ok(())
}
