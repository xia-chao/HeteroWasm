use std::borrow::Cow;
use std::collections::HashMap;

use heterowasm_corpus::{Case, CASES};
use heterowasm_trace::{Field, Level, Trace};
use waffle::{Block, FuncDecl, FunctionBody, Module, Operator, ValueDef};

use super::{expand_all_bodies, load_module, normalize_bulk_memory, LoadError};


type TestResult = Result<(), Box<dyn std::error::Error>>;


fn corpus_case(name: &str) -> Option<&'static Case> {
    CASES.iter().find(|case| case.name == name)
}


fn first_body<'m>(module: &'m Module<'_>) -> Option<&'m FunctionBody> {
    module.funcs.values().find_map(|decl| match decl {
        FuncDecl::Body(_, _, body) => Some(body),
        _ => None,
    })
}


fn cfg_has_cycle(body: &FunctionBody) -> bool {
    const UNVISITED: u8 = 0;
    const ON_STACK: u8 = 1;
    const DONE: u8 = 2;

    let mut color: HashMap<Block, u8> = HashMap::new();
    let mut stack: Vec<(Block, usize)> = vec![(body.entry, 0)];

    while let Some((block, next_succ)) = stack.pop() {
        if next_succ == 0 {
            color.insert(block, ON_STACK);
        }
        let succs = &body.blocks[block].succs;
        if next_succ < succs.len() {
            stack.push((block, next_succ + 1));
            let succ = succs[next_succ];
            match color.get(&succ).copied().unwrap_or(UNVISITED) {
                ON_STACK => return true,
                UNVISITED => stack.push((succ, 0)),
                DONE => {}
                _ => {}
            }
        } else {
            color.insert(block, DONE);
        }
    }
    false
}

#[test]
fn load_module_should_reject_empty_input_without_panicking() {
    let result = load_module(&[], &Trace::silent());
    assert!(matches!(result, Err(LoadError::Malformed { .. })));
}

#[test]
fn expand_all_bodies_should_recover_the_declared_function_body() -> TestResult {
    let case =
        corpus_case("pointwise/vector_add").ok_or("case pointwise/vector_add is not registered")?;
    let wasm = case.to_wasm()?;
    let mut module = load_module(&wasm, &Trace::silent())?;
    let bodies = expand_all_bodies(&mut module, &Trace::silent())?;
    assert_eq!(bodies, 1);
    Ok(())
}

#[test]
fn recovered_cfg_should_contain_a_cycle_for_the_vector_add_loop() -> TestResult {
    let case =
        corpus_case("pointwise/vector_add").ok_or("case pointwise/vector_add is not registered")?;
    let wasm = case.to_wasm()?;
    let mut module = load_module(&wasm, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = first_body(&module).ok_or("case module must contain one expanded function body")?;

    assert!(
        cfg_has_cycle(body),
        "CFG recovered from Wasm must contain a cycle, else natural loop detection cannot start"
    );
    Ok(())
}

#[test]
fn recovered_ssa_should_expose_the_three_scaled_address_operands() -> TestResult {
    let case =
        corpus_case("pointwise/vector_add").ok_or("case pointwise/vector_add is not registered")?;
    let wasm = case.to_wasm()?;
    let mut module = load_module(&wasm, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = first_body(&module).ok_or("case module must contain one expanded function body")?;

    let scaled_addresses = body
        .values
        .iter()
        .filter(|value| {
            matches!(
                body.values[*value],
                ValueDef::Operator(Operator::I32Mul, _, _)
            )
        })
        .count();

    assert!(
            scaled_addresses >= 3,
            "C[i]=A[i]+B[i] has three addresses each with i*4; must see each in SSA, saw {scaled_addresses}"
        );
    Ok(())
}

#[test]
fn cfg_detection_should_report_no_cycle_for_straight_line_code() -> TestResult {
    let case = corpus_case("control/no_loop").ok_or("case control/no_loop is not registered")?;
    let wasm = case.to_wasm()?;
    let mut module = load_module(&wasm, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = first_body(&module).ok_or("control case must contain one expanded function body")?;

    assert!(
        !cfg_has_cycle(body),
        "acyclic case was judged cyclic — cycle detection is always-true; the prior \"found a cycle\" conclusion is void"
    );
    Ok(())
}


#[test]
fn load_and_expand_should_leave_structured_events() -> TestResult {
    let case =
        corpus_case("pointwise/vector_add").ok_or("case pointwise/vector_add is not registered")?;
    let wasm = case.to_wasm()?;
    let (trace, sink) = Trace::to_memory(Level::Debug);

    let mut module = load_module(&wasm, &trace)?;
    expand_all_bodies(&mut module, &trace)?;

    let collected = sink.events();
    let loaded = collected
        .iter()
        .find(|event| event.message == "module load succeeded")
        .ok_or("successful load must leave an event")?;

    assert_eq!(loaded.int_field("functions"), Some(1));
    assert_eq!(loaded.field("bytes"), Some(&Field::from(wasm.len())));
    assert!(
        collected
            .iter()
            .any(|event| event.message == "function body expand complete"),
        "expand complete must leave an event"
    );
    Ok(())
}


#[test]
fn normalize_should_borrow_unchanged_bytes_when_no_bulk_memory_is_present() -> TestResult {
    let case =
        corpus_case("pointwise/vector_add").ok_or("case pointwise/vector_add is not registered")?;
    let wasm = case.to_wasm()?;
    let normalized = normalize_bulk_memory(&wasm, &Trace::silent());
    assert!(
        matches!(normalized, Cow::Borrowed(_)),
        "must not re-encode the whole module when there are no bulk-memory ops"
    );
    assert_eq!(normalized.as_ref(), wasm.as_slice());
    Ok(())
}


#[test]
fn normalize_should_make_memory_init_loadable_and_record_the_rewrite() -> TestResult {
    let case = corpus_case("bulk_memory_init_in_loop")
        .ok_or("case bulk_memory_init_in_loop is not registered")?;
    let wasm = case.to_wasm()?;


    let mut raw_module = load_module(&wasm, &Trace::silent())?;
    assert!(
        expand_all_bodies(&mut raw_module, &Trace::silent()).is_err(),
        "raw memory.init must be rejected when expanding the body; if expand succeeds, this test is moot"
    );

    let (trace, sink) = Trace::to_memory(Level::Debug);
    let normalized = normalize_bulk_memory(&wasm, &trace);
    assert!(
        matches!(normalized, Cow::Owned(_)),
        "must re-encode when memory.init is present"
    );

    let mut module = load_module(&normalized, &trace)?;
    expand_all_bodies(&mut module, &trace)?;

    let collected = sink.events();
    let rewrote = collected
        .iter()
        .find(|event| event.message == "rewrote unsupported bulk-memory operators")
        .ok_or("rewrite must leave an event, else \"rewrote\" is indistinguishable from \"never needed\"")?;
    assert_eq!(
        rewrote.int_field("rewrites"),
        Some(1),
        "this case has exactly one memory.init; rewrite count must be exactly 1"
    );
    Ok(())
}
