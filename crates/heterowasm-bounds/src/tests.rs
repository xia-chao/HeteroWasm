use heterowasm_address::AddressRecovery;
use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::Trace;
use waffle::FuncDecl;

use super::{BoundsAnalysis, BoundsConclusion, MemoryBounds};

type TestResult = Result<(), Box<dyn std::error::Error>>;


#[test]
fn constant_base_with_maximum_should_be_proven() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "integer/constant_base")
        .ok_or("case integer/constant_base is not registered")?;
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
    let first = loops.first().ok_or("case must contain a natural loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);
    let accesses = heterowasm_address_accesses(recovery.accesses(), &first.blocks);
    let memory = MemoryBounds::of(&module);
    let extent = super::LoopExtent::analyze(body, first, &evolution);

    assert_eq!(
        memory.maximum_bytes,
        Some(65_536),
        "this case declares (memory 1 1); maximum must be read"
    );

    let analysis = BoundsAnalysis::analyze(body, &accesses, &evolution, extent, &memory, &trace);
    assert!(
        analysis.proven() > 0,
        "constant base + constant trip count + declared maximum must prove statically"
    );
    Ok(())
}


#[test]
fn memory_bounds_should_follow_the_module_declaration() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "pointwise/vector_add")
        .ok_or("case pointwise/vector_add is not registered")?;
    let bytes = case.to_wasm()?;
    let module = load_module(&bytes, &Trace::silent())?;
    let bounds = MemoryBounds::of(&module);
    assert_eq!(bounds.initial_bytes, 65_536);
    assert_eq!(
        bounds.maximum_bytes, None,
        "case (memory 1) declares no maximum; cannot invent an upper bound"
    );
    Ok(())
}


#[test]
fn parameter_based_pointer_must_not_be_proven() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "pointwise/vector_add")
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
    let first = loops.first().ok_or("case must contain a natural loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);
    let accesses = heterowasm_address_accesses(recovery.accesses(), &first.blocks);
    let memory = MemoryBounds::of(&module);
    let extent = super::LoopExtent::analyze(body, first, &evolution);

    let analysis = BoundsAnalysis::analyze(body, &accesses, &evolution, extent, &memory, &trace);
    assert_eq!(
        analysis.proven(),
        0,
        "A/B/C bases are function parameters; cannot prove statically"
    );
    assert!(analysis.needs_guard() > 0);
    Ok(())
}


#[test]
fn conclusions_should_not_be_constant() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "indirect_index")
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
    let first = loops.first().ok_or("case must contain a natural loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);
    let accesses = heterowasm_address_accesses(recovery.accesses(), &first.blocks);
    let memory = MemoryBounds::of(&module);

    let analysis = BoundsAnalysis::analyze(body, &accesses, &evolution, None, &memory, &trace);
    assert!(
        analysis
            .conclusions()
            .iter()
            .any(|conclusion| matches!(conclusion, BoundsConclusion::Unprovable { .. })),
        "cases with non-affine addresses must yield at least one Unprovable"
    );
    Ok(())
}

fn heterowasm_address_accesses(
    accesses: &[heterowasm_address::MemoryAccess],
    blocks: &[waffle::Block],
) -> Vec<heterowasm_address::MemoryAccess> {
    heterowasm_dependence::accesses_in_loop(accesses, blocks)
}
