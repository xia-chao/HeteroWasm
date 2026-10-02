use heterowasm_address::AddressRecovery;
use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_ir::Verdict;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Level, Stage, Trace};
use waffle::FuncDecl;

use super::{Disposition, GuardKind, LoopLegal, ObligationKind};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn judge_first_loop(
    case_name: &str,
    trace: &Trace,
) -> Result<LoopLegal, Box<dyn std::error::Error>> {
    let case = CASES
        .iter()
        .find(|case| case.name == case_name)
        .ok_or_else(|| format!("case {case_name} is not registered"))?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, trace)?;
    expand_all_bodies(&mut module, trace)?;


    for decl in module.funcs.values() {
        let FuncDecl::Body(_, _, body) = decl else {
            continue;
        };
        let flow = ControlFlow::analyze(body, trace);
        let loops = flow.natural_loops(body, trace);
        let Some(first) = loops.first() else {
            continue;
        };
        let evolution = ScalarEvolution::analyze(body, first, trace);
        let recovery = AddressRecovery::analyze(body, &evolution, trace);
        let accesses = heterowasm_dependence::accesses_in_loop(recovery.accesses(), &first.blocks);
        return Ok(LoopLegal::judge(
            body, first, &accesses, &evolution, &module, trace,
        ));
    }

    Err("case must contain at least one natural loop".into())
}


#[test]
fn loop_carried_scalar_must_be_proven_illegal() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("loop_carried_scalar", &trace)?;
    assert_eq!(
        legal.verdict(),
        Verdict::ProvenIllegal,
        "loop-carried scalar accumulator is a cross-iteration RAW; must prove not parallelizable"
    );
    assert_eq!(legal.disposition(), Disposition::Cpu);
    Ok(())
}


#[test]
fn loop_carried_raw_must_be_proven_illegal() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("loop_carried_raw", &trace)?;
    assert_eq!(legal.verdict(), Verdict::ProvenIllegal);
    assert_eq!(legal.disposition(), Disposition::Cpu);
    Ok(())
}


#[test]
fn indirect_index_must_be_unknown_and_stay_on_cpu() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("indirect_index", &trace)?;
    assert_eq!(legal.verdict(), Verdict::Unknown);
    assert_eq!(
        legal.disposition(),
        Disposition::Cpu,
        "non-affine address cannot be fixed by a runtime check; must not report gpu_after_guard"
    );
    Ok(())
}


#[test]
fn vector_add_should_be_recoverable_by_alias_guard() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("pointwise/vector_add", &trace)?;
    assert_eq!(legal.verdict(), Verdict::Unknown);
    assert_eq!(legal.disposition(), Disposition::GpuAfterGuard);
    assert!(legal.guards().contains(&GuardKind::Alias));
    assert!(
        legal.guards().contains(&GuardKind::Bounds),
        "any memory access requires a bounds guard (spec §15)"
    );
    Ok(())
}


#[test]
fn every_loop_should_report_all_obligations() -> TestResult {
    let trace = Trace::silent();
    for case_name in ["pointwise/vector_add", "loop_carried_raw", "indirect_index"] {
        let legal = judge_first_loop(case_name, &trace)?;
        let kinds: Vec<ObligationKind> = legal
            .obligations()
            .iter()
            .map(|obligation| obligation.kind)
            .collect();
        assert!(kinds.contains(&ObligationKind::Control), "{case_name}");
        assert!(kinds.contains(&ObligationKind::Memory), "{case_name}");
        assert!(kinds.contains(&ObligationKind::Dependence), "{case_name}");
        assert!(kinds.contains(&ObligationKind::Effects), "{case_name}");
    }
    Ok(())
}


#[test]
fn loop_with_call_must_not_be_released() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("call_in_loop", &trace)?;

    assert_ne!(legal.disposition(), Disposition::Gpu);
    assert_ne!(legal.disposition(), Disposition::GpuAfterGuard);
    assert_eq!(
        legal
            .obligations()
            .iter()
            .find(|obligation| obligation.kind == ObligationKind::Effects)
            .map(|obligation| obligation.verdict),
        Some(Verdict::Unknown)
    );
    Ok(())
}


#[test]
fn loop_with_bulk_memory_must_not_be_released() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("bulk_memory_in_loop", &trace)?;

    assert_ne!(legal.disposition(), Disposition::Gpu);
    assert_ne!(legal.disposition(), Disposition::GpuAfterGuard);
    assert_eq!(
        legal
            .obligations()
            .iter()
            .find(|obligation| obligation.kind == ObligationKind::Effects)
            .map(|obligation| obligation.verdict),
        Some(Verdict::Unknown)
    );
    Ok(())
}


#[test]
fn dispositions_should_differ_across_cases() -> TestResult {
    let trace = Trace::silent();
    let serial = judge_first_loop("loop_carried_raw", &trace)?.disposition();
    let guard = judge_first_loop("pointwise/vector_add", &trace)?.disposition();
    let cpu = judge_first_loop("indirect_index", &trace)?.disposition();

    assert_eq!(serial, Disposition::Cpu);
    assert_eq!(guard, Disposition::GpuAfterGuard);
    assert_eq!(cpu, Disposition::Cpu);
    assert_ne!(
        guard, cpu,
        "recoverable and unrecoverable must be distinguished"
    );
    Ok(())
}


#[test]
fn judgement_should_leave_obligation_events() -> TestResult {
    let (trace, sink) = Trace::to_memory(Level::Debug);
    let legal = judge_first_loop("pointwise/vector_add", &trace)?;
    let _ = legal;

    assert!(
        sink.count(Stage::Legality, Level::Debug) >= 3,
        "each of the three obligations must leave one event"
    );
    assert!(sink.contains_message("legality decision complete"));
    Ok(())
}


#[test]
fn unaligned_access_must_be_proven_illegal() -> TestResult {
    let trace = Trace::silent();
    let legal = judge_first_loop("unaligned_access", &trace)?;
    assert_eq!(
        legal.verdict(),
        Verdict::ProvenIllegal,
        "1-byte access violates spec §16 4-byte alignment policy; must hard-reject"
    );
    assert_eq!(legal.disposition(), Disposition::Cpu);
    Ok(())
}


#[test]
fn loops_with_induction_variables_must_pass_control_obligation() -> TestResult {
    let compiled = heterowasm_corpus::compiled_cases()?;
    let optimized: Vec<_> = compiled
        .iter()
        .filter(|case| case.opt_level != "0")
        .collect();
    if optimized.is_empty() {
        eprintln!("note: no optimized-tier artifact found; run bash corpus/source-compiled/build.sh first");
        return Ok(());
    }

    let trace = Trace::silent();
    let mut checked = 0_usize;
    for product in optimized {
        let mut module = load_module(&product.bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for decl in module.funcs.values() {
            let FuncDecl::Body(_, _, body) = decl else {
                continue;
            };
            let flow = ControlFlow::analyze(body, &trace);
            for natural_loop in flow.natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                if evolution.induction_variables().is_empty() {
                    continue;
                }
                let recovery = AddressRecovery::analyze(body, &evolution, &trace);
                let accesses = heterowasm_dependence::accesses_in_loop(
                    recovery.accesses(),
                    &natural_loop.blocks,
                );
                let legal =
                    LoopLegal::judge(body, &natural_loop, &accesses, &evolution, &module, &trace);

                let control = legal
                    .obligations()
                    .iter()
                    .find(|obligation| obligation.kind == ObligationKind::Control);
                assert_eq!(
                    control.map(|obligation| obligation.verdict),
                    Some(Verdict::ProvenLegal),
                    "loops with recognized induction vars must pass the control obligation"
                );
                checked += 1;
            }
        }
    }

    assert!(
        checked > 0,
        "must check at least one loop with a recognized induction variable"
    );
    Ok(())
}
