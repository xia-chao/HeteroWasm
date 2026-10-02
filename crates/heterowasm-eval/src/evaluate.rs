use heterowasm_address::AddressRecovery;
use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::{compiled_cases, Category, Language, CASES};
use heterowasm_dependence::accesses_in_loop;
use heterowasm_frontend::{expand_all_bodies, load_module, normalize_bulk_memory};
use heterowasm_legality::LoopLegal;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::Trace;
use waffle::{FuncDecl, Module};

use crate::classify::classify;
use crate::decide::decide;
use crate::error::EvalError;
use crate::metrics::accumulate;
use crate::types::{CaseClass, CaseOutcome, LoopVerdict, Report};


pub fn evaluate(trace: &Trace) -> Result<Report, EvalError> {
    let mut outcomes = Vec::new();

    for case in CASES {
        let bytes = case.to_wasm().map_err(|err| EvalError::Compile {
            name: case.name.to_string(),
            message: err.to_string(),
        })?;
        let loops = analyze_loops(&bytes, case.name, trace)?;
        let class = classify(case.expectation, &loops);

        outcomes.push(CaseOutcome {
            name: case.name.to_string(),
            group: match case.category {
                Category::Synthetic => "synthetic".to_string(),
                Category::Negative => "negative".to_string(),
                Category::SourceCompiled => "source-compiled".to_string(),
            },
            promised: true,
            expectation: Some(case.expectation),
            loops,
            class,
        });
    }

    for case in compiled_cases()? {
        let label = group_label(case.language, case.opt_level);
        let loops = analyze_loops(&case.bytes, &label, trace)?;
        outcomes.push(CaseOutcome {
            name: label.clone(),
            group: label,

            promised: case.opt_level != "0",
            expectation: None,
            loops,
            class: CaseClass::NotApplicable,
        });
    }

    let metrics = accumulate(&outcomes);
    let decision = decide(&metrics);

    Ok(Report {
        outcomes,
        metrics,
        decision,
    })
}

fn group_label(language: Language, opt_level: &str) -> String {
    let prefix = match language {
        Language::Rust => "rust",
        Language::C => "c",
    };
    format!("{prefix}-o{opt_level}")
}


fn analyze_loops(
    bytes: &[u8],
    subject: &str,
    trace: &Trace,
) -> Result<Vec<LoopVerdict>, EvalError> {

    let normalized = normalize_bulk_memory(bytes, trace);
    let mut module: Module<'_> =
        load_module(&normalized, trace).map_err(|err| EvalError::Load {
            name: subject.to_string(),
            message: err.to_string(),
        })?;
    expand_all_bodies(&mut module, trace).map_err(|err| EvalError::Load {
        name: subject.to_string(),
        message: err.to_string(),
    })?;

    let mut verdicts = Vec::new();
    for decl in module.funcs.values() {
        let FuncDecl::Body(_, _, body) = decl else {
            continue;
        };
        let flow = ControlFlow::analyze(body, trace);
        for natural_loop in flow.natural_loops(body, trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
            let recovery = AddressRecovery::analyze(body, &evolution, trace);
            let accesses = accesses_in_loop(recovery.accesses(), &natural_loop.blocks);
            let legal =
                LoopLegal::judge(body, &natural_loop, &accesses, &evolution, &module, trace);

            verdicts.push(LoopVerdict {
                outside_coverage: evolution.outside_coverage(),
                induction_variables: evolution.induction_variables().len(),

                accesses: accesses.len(),
                affine: accesses.iter().filter(|access| access.is_affine()).count(),
                bounds_proven: legal.bounds_proven(),
                disposition: legal.disposition(),
                dependence: legal.dependence(),
            });
        }
    }
    Ok(verdicts)
}
