use super::{render_json, AnalysisReport, FunctionReport, LoopReport};
use heterowasm_ir::Verdict;
use heterowasm_legality::{Disposition, GuardKind, Obligation, ObligationKind};

fn sample_report() -> AnalysisReport {
    AnalysisReport {
        functions: vec![FunctionReport {
            name: "vector_add".to_string(),
            blocks: 10,
            values: 81,
            loops: vec![LoopReport {
                header: 6,
                outside_coverage: false,
                blocks: 5,
                induction_variables: 4,
                accesses_total: 9,
                accesses_affine: 6,
                accesses_unknown: 3,
                obligations: vec![
                    Obligation {
                        kind: ObligationKind::Control,
                        verdict: Verdict::ProvenLegal,
                        detail: "",
                    },
                    Obligation {
                        kind: ObligationKind::Dependence,
                        verdict: Verdict::Unknown,
                        detail: "",
                    },
                ],
                verdict: Verdict::Unknown,
                disposition: Disposition::GpuAfterGuard,
                guards: vec![GuardKind::Alias, GuardKind::Bounds],
            }],
        }],
    }
}


#[test]
fn json_should_carry_schema_and_counts() {
    let text = render_json(&sample_report());
    assert!(text.starts_with('{') && text.ends_with('}'));
    assert!(text.contains("\"schema\":\"heterowasm.analysis.v1\""));
    assert!(text.contains("\"name\":\"vector_add\""));
    assert!(text.contains("\"induction_variables\":4"));
    assert!(text.contains("\"accesses\":{\"total\":9,\"affine\":6,\"unknown\":3}"));
    assert!(text.contains("\"disposition\":\"gpu_after_guard\""));
    assert!(text.contains("\"guards\":[\"alias\",\"bounds\"]"));
    assert!(text.contains("\"summary\":{\"loops\":1"));
}


#[test]
fn json_should_list_every_obligation_with_its_kind() {
    let text = render_json(&sample_report());
    assert!(text.contains("{\"kind\":\"control\",\"verdict\":\"proven_legal\"}"));
    assert!(text.contains("{\"kind\":\"dependence\",\"verdict\":\"unknown\"}"));
}


#[test]
fn empty_report_should_still_be_well_formed() {
    let text = render_json(&AnalysisReport::default());
    assert!(text.contains("\"functions\":[]"));
    assert!(text.contains("\"summary\":{\"loops\":0"));
}


#[test]
fn function_names_should_be_escaped() {
    let mut report = sample_report();
    report.functions[0].name = "a\"b\\c".to_string();
    let text = render_json(&report);
    assert!(text.contains("\"name\":\"a\\\"b\\\\c\""));
}


#[test]
fn totals_should_follow_the_loops() {
    let mut report = sample_report();
    assert_eq!(report.totals().gpu_after_guard, 1);
    assert_eq!(report.totals().gpu, 0);
    report.functions[0].loops[0].disposition = Disposition::Cpu;
    assert_eq!(report.totals().cpu, 1);
    assert_eq!(report.totals().gpu_after_guard, 0);
}
