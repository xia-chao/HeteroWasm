use heterowasm_eval::{CaseOutcome, Report as EvalReport};
use heterowasm_trace::escape_json_into;

use crate::consts::{EVAL_SCHEMA, SCHEMA};
use crate::types::{AnalysisReport, FunctionReport, LoopReport, Totals};


pub fn render_json(report: &AnalysisReport) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("{\"schema\":\"");
    out.push_str(SCHEMA);
    out.push_str(
        "\",\"coverage\":{\"promised\":\"o1_and_above\",\"measured\":[\"o0\",\"o1\",\"o3\"]}",
    );
    out.push_str(",\"functions\":[");

    for (index, function) in report.functions.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_function_json(function, &mut out);
    }

    out.push_str("],\"summary\":");
    write_totals_json(&report.totals(), &mut out);
    out.push('}');
    out
}

fn write_function_json(function: &FunctionReport, out: &mut String) {
    out.push_str("{\"name\":\"");
    escape_json_into(&function.name, out);
    out.push_str("\",\"blocks\":");
    push_usize(function.blocks, out);
    out.push_str(",\"values\":");
    push_usize(function.values, out);
    out.push_str(",\"loops\":[");

    for (index, loop_report) in function.loops.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_loop_json(loop_report, out);
    }

    out.push_str("]}");
}

fn write_loop_json(loop_report: &LoopReport, out: &mut String) {
    out.push_str("{\"header\":");
    push_usize(loop_report.header, out);
    out.push_str(",\"outside_coverage\":");
    out.push_str(if loop_report.outside_coverage {
        "true"
    } else {
        "false"
    });
    out.push_str(",\"blocks\":");
    push_usize(loop_report.blocks, out);
    out.push_str(",\"induction_variables\":");
    push_usize(loop_report.induction_variables, out);

    out.push_str(",\"accesses\":{\"total\":");
    push_usize(loop_report.accesses_total, out);
    out.push_str(",\"affine\":");
    push_usize(loop_report.accesses_affine, out);
    out.push_str(",\"unknown\":");
    push_usize(loop_report.accesses_unknown, out);
    out.push('}');

    out.push_str(",\"obligations\":[");
    for (index, obligation) in loop_report.obligations.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"kind\":\"");
        out.push_str(obligation.kind.as_str());
        out.push_str("\",\"verdict\":\"");
        out.push_str(obligation.verdict_str());
        out.push_str("\"}");
    }
    out.push(']');

    out.push_str(",\"verdict\":\"");
    out.push_str(loop_report.verdict.as_str());
    out.push_str("\",\"disposition\":\"");
    out.push_str(loop_report.disposition.as_str());

    out.push_str("\",\"guards\":[");
    for (index, guard) in loop_report.guards.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(guard.as_str());
        out.push('"');
    }
    out.push_str("]}");
}

fn write_totals_json(totals: &Totals, out: &mut String) {
    out.push_str("{\"loops\":");
    push_usize(totals.loops, out);
    out.push_str(",\"outside_coverage\":");
    push_usize(totals.outside_coverage, out);
    out.push_str(",\"gpu\":");
    push_usize(totals.gpu, out);
    out.push_str(",\"gpu_after_guard\":");
    push_usize(totals.gpu_after_guard, out);
    out.push_str(",\"cpu\":");
    push_usize(totals.cpu, out);
    out.push('}');
}


pub(crate) fn render_eval_json(report: &EvalReport) -> String {
    let metrics = &report.metrics;
    let mut out = String::with_capacity(1024);

    out.push_str("{\"schema\":\"");
    out.push_str(EVAL_SCHEMA);
    out.push_str("\",\"criteria\":{\"min_e1\":");
    push_f64(heterowasm_eval::GATE_1_MIN_E1, &mut out);
    out.push_str(",\"min_e2\":");
    push_f64(heterowasm_eval::GATE_1_MIN_E2, &mut out);
    out.push_str(",\"max_dependence_unknown\":");
    push_f64(heterowasm_eval::GATE_1_MAX_UNKNOWN, &mut out);
    out.push_str(",\"min_loops\":");
    push_usize(heterowasm_eval::GATE_1_MIN_LOOPS, &mut out);

    out.push_str("},\"metrics\":{\"loops\":");
    push_usize(metrics.loops, &mut out);
    out.push_str(",\"loops_with_induction\":");
    push_usize(metrics.loops_with_induction, &mut out);
    out.push_str(",\"accesses\":");
    push_usize(metrics.accesses, &mut out);
    out.push_str(",\"affine_accesses\":");
    push_usize(metrics.affine_accesses, &mut out);
    out.push_str(",\"e1\":");
    push_f64(metrics.e1(), &mut out);
    out.push_str(",\"e2\":");
    push_f64(metrics.e2(), &mut out);
    out.push_str(",\"dependence_unknown\":");
    push_usize(metrics.dependence_unknown, &mut out);
    out.push_str(",\"dependence_unknown_rate\":");
    push_f64(metrics.unknown_rate(), &mut out);
    out.push_str(",\"dependence_parallelizable\":");
    push_usize(metrics.dependence_parallelizable, &mut out);
    out.push_str(",\"dependence_guard\":");
    push_usize(metrics.dependence_guard, &mut out);
    out.push_str(",\"dependence_serial\":");
    push_usize(metrics.dependence_serial, &mut out);
    out.push_str(",\"agree\":");
    push_usize(metrics.agree, &mut out);
    out.push_str(",\"conservative\":");
    push_usize(metrics.conservative, &mut out);
    out.push_str(",\"unsafe\":");
    push_usize(metrics.unsafe_count, &mut out);
    out.push_str(",\"outside_coverage\":");
    push_usize(metrics.outside_coverage, &mut out);
    out.push_str(",\"bounds_proven\":");
    push_usize(metrics.bounds_proven, &mut out);
    out.push_str(",\"bounds_rate\":");
    push_f64(metrics.bounds_rate(), &mut out);
    out.push('}');

    out.push_str(",\"decision\":\"");
    out.push_str(report.decision.as_str());
    out.push_str("\",\"reasons\":[");
    for (index, reason) in report.decision.reasons().iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        escape_json_into(reason, &mut out);
        out.push('"');
    }
    out.push(']');

    out.push_str(",\"cases\":[");
    for (index, outcome) in report.outcomes.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_outcome_json(outcome, &mut out);
    }
    out.push_str("]}");
    out
}

fn write_outcome_json(outcome: &CaseOutcome, out: &mut String) {
    out.push_str("{\"name\":\"");
    escape_json_into(&outcome.name, out);
    out.push_str("\",\"group\":\"");
    escape_json_into(&outcome.group, out);
    out.push_str("\",\"promised\":");
    out.push_str(if outcome.promised { "true" } else { "false" });
    out.push_str(",\"expectation\":");
    match outcome.expectation {
        Some(expectation) => {
            out.push('"');
            out.push_str(expectation.as_str());
            out.push('"');
        }
        None => out.push_str("null"),
    }
    out.push_str(",\"class\":\"");
    out.push_str(outcome.class.as_str());
    out.push_str("\",\"loops\":[");
    for (index, verdict) in outcome.loops.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str("{\"outside_coverage\":");
        out.push_str(if verdict.outside_coverage {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"induction_variables\":");
        push_usize(verdict.induction_variables, out);
        out.push_str(",\"accesses\":");
        push_usize(verdict.accesses, out);
        out.push_str(",\"affine\":");
        push_usize(verdict.affine, out);
        out.push_str(",\"disposition\":\"");
        out.push_str(verdict.disposition.as_str());
        out.push_str("\",\"dependence\":\"");
        out.push_str(verdict.dependence.as_str());
        out.push_str("\"}");
    }
    out.push_str("]}");
}


fn push_usize(value: usize, out: &mut String) {
    out.push_str(&value.to_string());
}


fn push_f64(value: f64, out: &mut String) {
    out.push_str(&format!("{value:.4}"));
}
