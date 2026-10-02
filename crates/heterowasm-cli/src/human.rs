use heterowasm_eval::{
    CaseClass, Report as EvalReport, GATE_1_MAX_UNKNOWN, GATE_1_MIN_E1, GATE_1_MIN_E2,
};
use heterowasm_legality::{GuardKind, Obligation};

use crate::types::AnalysisReport;


pub(crate) fn render_human(report: &AnalysisReport) {
    for function in &report.functions {
        if function.loops.is_empty() {
            continue;
        }
        println!(
            "  {} — {} basic blocks, {} SSA values, {} natural loops",
            function.name,
            function.blocks,
            function.values,
            function.loops.len()
        );
        for loop_report in &function.loops {

            let marker = if loop_report.outside_coverage {
                "  [out of coverage]"
            } else {
                ""
            };
            println!(
                "    loop @{} — {} induction vars, {} memory accesses ({} affine / {} unrecovered){marker}",
                loop_report.header,
                loop_report.induction_variables,
                loop_report.accesses_total,
                loop_report.accesses_affine,
                loop_report.accesses_unknown
            );
            println!(
                "      obligation: {}",
                describe_obligations(&loop_report.obligations)
            );
            println!(
                "      destination: {}{}",
                loop_report.disposition.as_str(),
                describe_guards(&loop_report.guards)
            );
        }
    }

    let totals = report.totals();
    println!();
    println!("── loop destination summary ───────────────────────────");
    println!("  direct GPU     {}", totals.gpu);
    println!("  GPU after guard {}", totals.gpu_after_guard);
    println!("  fall back CPU  {}", totals.cpu);

    println!();
    println!("── coverage ───────────────────────────────────────────");
    println!(
        "  Phase 0 only commits to -O1 and above artifacts (Gate 1 decision b, see DEVPLAN §5.5)."
    );
    println!("  measurement still covers three tiers (spec §62), but debug artifacts are excluded from E1/E2/E3.");
    if totals.outside_coverage > 0 {
        println!(
            "  this run has {} loops outside that range: induction vars stay in memory stack slots.",
            totals.outside_coverage
        );
        println!(
            "  bringing them into coverage requires mem2reg first — a post-Phase-0 extension."
        );
    } else {
        println!("  no out-of-coverage loops found in this run.");
    }
}


fn describe_obligations(obligations: &[Obligation]) -> String {
    let mut text = String::new();
    for (index, obligation) in obligations.iter().enumerate() {
        if index > 0 {
            text.push(' ');
        }
        text.push_str(obligation.kind.as_str());
        text.push('=');
        text.push_str(obligation.verdict_str());
    }
    text
}


fn describe_guards(guards: &[GuardKind]) -> String {
    if guards.is_empty() {
        return String::new();
    }
    let mut text = String::from("（guard: ");
    for (index, guard) in guards.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        text.push_str(guard.as_str());
    }
    text.push('）');
    text
}


pub(crate) fn render_eval_human(report: &EvalReport) {
    let metrics = &report.metrics;

    println!("── group detail ───────────────────────────────────────");
    println!(
        "  {:<14}{:>6}{:>8}{:>10}{:>10}",
        "group", "loop", "has IV", "accesses", "affine"
    );
    for group in summarize_groups(report) {
        let mark = if group.promised {
            ""
        } else {
            "  [outside commitment]"
        };
        println!(
            "  {:<14}{:>6}{:>8}{:>10}{:>10}{mark}",
            group.label, group.loops, group.loops_with_induction, group.accesses, group.affine
        );
    }

    println!();
    println!("── corpus reconciliation ──────────────────────────────");
    println!("  match           {}", metrics.agree);
    println!("  more conservative {}", metrics.conservative);
    println!("  **unsafe pass**  {}", metrics.unsafe_count);
    println!("  excluded        {}", count_not_applicable(report));
    for outcome in &report.outcomes {
        if let CaseClass::Unsafe { reason } = outcome.class {
            println!("    ✖ {} — {reason}", outcome.name);
        }
    }

    println!();
    println!("── E1/E2/E3 (within commitment, spec §E1–§E3) ──────────");
    println!(
        "  E1 induction recovery  {}/{} = {:.3}   (need ≥ {:.2})  {}",
        metrics.loops_with_induction,
        metrics.loops,
        metrics.e1(),
        GATE_1_MIN_E1,
        verdict_mark(metrics.e1() >= GATE_1_MIN_E1)
    );
    println!(
        "  E2 affine addr recovery {}/{} = {:.3}   (need ≥ {:.2})  {}",
        metrics.affine_accesses,
        metrics.accesses,
        metrics.e2(),
        GATE_1_MIN_E2,
        verdict_mark(metrics.e2() >= GATE_1_MIN_E2)
    );
    println!(
        "  E3 dep Unknown rate  {}/{} = {:.3}   (need ≤ {:.2})  {}",
        metrics.dependence_unknown,
        metrics.loops,
        metrics.unknown_rate(),
        GATE_1_MAX_UNKNOWN,
        verdict_mark(metrics.unknown_rate() <= GATE_1_MAX_UNKNOWN)
    );
    println!(
        "  E3 unsafe pass       {}                       (need = 0)  {}",
        metrics.unsafe_count,
        verdict_mark(metrics.unsafe_count == 0)
    );
    println!(
        "  bounds static proof  {}/{} = {:.3}   (spec §15 path 1)",
        metrics.bounds_proven,
        metrics.accesses,
        metrics.bounds_rate()
    );
    println!();
    println!(
        "  dependence verdicts: parallel {} / needs guard {} / must serialize {} / Unknown {}",
        metrics.dependence_parallelizable,
        metrics.dependence_guard,
        metrics.dependence_serial,
        metrics.dependence_unknown
    );
    if metrics.outside_coverage > 0 {
        println!(
            "  plus {} loops out of coverage, excluded from denominators (see DEVPLAN §5.5)",
            metrics.outside_coverage
        );
    }

    println!();
    println!("── Gate 1 verdict ─────────────────────────────────────");
    println!("  {}", report.decision.as_str().to_uppercase());
    for reason in report.decision.reasons() {
        println!("    · {reason}");
    }
}


struct GroupSummary {
    label: String,
    promised: bool,
    loops: usize,
    loops_with_induction: usize,
    accesses: usize,
    affine: usize,
}

fn summarize_groups(report: &EvalReport) -> Vec<GroupSummary> {
    let mut groups: Vec<GroupSummary> = Vec::new();

    for outcome in &report.outcomes {
        let index = match groups.iter().position(|group| group.label == outcome.group) {
            Some(index) => index,
            None => {
                groups.push(GroupSummary {
                    label: outcome.group.clone(),
                    promised: outcome.promised,
                    loops: 0,
                    loops_with_induction: 0,
                    accesses: 0,
                    affine: 0,
                });
                groups.len() - 1
            }
        };
        let group = &mut groups[index];
        for verdict in &outcome.loops {
            group.loops += 1;
            if verdict.induction_variables > 0 {
                group.loops_with_induction += 1;
            }
            group.accesses += verdict.accesses;
            group.affine += verdict.affine;
        }
    }

    groups
}

fn count_not_applicable(report: &EvalReport) -> usize {
    report
        .outcomes
        .iter()
        .filter(|outcome| outcome.class == CaseClass::NotApplicable)
        .count()
}

fn verdict_mark(passed: bool) -> &'static str {
    if passed {
        "PASS"
    } else {
        "FAIL"
    }
}
