use crate::consts::{GATE_1_MAX_UNKNOWN, GATE_1_MIN_E1, GATE_1_MIN_E2, GATE_1_MIN_LOOPS};
use crate::types::{Decision, Metrics};


pub(crate) fn decide(metrics: &Metrics) -> Decision {
    let mut reasons = Vec::new();


    if metrics.unsafe_count > 0 {
        reasons.push(describe_unsafe(metrics.unsafe_count));
        return Decision::NoGo { reasons };
    }

    if metrics.loops < GATE_1_MIN_LOOPS {
        reasons.push(describe_sample(metrics.loops));
        return Decision::Inconclusive { reasons };
    }

    if metrics.e1() < GATE_1_MIN_E1 {
        reasons.push(describe_e1(metrics));
    }
    if metrics.e2() < GATE_1_MIN_E2 {
        reasons.push(describe_e2(metrics));
    }
    if metrics.unknown_rate() > GATE_1_MAX_UNKNOWN {
        reasons.push(describe_unknown(metrics));
    }

    if reasons.is_empty() {
        Decision::Go
    } else {
        Decision::NoGo { reasons }
    }
}

fn describe_unsafe(count: usize) -> String {
    format!("{count} unsafe false-negatives present (spec: blocker bug)")
}

fn describe_sample(loops: usize) -> String {
    format!("only {loops} loops in commitment, below floor {GATE_1_MIN_LOOPS}; sample too small to judge")
}

fn describe_e1(metrics: &Metrics) -> String {
    format!(
        "E1 {:.3} below {:.2} ({}/{})",
        metrics.e1(),
        GATE_1_MIN_E1,
        metrics.loops_with_induction,
        metrics.loops
    )
}

fn describe_e2(metrics: &Metrics) -> String {
    format!(
        "E2 {:.3} below {:.2} ({}/{})",
        metrics.e2(),
        GATE_1_MIN_E2,
        metrics.affine_accesses,
        metrics.accesses
    )
}

fn describe_unknown(metrics: &Metrics) -> String {
    format!(
        "dependence Unknown rate {:.3} above {:.2} ({}/{})",
        metrics.unknown_rate(),
        GATE_1_MAX_UNKNOWN,
        metrics.dependence_unknown,
        metrics.loops
    )
}
