use heterowasm_corpus::Expectation;
use heterowasm_dependence::LoopLegality;
use heterowasm_legality::Disposition;


#[derive(Debug, Clone, Copy)]
pub struct LoopVerdict {
    pub outside_coverage: bool,
    pub induction_variables: usize,
    pub accesses: usize,
    pub affine: usize,
    pub bounds_proven: usize,
    pub disposition: Disposition,
    pub dependence: LoopLegality,
}


#[derive(Debug, Clone)]
pub struct CaseOutcome {
    pub name: String,
    pub group: String,
    pub promised: bool,
    pub expectation: Option<Expectation>,
    pub loops: Vec<LoopVerdict>,
    pub class: CaseClass,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseClass {
    Agree,
    Conservative { reason: &'static str },
    Unsafe { reason: &'static str },
    NotApplicable,
}

impl CaseClass {

    pub fn as_str(self) -> &'static str {
        match self {
            CaseClass::Agree => "agree",
            CaseClass::Conservative { .. } => "conservative",
            CaseClass::Unsafe { .. } => "unsafe",
            CaseClass::NotApplicable => "not_applicable",
        }
    }
}


#[derive(Debug, Clone, Copy, Default)]
pub struct Metrics {
    pub loops: usize,
    pub loops_with_induction: usize,
    pub accesses: usize,
    pub affine_accesses: usize,
    pub dependence_parallelizable: usize,
    pub dependence_guard: usize,
    pub dependence_serial: usize,
    pub dependence_unknown: usize,
    pub agree: usize,
    pub conservative: usize,
    pub unsafe_count: usize,
    pub outside_coverage: usize,
    pub bounds_proven: usize,
}

impl Metrics {

    pub fn e1(self) -> f64 {
        ratio(self.loops_with_induction, self.loops)
    }


    pub fn e2(self) -> f64 {
        ratio(self.affine_accesses, self.accesses)
    }


    pub fn unknown_rate(self) -> f64 {
        ratio(self.dependence_unknown, self.loops)
    }


    pub fn bounds_rate(self) -> f64 {
        ratio(self.bounds_proven, self.accesses)
    }
}


fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Go,
    NoGo { reasons: Vec<String> },
    Inconclusive { reasons: Vec<String> },
}

impl Decision {

    pub fn as_str(&self) -> &'static str {
        match self {
            Decision::Go => "go",
            Decision::NoGo { .. } => "no_go",
            Decision::Inconclusive { .. } => "inconclusive",
        }
    }


    pub fn reasons(&self) -> &[String] {
        match self {
            Decision::Go => &[],
            Decision::NoGo { reasons } | Decision::Inconclusive { reasons } => reasons,
        }
    }
}


#[derive(Debug, Clone)]
pub struct Report {
    pub outcomes: Vec<CaseOutcome>,
    pub metrics: Metrics,
    pub decision: Decision,
}
