use heterowasm_dependence::LoopLegality;
use heterowasm_ir::Verdict;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationKind {
    Control,
    Memory,
    Dependence,

    Effects,
}

impl ObligationKind {

    pub fn as_str(self) -> &'static str {
        match self {
            ObligationKind::Control => "control",
            ObligationKind::Memory => "memory",
            ObligationKind::Dependence => "dependence",
            ObligationKind::Effects => "effects",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardKind {
    Alias,
    Bounds,
}

impl GuardKind {

    pub fn as_str(self) -> &'static str {
        match self {
            GuardKind::Alias => "alias",
            GuardKind::Bounds => "bounds",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    Gpu,
    GpuAfterGuard,
    Cpu,
}

impl Disposition {

    pub fn as_str(self) -> &'static str {
        match self {
            Disposition::Gpu => "gpu",
            Disposition::GpuAfterGuard => "gpu_after_guard",
            Disposition::Cpu => "cpu",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Obligation {
    pub kind: ObligationKind,
    pub verdict: Verdict,
    pub detail: &'static str,
}

impl Obligation {

    pub fn verdict_str(&self) -> &'static str {
        verdict_name(self.verdict)
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopLegal {
    pub(crate) verdict: Verdict,
    pub(crate) disposition: Disposition,
    pub(crate) obligations: Vec<Obligation>,
    pub(crate) guards: Vec<GuardKind>,

    pub(crate) dependence: LoopLegality,

    pub(crate) bounds_proven: usize,
    pub(crate) bounds_total: usize,
}

impl LoopLegal {

    pub fn bounds_proven(&self) -> usize {
        self.bounds_proven
    }


    pub fn bounds_total(&self) -> usize {
        self.bounds_total
    }


    pub fn dependence(&self) -> LoopLegality {
        self.dependence
    }


    pub fn verdict(&self) -> Verdict {
        self.verdict
    }


    pub fn disposition(&self) -> Disposition {
        self.disposition
    }


    pub fn obligations(&self) -> &[Obligation] {
        &self.obligations
    }


    pub fn guards(&self) -> &[GuardKind] {
        &self.guards
    }
}


pub(crate) fn verdict_name(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::ProvenLegal => "proven_legal",
        Verdict::ProvenIllegal => "proven_illegal",
        Verdict::Unknown => "unknown",
    }
}
