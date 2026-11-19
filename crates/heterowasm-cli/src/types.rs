use heterowasm_ir::Verdict;
use heterowasm_legality::{Disposition, GuardKind, Obligation, ObligationKind};


pub(crate) struct KernelEmission {
    pub(crate) written: usize,
    pub(crate) skipped: usize,
    pub(crate) emitted: Vec<(usize, usize, String)>,
}


#[derive(Default)]
pub struct AnalysisReport {
    pub(crate) functions: Vec<FunctionReport>,
}

pub struct FunctionReport {
    pub(crate) name: String,
    pub(crate) blocks: usize,
    pub(crate) values: usize,
    pub(crate) loops: Vec<LoopReport>,
}

pub struct LoopReport {
    pub(crate) header: usize,
    pub(crate) outside_coverage: bool,
    pub(crate) blocks: usize,
    pub(crate) induction_variables: usize,
    pub(crate) accesses_total: usize,
    pub(crate) accesses_affine: usize,
    pub(crate) accesses_unknown: usize,
    pub(crate) obligations: Vec<Obligation>,
    pub(crate) verdict: Verdict,
    pub(crate) disposition: Disposition,
    pub(crate) guards: Vec<GuardKind>,
}


#[derive(Default)]
pub struct Totals {
    pub(crate) loops: usize,
    pub(crate) outside_coverage: usize,
    pub(crate) gpu: usize,
    pub(crate) gpu_after_guard: usize,
    pub(crate) cpu: usize,

    pub(crate) illegal: [usize; 4],
    pub(crate) unknown: [usize; 4],
}

impl Totals {

    fn slot(kind: ObligationKind) -> usize {
        match kind {
            ObligationKind::Control => 0,
            ObligationKind::Memory => 1,
            ObligationKind::Dependence => 2,
            ObligationKind::Effects => 3,
        }
    }
}

impl AnalysisReport {
    pub(crate) fn totals(&self) -> Totals {
        let mut totals = Totals::default();
        for function in &self.functions {
            for loop_report in &function.loops {
                totals.loops += 1;
                if loop_report.outside_coverage {
                    totals.outside_coverage += 1;
                }
                match loop_report.disposition {
                    Disposition::Gpu => totals.gpu += 1,
                    Disposition::GpuAfterGuard => totals.gpu_after_guard += 1,
                    Disposition::Cpu => {
                        totals.cpu += 1;

                        let decisive = loop_report
                            .obligations
                            .iter()
                            .find(|obligation| obligation.verdict == Verdict::ProvenIllegal)
                            .or_else(|| {
                                loop_report
                                    .obligations
                                    .iter()
                                    .find(|obligation| obligation.verdict == Verdict::Unknown)
                            });
                        if let Some(obligation) = decisive {
                            let slot = Totals::slot(obligation.kind);
                            if obligation.verdict == Verdict::ProvenIllegal {
                                totals.illegal[slot] += 1;
                            } else {
                                totals.unknown[slot] += 1;
                            }
                        }
                    }
                }
            }
        }
        totals
    }
}
