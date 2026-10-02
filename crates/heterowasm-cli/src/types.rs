use heterowasm_ir::Verdict;
use heterowasm_legality::{Disposition, GuardKind, Obligation};


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
                    Disposition::Cpu => totals.cpu += 1,
                }
            }
        }
        totals
    }
}
