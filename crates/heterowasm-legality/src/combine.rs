use heterowasm_ir::Verdict;

use crate::types::{Disposition, GuardKind, Obligation, ObligationKind};


pub(crate) fn combine(obligations: &[Obligation], guards: &[GuardKind]) -> (Verdict, Disposition) {
    if obligations
        .iter()
        .any(|obligation| obligation.verdict == Verdict::ProvenIllegal)
    {
        return (Verdict::ProvenIllegal, Disposition::Cpu);
    }


    let unrecoverable = obligations.iter().any(|obligation| {
        obligation.verdict == Verdict::Unknown && obligation.kind != ObligationKind::Dependence
    });
    if unrecoverable {
        return (Verdict::Unknown, Disposition::Cpu);
    }

    if obligations
        .iter()
        .any(|obligation| obligation.verdict == Verdict::Unknown)
    {

        return if guards.contains(&GuardKind::Alias) {
            (Verdict::Unknown, Disposition::GpuAfterGuard)
        } else {
            (Verdict::Unknown, Disposition::Cpu)
        };
    }

    if guards.is_empty() {
        (Verdict::ProvenLegal, Disposition::Gpu)
    } else {
        (Verdict::ProvenLegal, Disposition::GpuAfterGuard)
    }
}
