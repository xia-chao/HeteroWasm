use heterowasm_address::MemoryAccess;
use heterowasm_ir::Verdict;

use crate::types::{Obligation, ObligationKind};


pub(crate) fn memory_obligation(accesses: &[MemoryAccess], proven: usize) -> Obligation {
    if accesses.is_empty() {
        return Obligation {
            kind: ObligationKind::Memory,
            verdict: Verdict::ProvenLegal,
            detail: "this loop has no memory accesses",
        };
    }
    if accesses.iter().any(|access| !access.is_affine()) {
        return Obligation {
            kind: ObligationKind::Memory,
            verdict: Verdict::Unknown,
            detail: "there are addresses that cannot be recovered as affine",
        };
    }


    let misaligned = accesses
        .iter()
        .any(|access| access.bytes != 4 || access.instruction_offset.rem_euclid(4) != 0);
    if misaligned {
        return Obligation {
            kind: ObligationKind::Memory,
            verdict: Verdict::ProvenIllegal,
            detail: "access width or offset violates 4-byte alignment policy (spec §16)",
        };
    }


    if proven == accesses.len() {
        return Obligation {
            kind: ObligationKind::Memory,
            verdict: Verdict::ProvenLegal,
            detail: "address is affine, width/alignment compliant, and bounds proven statically",
        };
    }

    Obligation {
        kind: ObligationKind::Memory,
        verdict: Verdict::ProvenLegal,
        detail: "address is affine and width/alignment compliant (bounds deferred to runtime guard, see §15)",
    }
}
