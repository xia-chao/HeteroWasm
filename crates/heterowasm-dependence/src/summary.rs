use heterowasm_address::{AccessKind, AddressConclusion, MemoryAccess};
use heterowasm_scev::ScalarEvolution;

use crate::consts::NOT_SINGLE_TERM;
use crate::types::AccessSummary;


pub fn summarize(access: &MemoryAccess, evolution: &ScalarEvolution) -> Option<AccessSummary> {
    let normalized = access.normalized(evolution)?;
    Some(AccessSummary {
        value: access.value,
        block: access.block,
        kind: access.kind,
        bytes: access.bytes,
        base: normalized.base,
        stride: normalized.stride,
        offset: normalized.offset,
    })
}

pub(crate) fn summarize_reason(access: &MemoryAccess) -> &'static str {
    match &access.conclusion {
        AddressConclusion::Unknown { reason } => reason,

        AddressConclusion::Affine(_) => NOT_SINGLE_TERM,
    }
}

pub(crate) fn kind_name(kind: AccessKind) -> &'static str {
    match kind {
        AccessKind::Load => "load",
        AccessKind::Store => "store",
    }
}
