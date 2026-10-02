use heterowasm_scev::{AffineForm, ScalarEvolution};
use waffle::entity::EntityRef;
use waffle::{Block, Value};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Load,
    Store,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddressConclusion {
    Affine(AffineForm),
    Unknown { reason: &'static str },
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryAccess {
    pub value: Value,
    pub block: Block,
    pub kind: AccessKind,
    pub bytes: u32,
    pub instruction_offset: i64,
    pub conclusion: AddressConclusion,
}

impl MemoryAccess {

    pub fn is_affine(&self) -> bool {
        matches!(self.conclusion, AddressConclusion::Affine(_))
    }


    pub fn affine(&self) -> Option<&AffineForm> {
        match &self.conclusion {
            AddressConclusion::Affine(form) => Some(form),
            AddressConclusion::Unknown { .. } => None,
        }
    }


    pub fn normalized(&self, evolution: &ScalarEvolution) -> Option<NormalizedAddress> {
        let AddressConclusion::Affine(form) = &self.conclusion else {
            return None;
        };
        NormalizedAddress::of(form, self.instruction_offset, evolution)
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedAddress {

    pub base: Vec<(Value, i64)>,
    pub stride: i64,
    pub offset: i64,
    pub initial_in_base: bool,
}

impl NormalizedAddress {

    pub fn of(
        form: &AffineForm,
        instruction_offset: i64,
        evolution: &ScalarEvolution,
    ) -> Option<Self> {
        let offset = form.offset.saturating_add(instruction_offset);

        match (form.invariants.as_slice(), form.induction.as_slice()) {

            (invariants, [(_, stride)]) if !invariants.is_empty() => {
                let base = sorted_base(invariants)?;
                Some(NormalizedAddress {
                    base,
                    stride: *stride,
                    offset,
                    initial_in_base: false,
                })
            }
            ([], [(induction, coefficient)]) => {
                let variable = evolution
                    .induction_variables()
                    .iter()
                    .find(|variable| variable.value == *induction)?;

                Some(NormalizedAddress {
                    base: vec![(variable.initial, *coefficient)],
                    stride: coefficient.saturating_mul(variable.step),
                    offset,
                    initial_in_base: true,
                })
            }
            (invariants, []) if !invariants.is_empty() => {
                let base = sorted_base(invariants)?;
                Some(NormalizedAddress {
                    base,
                    stride: 0,
                    offset,
                    initial_in_base: false,
                })
            }
            _ => None,
        }
    }
}


fn sorted_base(invariants: &[(Value, i64)]) -> Option<Vec<(Value, i64)>> {
    let mut base: Vec<(Value, i64)> = invariants
        .iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .copied()
        .collect();
    if base.is_empty() {
        return None;
    }
    base.sort_by_key(|(value, _)| value.index());
    Some(base)
}
