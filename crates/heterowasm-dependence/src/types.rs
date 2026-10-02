use heterowasm_address::AccessKind;
use waffle::{Block, Value};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopLegality {
    Parallelizable,
    RequiresAliasGuard { pairs: usize },
    SerialOnly { reason: &'static str },
    Unknown { reason: &'static str },
}

impl LoopLegality {

    pub fn as_str(self) -> &'static str {
        match self {
            LoopLegality::Parallelizable => "parallelizable",
            LoopLegality::RequiresAliasGuard { .. } => "requires_alias_guard",
            LoopLegality::SerialOnly { .. } => "serial_only",
            LoopLegality::Unknown { .. } => "unknown",
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairDependence {
    None,
    Overlap,
    NeedsAliasCheck,
    Unanalyzable { reason: &'static str },
}

impl PairDependence {

    pub fn as_str(self) -> &'static str {
        match self {
            PairDependence::None => "none",
            PairDependence::Overlap => "overlap",
            PairDependence::NeedsAliasCheck => "needs_alias_check",
            PairDependence::Unanalyzable { .. } => "unanalyzable",
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessSummary {
    pub value: Value,
    pub block: Block,
    pub kind: AccessKind,
    pub bytes: u32,
    pub base: Vec<(Value, i64)>,
    pub stride: i64,
    pub offset: i64,
}
