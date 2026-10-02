#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    ProvenLegal,
    ProvenIllegal,
    Unknown,
}

impl Verdict {

    pub fn permits_gpu_transform(self) -> bool {
        matches!(self, Verdict::ProvenLegal)
    }


    pub fn as_str(self) -> &'static str {
        match self {
            Verdict::ProvenLegal => "proven_legal",
            Verdict::ProvenIllegal => "proven_illegal",
            Verdict::Unknown => "unknown",
        }
    }
}
