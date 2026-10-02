#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Synthetic,
    SourceCompiled,
    Negative,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expectation {
    Parallelizable,
    RequiresGuard { guard: &'static str },
    MustReject { reason: &'static str },
    NoCandidate { reason: &'static str },
    Undetermined,
}

impl Expectation {

    pub fn as_str(self) -> &'static str {
        match self {
            Expectation::Parallelizable => "parallelizable",
            Expectation::RequiresGuard { .. } => "requires_guard",
            Expectation::MustReject { .. } => "must_reject",
            Expectation::NoCandidate { .. } => "no_candidate",
            Expectation::Undetermined => "undetermined",
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub category: Category,
    pub expression: &'static str,
    pub expectation: Expectation,
    pub wat: &'static str,
}

impl Case {

    pub fn to_wasm(&self) -> Result<Vec<u8>, wat::Error> {
        wat::parse_str(self.wat)
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    C,
}


#[derive(Debug, Clone)]
pub struct CompiledCase {
    pub language: Language,
    pub opt_level: &'static str,
    pub bytes: Vec<u8>,
}
