use waffle::{Module, Value, WASM_PAGE};


#[derive(Debug, Clone, Copy)]
pub struct MemoryBounds {
    pub maximum_bytes: Option<u64>,
    pub initial_bytes: u64,
}

impl MemoryBounds {

    pub fn of(module: &Module<'_>) -> Self {
        let Some((_, data)) = module.memories.entries().next() else {
            return Self {
                maximum_bytes: Some(0),
                initial_bytes: 0,
            };
        };

        let page = u64::try_from(WASM_PAGE).unwrap_or(u64::MAX);
        Self {
            maximum_bytes: data.maximum_pages.map(|pages| {
                u64::try_from(pages)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(page)
            }),
            initial_bytes: u64::try_from(data.initial_pages)
                .unwrap_or(u64::MAX)
                .saturating_mul(page),
        }
    }
}


#[derive(Debug, Clone, Copy, Default)]
pub struct LoopExtent {
    pub start: Option<i64>,
    pub limit: Option<i64>,

    pub limit_value: Option<Value>,

    pub start_value: Option<Value>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundsConclusion {
    Proven,
    NeedsGuard { reason: &'static str },
    Unprovable { reason: &'static str },
}

impl BoundsConclusion {

    pub fn as_str(self) -> &'static str {
        match self {
            BoundsConclusion::Proven => "proven",
            BoundsConclusion::NeedsGuard { .. } => "needs_guard",
            BoundsConclusion::Unprovable { .. } => "unprovable",
        }
    }


    pub fn reason(self) -> Option<&'static str> {
        match self {
            BoundsConclusion::Proven => None,
            BoundsConclusion::NeedsGuard { reason } | BoundsConclusion::Unprovable { reason } => {
                Some(reason)
            }
        }
    }
}
