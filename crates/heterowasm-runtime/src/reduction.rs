pub mod folding {

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ReductionOp {
        Add,
        And,
        Or,
        Xor,
    }

    impl ReductionOp {

        pub fn parse(name: &str) -> Option<Self> {
            match name {
                "add" => Some(ReductionOp::Add),
                "and" => Some(ReductionOp::And),
                "or" => Some(ReductionOp::Or),
                "xor" => Some(ReductionOp::Xor),
                _ => None,
            }
        }


        pub fn as_str(self) -> &'static str {
            match self {
                ReductionOp::Add => "add",
                ReductionOp::And => "and",
                ReductionOp::Or => "or",
                ReductionOp::Xor => "xor",
            }
        }


        pub fn identity(self) -> u32 {
            match self {
                ReductionOp::Add | ReductionOp::Or | ReductionOp::Xor => 0,
                ReductionOp::And => u32::MAX,
            }
        }


        pub fn combine(self, left: u32, right: u32) -> u32 {
            match self {
                ReductionOp::Add => left.wrapping_add(right),
                ReductionOp::And => left & right,
                ReductionOp::Or => left | right,
                ReductionOp::Xor => left ^ right,
            }
        }
    }


    pub fn fold(op: ReductionOp, partials: &[u32]) -> u32 {
        partials.iter().fold(op.identity(), |accumulator, partial| {
            op.combine(accumulator, *partial)
        })
    }
}

pub use folding::{fold, ReductionOp};
