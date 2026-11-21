pub mod conformance;
pub mod dispatch;
pub mod error;
pub mod fused;
pub mod fusion;
pub mod inject;
pub mod ir;
pub mod live_out;
pub mod lower;
pub mod manifest;
pub mod plans;
pub mod region;
pub mod single;
pub mod two_dispatch;
pub mod types;

pub use dispatch::decide_dispatch;
pub use error::LowerError;
pub use fused::{rewrite_module_fused, rewrite_module_fused_from};
pub use fusion::{plan_gpu_fusion, plan_gpu_fusion_from, FusionInput};
pub use inject::inject_dispatch_import;
pub use lower::{lower, lower_with_mode};
pub use manifest::artifact;
pub use plans::{
    plan_gpu_loops, plan_gpu_loops_with, rewrite_with_plans, LoopPlanSet, OffloadPlan,
};
pub use single::{plan_gpu_arity, rewrite_module_for_gpu};
pub use two_dispatch::rewrite_module_two_dispatches;
pub use types::*;
pub use types::{NumericMode, F32_ATOL, F32_RTOL, F32_ULP_BOUND};


#[cfg(test)]
mod tests;
