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
pub use lower::lower;
pub use manifest::artifact;
pub use plans::{plan_gpu_loops, rewrite_with_plans, LoopPlanSet, OffloadPlan};
pub use single::{plan_gpu_arity, rewrite_module_for_gpu};
pub use two_dispatch::rewrite_module_two_dispatches;
pub use types::*;


#[cfg(test)]
mod tests;
