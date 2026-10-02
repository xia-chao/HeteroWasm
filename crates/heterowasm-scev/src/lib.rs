pub mod affine;
pub mod analyze;
pub mod canonical;
pub mod error;
pub mod invariance;
pub mod memory_recurrence;
pub mod types;

pub use canonical::{canonical_value, constant_value};
pub use invariance::LoopInvariance;
pub use types::*;


#[cfg(test)]
mod tests;
