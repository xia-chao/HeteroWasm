use heterowasm_address::{AccessKind, MemoryAccess};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;

use crate::consts::{NON_AFFINE_ADDRESS, OUTSIDE_COVERAGE, OVERLAPPING_ACCESS};
use crate::pair::test_pair;
use crate::summary::{kind_name, summarize, summarize_reason};
use crate::types::{LoopLegality, PairDependence};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DependenceAnalysis {
    legality: LoopLegality,
    pairs_checked: usize,
}

impl DependenceAnalysis {

    pub fn analyze(accesses: &[MemoryAccess], evolution: &ScalarEvolution, trace: &Trace) -> Self {
        let _scope = trace.stage(Stage::Dependence, "loop");


        if evolution.outside_coverage() {
            trace
                .warn(
                    Stage::Dependence,
                    "loop is out of coverage; skip dependence decision",
                )
                .field("accesses", accesses.len())
                .field("reason", OUTSIDE_COVERAGE)
                .emit();
            return Self {
                legality: LoopLegality::Unknown {
                    reason: OUTSIDE_COVERAGE,
                },
                pairs_checked: 0,
            };
        }

        let mut summaries = Vec::new();
        let mut unanalyzable = 0_usize;
        for access in accesses {
            match summarize(access, evolution) {
                Some(summary) => summaries.push(summary),
                None => {
                    unanalyzable += 1;
                    trace
                        .debug(Stage::Dependence, "access cannot enter dependence testing")
                        .field("block", access.block.index())
                        .field("kind", kind_name(access.kind))
                        .field("bytes", i64::from(access.bytes))
                        .field("reason", summarize_reason(access))
                        .emit();
                }
            }
        }

        let mut overlapping = 0_usize;
        let mut needs_alias = 0_usize;
        let mut pairs_checked = 0_usize;

        for (index, first) in summaries.iter().enumerate() {
            for second in summaries.iter().skip(index + 1) {

                if first.kind == AccessKind::Load && second.kind == AccessKind::Load {
                    continue;
                }
                pairs_checked += 1;

                let verdict = test_pair(first, second);
                match verdict {
                    PairDependence::Overlap => overlapping += 1,
                    PairDependence::NeedsAliasCheck => needs_alias += 1,
                    PairDependence::None | PairDependence::Unanalyzable { .. } => {}
                }

                trace
                    .debug(Stage::Dependence, "access-pair dependence decision")
                    .field("first_block", first.block.index())
                    .field("second_block", second.block.index())
                    .field("first_kind", kind_name(first.kind))
                    .field("second_kind", kind_name(second.kind))
                    .field("same_base", first.base == second.base)
                    .field("first_stride", first.stride)
                    .field("second_stride", second.stride)
                    .field("offset_delta", second.offset.saturating_sub(first.offset))
                    .field("verdict", verdict.as_str())
                    .emit();
            }
        }

        let legality = if overlapping > 0 {
            LoopLegality::SerialOnly {
                reason: OVERLAPPING_ACCESS,
            }
        } else if unanalyzable > 0 {
            LoopLegality::Unknown {
                reason: NON_AFFINE_ADDRESS,
            }
        } else if needs_alias > 0 {
            LoopLegality::RequiresAliasGuard { pairs: needs_alias }
        } else {
            LoopLegality::Parallelizable
        };

        trace
            .info(Stage::Dependence, "loop dependence decision complete")
            .field("accesses", accesses.len())
            .field("pairs_checked", pairs_checked)
            .field("overlapping", overlapping)
            .field("needs_alias_check", needs_alias)
            .field("unanalyzable", unanalyzable)
            .field("legality", legality.as_str())
            .emit();

        Self {
            legality,
            pairs_checked,
        }
    }


    pub fn legality(self) -> LoopLegality {
        self.legality
    }


    pub fn pairs_checked(self) -> usize {
        self.pairs_checked
    }
}
