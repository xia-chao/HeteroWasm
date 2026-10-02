use heterowasm_address::MemoryAccess;
use heterowasm_bounds::{BoundsAnalysis, LoopExtent, MemoryBounds};
use heterowasm_cfg::NaturalLoop;
use heterowasm_dependence::{DependenceAnalysis, LoopLegality};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{FunctionBody, Module};

use crate::combine::combine;
use crate::control::control_obligation;
use crate::dependence::dependence_obligation;
use crate::effects::effects_obligation;
use crate::memory::memory_obligation;
use crate::types::{verdict_name, GuardKind, LoopLegal};

impl LoopLegal {

    pub fn judge(
        body: &FunctionBody,
        natural_loop: &NaturalLoop,
        accesses: &[MemoryAccess],
        evolution: &ScalarEvolution,
        module: &Module<'_>,
        trace: &Trace,
    ) -> Self {
        let subject = format!("loop@{}", natural_loop.header.index());
        let _scope = trace.stage(Stage::Legality, subject.as_str());


        let memory_bounds = MemoryBounds::of(module);
        let extent = LoopExtent::analyze(body, natural_loop, evolution);
        let bounds =
            BoundsAnalysis::analyze(body, accesses, evolution, extent, &memory_bounds, trace);

        let dependence = DependenceAnalysis::analyze(accesses, evolution, trace);
        let obligations = vec![
            control_obligation(body, natural_loop, evolution),
            memory_obligation(accesses, bounds.proven()),
            dependence_obligation(dependence, evolution),
            effects_obligation(body, natural_loop),
        ];

        let mut guards = Vec::new();

        if matches!(
            dependence.legality(),
            LoopLegality::RequiresAliasGuard { .. }
        ) {
            guards.push(GuardKind::Alias);
        }

        if !accesses.is_empty() && bounds.proven() < accesses.len() {
            guards.push(GuardKind::Bounds);
        }

        for obligation in &obligations {
            trace
                .debug(Stage::Legality, "proof-obligation decision")
                .field("obligation", obligation.kind.as_str())
                .field("verdict", verdict_name(obligation.verdict))
                .field("detail", obligation.detail)
                .emit();
        }

        let (verdict, disposition) = combine(&obligations, &guards);

        trace
            .info(Stage::Legality, "legality decision complete")
            .field("verdict", verdict_name(verdict))
            .field("disposition", disposition.as_str())
            .field("obligations", obligations.len())
            .field("guards", guards.len())
            .emit();

        Self {
            verdict,
            disposition,
            obligations,
            guards,
            dependence: dependence.legality(),
            bounds_proven: bounds.proven(),
            bounds_total: accesses.len(),
        }
    }
}
