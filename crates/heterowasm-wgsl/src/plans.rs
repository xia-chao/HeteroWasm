pub mod planning {
    use heterowasm_address::{AccessKind, AddressRecovery, NormalizedAddress};
    use heterowasm_bounds::LoopExtent;
    use heterowasm_cfg::ControlFlow;
    use heterowasm_ir::Verdict;
    use heterowasm_legality::{Disposition, LoopLegal, Obligation, ObligationKind};
    use heterowasm_scev::ScalarEvolution;
    use heterowasm_trace::Trace;
    use std::collections::{BTreeMap, HashMap, HashSet};

    use waffle::entity::EntityRef;
    use waffle::{Block, Func, FuncDecl, Module, Operator, ValueDef};

    use crate::fusion::{from_plans::FusionInput, region_key, Collected, RegionUses};
    use crate::lower::lower;
    use crate::single::{loop_exit, offload_loop};
    use crate::types::{Dispatch, Kernel, LoopRegion};


    type FusionSeeds = HashMap<
        Func,
        Vec<(
            heterowasm_cfg::NaturalLoop,
            Option<LoopExtent>,
            Vec<(NormalizedAddress, AccessKind)>,
        )>,
    >;


    fn scan_operators(body: &waffle::FunctionBody, members: &HashSet<Block>) -> (bool, bool) {
        let mut side_effects = false;
        let mut has_store = false;
        for value in body.values.iter() {
            if !members.contains(&body.value_blocks[value]) {
                continue;
            }
            let Some(ValueDef::Operator(operator, _, _)) = body.values.get(value) else {
                continue;
            };
            if matches!(
                operator,
                Operator::Call { .. }
                    | Operator::CallIndirect { .. }
                    | Operator::MemoryCopy { .. }
                    | Operator::MemoryFill { .. }
                    | Operator::MemoryGrow { .. }
                    | Operator::GlobalSet { .. }
                    | Operator::TableSet { .. }
                    | Operator::TableGrow { .. }
                    | Operator::Unreachable
            ) {
                side_effects = true;
            }
            if matches!(
                operator,
                Operator::I32Store { .. }
                    | Operator::I64Store { .. }
                    | Operator::I32Store8 { .. }
                    | Operator::I32Store16 { .. }
                    | Operator::I64Store8 { .. }
                    | Operator::I64Store16 { .. }
                    | Operator::I64Store32 { .. }
            ) {
                has_store = true;
            }
        }
        (side_effects, has_store)
    }


    pub(crate) fn has_side_effects(body: &waffle::FunctionBody, members: &HashSet<Block>) -> bool {
        scan_operators(body, members).0
    }


    pub(crate) fn cheaply_unrewardable(
        body: &waffle::FunctionBody,
        members: &HashSet<Block>,
    ) -> bool {
        let (side_effects, has_store) = scan_operators(body, members);
        side_effects || !has_store
    }


    #[derive(Clone)]
    pub struct OffloadPlan {
        pub func: Func,
        pub header: Block,
        pub exit: Block,
        pub members: Vec<Block>,
        pub kernel: Kernel,
    }


    pub struct LoopPlanSet {
        pub eligible: Vec<OffloadPlan>,
        pub rewritable: Vec<OffloadPlan>,
        pub skipped_legality: usize,
        pub skipped_lower: usize,
        pub skipped_gate: usize,
        pub skipped_exit: usize,
        pub skipped_overlap: usize,

        pub skipped_illegal: [usize; 4],
        pub skipped_unknown: [usize; 4],

        pub skipped_side_effects: usize,
        pub lower_failures: Vec<(String, String)>,

        pub fusion_candidates: HashSet<Func>,

        pub fusion_inputs: FusionInput,
    }


    fn decisive(obligations: &[Obligation]) -> Option<(usize, bool)> {
        let chosen = obligations
            .iter()
            .find(|obligation| obligation.verdict == Verdict::ProvenIllegal)
            .or_else(|| {
                obligations
                    .iter()
                    .find(|obligation| obligation.verdict == Verdict::Unknown)
            })?;
        let slot = match chosen.kind {
            ObligationKind::Control => 0,
            ObligationKind::Memory => 1,
            ObligationKind::Dependence => 2,
            ObligationKind::Effects => 3,
        };
        Some((slot, chosen.verdict == Verdict::ProvenIllegal))
    }

    impl LoopPlanSet {

        pub fn skipped(&self) -> usize {
            self.skipped_legality + self.skipped_lower + self.skipped_gate + self.skipped_exit
        }


        pub fn analysis_funcs(&self) -> Vec<Func> {
            let mut funcs: HashSet<Func> = HashSet::new();
            for plan in self.eligible.iter().chain(self.rewritable.iter()) {
                funcs.insert(plan.func);
            }
            funcs.extend(self.fusion_inputs.by_func.keys().copied());
            funcs.extend(self.fusion_candidates.iter().copied());
            let mut ordered: Vec<Func> = funcs.into_iter().collect();
            ordered.sort_by_key(|func| func.index());
            ordered
        }


        pub fn shifted(mut self, shift: usize) -> LoopPlanSet {
            if shift == 0 {
                return self;
            }
            let bumped = |func: Func| Func::new(func.index() + shift);
            for plan in &mut self.eligible {
                plan.func = bumped(plan.func);
            }
            for plan in &mut self.rewritable {
                plan.func = bumped(plan.func);
            }
            self.fusion_candidates = self
                .fusion_candidates
                .iter()
                .map(|func| bumped(*func))
                .collect();
            let inputs = std::mem::take(&mut self.fusion_inputs);
            self.fusion_inputs = inputs.into_shifted(shift);
            self
        }


        pub fn fits_module(&self, module: &Module<'_>, trace: &Trace) -> bool {
            let mut by_func: HashMap<Func, Vec<&OffloadPlan>> = HashMap::new();
            for plan in self.eligible.iter().chain(self.rewritable.iter()) {
                by_func.entry(plan.func).or_default().push(plan);
            }
            for (func, plans) in by_func {
                let Some(body) = module.funcs.get(func).and_then(FuncDecl::body) else {
                    return false;
                };
                let loops = ControlFlow::analyze(body, trace).natural_loops(body, trace);
                for plan in plans {
                    let Some(natural) = loops.iter().find(|natural| natural.header == plan.header)
                    else {
                        return false;
                    };
                    let members: HashSet<Block> = natural.blocks.iter().copied().collect();
                    let planned: HashSet<Block> = plan.members.iter().copied().collect();
                    if members != planned {
                        return false;
                    }
                }
            }

            for (func, collected) in &self.fusion_inputs.by_func {
                let Some(body) = module.funcs.get(*func).and_then(FuncDecl::body) else {
                    return false;
                };
                let loops = ControlFlow::analyze(body, trace).natural_loops(body, trace);
                for (planned, _, _) in collected {
                    let Some(natural) = loops
                        .iter()
                        .find(|natural| natural.header == planned.header)
                    else {
                        return false;
                    };
                    if natural.blocks != planned.blocks {
                        return false;
                    }
                }
            }
            true
        }
    }


    pub fn plan_gpu_loops(module: &Module<'_>, trace: &Trace) -> LoopPlanSet {
        let mut set = LoopPlanSet {
            eligible: Vec::new(),
            rewritable: Vec::new(),
            skipped_legality: 0,
            skipped_lower: 0,
            skipped_gate: 0,
            skipped_exit: 0,
            skipped_overlap: 0,
            skipped_illegal: [0; 4],
            skipped_unknown: [0; 4],
            skipped_side_effects: 0,
            lower_failures: Vec::new(),
            fusion_candidates: HashSet::new(),
            fusion_inputs: FusionInput::default(),
        };
        let mut lowerable_by_func: HashMap<Func, usize> = HashMap::new();

        let mut fusion_seeds: FusionSeeds = HashMap::new();
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
                let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();
                if has_side_effects(body, &members) {

                    set.skipped_legality += 1;
                    set.skipped_side_effects += 1;
                    continue;
                }
                let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
                let recovery =
                    AddressRecovery::analyze_loop(body, &evolution, &natural_loop.blocks, trace);
                let accesses: Vec<_> = recovery.accesses().to_vec();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);

                let normalized: Vec<(NormalizedAddress, AccessKind)> = accesses
                    .iter()
                    .filter_map(|access| {
                        access
                            .normalized(&evolution)
                            .map(|address| (address, access.kind))
                    })
                    .collect();
                fusion_seeds.entry(func).or_default().push((
                    natural_loop.clone(),
                    extent,
                    normalized,
                ));
                let legality =
                    LoopLegal::judge(body, &natural_loop, &accesses, &evolution, module, trace);
                if !matches!(
                    legality.disposition(),
                    Disposition::Gpu | Disposition::GpuAfterGuard
                ) {
                    set.skipped_legality += 1;
                    match decisive(legality.obligations()) {
                        Some((slot, true)) => set.skipped_illegal[slot] += 1,
                        Some((slot, false)) => set.skipped_unknown[slot] += 1,

                        None => set.skipped_unknown[0] += 1,
                    }
                    continue;
                }
                let kernel = match lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    trace,
                ) {
                    Ok(kernel) => kernel,
                    Err(error) => {
                        let name = match &module.funcs[func] {
                            FuncDecl::Body(_, name, _) => name.clone(),
                            _ => String::new(),
                        };
                        set.skipped_lower += 1;
                        set.lower_failures.push((name, error.to_string()));
                        continue;
                    }
                };
                *lowerable_by_func.entry(func).or_insert(0) += 1;
                if kernel.dispatch.fixed_trip_is_slower_than_cpu()
                    || kernel.shader_multiplies_still_slower_than_cpu()
                {
                    set.skipped_gate += 1;
                    continue;
                }
                let Ok(exit) = loop_exit(body, &natural_loop.blocks) else {
                    set.skipped_exit += 1;
                    continue;
                };
                set.eligible.push(OffloadPlan {
                    func,
                    header: natural_loop.header,
                    exit,
                    members: natural_loop.blocks.clone(),
                    kernel,
                });
            }
        }
        set.fusion_candidates = lowerable_by_func
            .into_iter()
            .filter(|(_, count)| *count >= 2)
            .map(|(func, _)| func)
            .collect();

        let candidates: Vec<Func> = set.fusion_candidates.iter().copied().collect();
        for func in candidates {
            let Some(seeds) = fusion_seeds.remove(&func) else {
                continue;
            };
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let mut collected: Vec<Collected> = Vec::with_capacity(seeds.len());
            for (index, (natural_loop, extent, normalized)) in seeds.into_iter().enumerate() {
                let mut uses: RegionUses = BTreeMap::new();
                for (address, kind) in normalized {
                    uses.entry(region_key(body, &address))
                        .or_default()
                        .push((index, kind == AccessKind::Store));
                }
                collected.push((natural_loop, extent, uses));
            }
            set.fusion_inputs.by_func.insert(func, collected);
        }
        for plan in &set.eligible {
            let overlaps = set.rewritable.iter().any(|existing: &OffloadPlan| {
                existing.func == plan.func
                    && plan
                        .members
                        .iter()
                        .any(|block| existing.members.contains(block))
            });
            if overlaps {
                set.skipped_overlap += 1;
                continue;
            }
            set.rewritable.push(plan.clone());
        }
        set
    }


    pub fn rewrite_with_plans(
        module: &mut Module<'_>,
        plans: &[OffloadPlan],
        trace: &Trace,
    ) -> Result<Vec<(usize, usize)>, String> {
        if plans.is_empty() {
            return Err("no extractable loop found".to_string());
        }
        let mut replaced = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            let selector =
                u32::try_from(index).map_err(|_| "kernel index exceeds u32".to_string())?;
            if let Err(reason) = offload_loop(
                module,
                plan.func,
                LoopRegion {
                    header: plan.header,
                    exit: plan.exit,
                    members: &plan.members,
                },
                &plan.kernel.fields,
                selector,
                runtime_trip(&plan.kernel),
            ) {
                if replaced.is_empty() {
                    return Err(reason);
                }
                trace
                    .info(heterowasm_trace::Stage::Wgsl, "rewrite stopped")
                    .field("reason", reason)
                    .field("replaced", replaced.len())
                    .emit();
                break;
            }
            replaced.push((plan.func.index(), plan.header.index()));
        }
        Ok(replaced)
    }


    fn runtime_trip(kernel: &Kernel) -> Option<crate::region::RuntimeTrip> {
        let (limit, mask) = match kernel.dispatch {
            Dispatch::FromField(slot) => (slot, u32::MAX),
            Dispatch::FromFieldMask(slot, mask) => (slot, mask),
            _ => return None,
        };
        if let Some(start) = kernel.index_field {
            let Ok(stride) = u32::try_from(kernel.max_stride_bytes) else {
                return None;
            };
            if stride < 4 || stride % 4 != 0 {
                return None;
            }
            let guard = format!("params.p{start} + 4u * gid.x >= params.p{limit}");
            if !kernel.source.contains(&guard) {
                return None;
            }
            return Some(crate::region::RuntimeTrip {
                start_field: Some(start),
                limit_field: limit,
                stride_bytes: stride,
                mask,
            });
        }
        Some(crate::region::RuntimeTrip {
            start_field: None,
            limit_field: limit,
            stride_bytes: 1,
            mask,
        })
    }
}

pub(crate) use planning::cheaply_unrewardable;
pub use planning::{plan_gpu_loops, rewrite_with_plans, LoopPlanSet, OffloadPlan};
