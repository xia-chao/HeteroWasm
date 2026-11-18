pub mod planning {
    use heterowasm_address::AddressRecovery;
    use heterowasm_bounds::LoopExtent;
    use heterowasm_cfg::ControlFlow;
    use heterowasm_legality::{Disposition, LoopLegal};
    use heterowasm_scev::ScalarEvolution;
    use heterowasm_trace::Trace;
    use waffle::entity::EntityRef;
    use waffle::{Block, Func, FuncDecl, Module};

    use crate::lower::lower;
    use crate::single::{loop_exit, offload_loop};
    use crate::types::{Dispatch, Kernel, LoopRegion};


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
        pub lower_failures: Vec<(String, String)>,
    }

    impl LoopPlanSet {

        pub fn skipped(&self) -> usize {
            self.skipped_legality + self.skipped_lower + self.skipped_gate + self.skipped_exit
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
            lower_failures: Vec::new(),
        };
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            for natural_loop in ControlFlow::analyze(body, trace).natural_loops(body, trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, trace);
                let recovery = AddressRecovery::analyze(body, &evolution, trace);
                let accesses: Vec<_> = recovery
                    .accesses()
                    .iter()
                    .filter(|access| natural_loop.blocks.contains(&access.block))
                    .cloned()
                    .collect();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
                let legality =
                    LoopLegal::judge(body, &natural_loop, &accesses, &evolution, module, trace);
                if !matches!(
                    legality.disposition(),
                    Disposition::Gpu | Disposition::GpuAfterGuard
                ) {
                    set.skipped_legality += 1;
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

pub use planning::{plan_gpu_loops, rewrite_with_plans, LoopPlanSet, OffloadPlan};
