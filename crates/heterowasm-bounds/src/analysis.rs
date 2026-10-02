use std::collections::HashSet;

use heterowasm_address::MemoryAccess;
use heterowasm_cfg::NaturalLoop;
use heterowasm_scev::{canonical_value, constant_value, ScalarEvolution};
use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody, Operator, Terminator, Value, ValueDef};

use crate::consts::{BASED_ON_PARAMETER, NOT_AFFINE, UNBOUNDED_MEMORY, UNKNOWN_EXTENT};
use crate::types::{BoundsConclusion, LoopExtent, MemoryBounds};


#[derive(Debug, Clone, Default)]
pub struct BoundsAnalysis {
    conclusions: Vec<BoundsConclusion>,
}

impl BoundsAnalysis {

    pub fn analyze(
        body: &FunctionBody,
        accesses: &[MemoryAccess],
        evolution: &ScalarEvolution,
        extent: Option<LoopExtent>,
        memory: &MemoryBounds,
        trace: &Trace,
    ) -> Self {
        let _scope = trace.stage(Stage::Bounds, "loop");

        let mut conclusions = Vec::with_capacity(accesses.len());
        for access in accesses {
            let conclusion = conclude(body, access, evolution, extent, memory);
            trace
                .debug(Stage::Bounds, "access bounds decision")
                .field("block", access.block.index())
                .field("bytes", i64::from(access.bytes))
                .field("conclusion", conclusion.as_str())
                .field("reason", conclusion.reason().unwrap_or(""))
                .emit();
            conclusions.push(conclusion);
        }

        let proven = conclusions
            .iter()
            .filter(|conclusion| matches!(conclusion, BoundsConclusion::Proven))
            .count();
        trace
            .info(Stage::Bounds, "loop bounds decision complete")
            .field("accesses", accesses.len())
            .field("proven", proven)
            .field(
                "maximum_bytes",
                i64::try_from(memory.maximum_bytes.unwrap_or(u64::MAX)).unwrap_or(i64::MAX),
            )
            .emit();

        Self { conclusions }
    }


    pub fn proven(&self) -> usize {
        self.conclusions
            .iter()
            .filter(|conclusion| matches!(conclusion, BoundsConclusion::Proven))
            .count()
    }


    pub fn needs_guard(&self) -> usize {
        self.conclusions
            .iter()
            .filter(|conclusion| matches!(conclusion, BoundsConclusion::NeedsGuard { .. }))
            .count()
    }


    pub fn unprovable(&self) -> usize {
        self.conclusions
            .iter()
            .filter(|conclusion| matches!(conclusion, BoundsConclusion::Unprovable { .. }))
            .count()
    }


    pub fn conclusions(&self) -> &[BoundsConclusion] {
        &self.conclusions
    }
}

fn conclude(
    body: &FunctionBody,
    access: &MemoryAccess,
    evolution: &ScalarEvolution,
    extent: Option<LoopExtent>,
    memory: &MemoryBounds,
) -> BoundsConclusion {
    let Some(normalized) = access.normalized(evolution) else {
        return BoundsConclusion::Unprovable { reason: NOT_AFFINE };
    };


    let Some(base) = constant_base(body, &normalized.base) else {
        return BoundsConclusion::NeedsGuard {
            reason: BASED_ON_PARAMETER,
        };
    };

    let Some((start, limit)) = extent.and_then(LoopExtent::trip) else {
        return BoundsConclusion::NeedsGuard {
            reason: UNKNOWN_EXTENT,
        };
    };

    let Some(maximum) = memory.maximum_bytes else {
        return BoundsConclusion::NeedsGuard {
            reason: UNBOUNDED_MEMORY,
        };
    };

    let width = i128::from(access.bytes);
    let stride = i128::from(normalized.stride);
    let offset = i128::from(normalized.offset);
    let base = i128::from(base);

    let first = base + stride * i128::from(start) + offset;
    let last_index = i128::from(limit) - 1;
    let last = base + stride * last_index + offset + width;

    let (lower, upper) = if first <= last {
        (first, last)
    } else {
        (last, first)
    };

    if lower >= 0 && upper <= i128::from(maximum) {
        BoundsConclusion::Proven
    } else {
        BoundsConclusion::NeedsGuard {
            reason: "address range may exceed memory upper bound",
        }
    }
}


fn constant_base(body: &FunctionBody, base: &[(Value, i64)]) -> Option<i64> {
    let mut total: i64 = 0;
    for (value, coefficient) in base {
        total = total.saturating_add(coefficient.saturating_mul(constant_value(body, *value)?));
    }
    Some(total)
}


fn limit_operand(
    body: &FunctionBody,
    condition: Value,
    evolution: &ScalarEvolution,
) -> Option<Value> {
    let canonical = canonical_value(body, condition);
    let ValueDef::Operator(operator, args, _) = body.values.get(canonical)? else {
        return None;
    };
    if !is_comparison(operator) {
        return None;
    }
    let operands: &[Value] = &body.arg_pool[*args];

    if !operands
        .iter()
        .any(|operand| evolution.depends_on_induction(body, *operand, 8))
    {
        return None;
    }
    operands
        .iter()
        .copied()
        .find(|operand| !evolution.depends_on_induction(body, *operand, 8))
}

fn induction_start(body: &FunctionBody, evolution: &ScalarEvolution) -> Option<i64> {
    constant_value(body, induction_start_value(evolution)?)
}


fn induction_start_value(evolution: &ScalarEvolution) -> Option<Value> {
    Some(evolution.induction_variables().first()?.initial)
}

fn is_comparison(operator: &Operator) -> bool {
    matches!(
        operator,
        Operator::I32Eq
            | Operator::I32Ne
            | Operator::I32LtS
            | Operator::I32LtU
            | Operator::I32GtS
            | Operator::I32GtU
            | Operator::I32LeS
            | Operator::I32LeU
            | Operator::I32GeS
            | Operator::I32GeU
            | Operator::I64Eq
            | Operator::I64Ne
            | Operator::I64LtS
            | Operator::I64LtU
            | Operator::I64GtS
            | Operator::I64GtU
            | Operator::I64LeS
            | Operator::I64LeU
            | Operator::I64GeS
            | Operator::I64GeU
    )
}

impl LoopExtent {

    pub fn analyze(
        body: &FunctionBody,
        natural_loop: &NaturalLoop,
        evolution: &ScalarEvolution,
    ) -> Option<Self> {
        let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();

        for &block in &natural_loop.blocks {
            let Some(def) = body.blocks.get(block) else {
                continue;
            };
            let Terminator::CondBr {
                cond,
                if_true,
                if_false,
            } = &def.terminator
            else {
                continue;
            };
            let is_exit = !members.contains(&if_true.block) || !members.contains(&if_false.block);
            if !is_exit {
                continue;
            }
            let Some(limit) = limit_operand(body, *cond, evolution) else {
                continue;
            };
            return Some(Self {
                start: induction_start(body, evolution),
                limit: constant_value(body, limit),
                limit_value: Some(limit),
                start_value: induction_start_value(evolution),
            });
        }

        None
    }


    pub fn trip(self) -> Option<(i64, i64)> {
        let start = self.start?;
        let limit = self.limit?;
        if limit <= start {
            return None;
        }
        Some((start, limit))
    }
}
