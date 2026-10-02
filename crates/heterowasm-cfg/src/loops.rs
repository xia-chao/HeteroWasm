use std::collections::{HashMap, HashSet};

use heterowasm_trace::{Stage, Trace};
use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody};

use crate::dominators::{compute_immediate_dominators, reverse_postorder};
use crate::types::{ControlFlow, NaturalLoop};

impl ControlFlow {

    pub fn analyze(body: &FunctionBody, trace: &Trace) -> Self {
        let _scope = trace.stage(Stage::ControlFlow, "function");
        let count = body.blocks.len();
        let reverse_postorder = reverse_postorder(body);

        let mut position = vec![None; count];
        for (index, block) in reverse_postorder.iter().enumerate() {
            if let Some(slot) = position.get_mut(block.index()) {
                *slot = Some(index);
            }
        }

        let mut reachable = vec![false; count];
        for block in &reverse_postorder {
            if let Some(slot) = reachable.get_mut(block.index()) {
                *slot = true;
            }
        }

        let immediate_dominator =
            compute_immediate_dominators(body, &reverse_postorder, &position, &reachable);

        trace
            .info(Stage::ControlFlow, "dominance computation complete")
            .field("blocks", count)
            .field("reachable", reverse_postorder.len())
            .emit();

        Self {
            reverse_postorder,
            immediate_dominator,
            reachable,
        }
    }


    pub fn reverse_postorder(&self) -> &[Block] {
        &self.reverse_postorder
    }


    pub fn is_reachable(&self, block: Block) -> bool {
        self.reachable.get(block.index()).copied().unwrap_or(false)
    }


    pub fn immediate_dominator(&self, block: Block) -> Option<Block> {
        self.immediate_dominator
            .get(block.index())
            .copied()
            .flatten()
    }


    pub fn dominates(&self, dominator: Block, block: Block) -> bool {
        if !self.is_reachable(dominator) || !self.is_reachable(block) {
            return false;
        }
        let mut current = Some(block);
        while let Some(candidate) = current {
            if candidate == dominator {
                return true;
            }

            current = match self.immediate_dominator(candidate) {
                Some(parent) if parent != candidate => Some(parent),
                _ => None,
            };
        }
        false
    }


    pub fn natural_loops(&self, body: &FunctionBody, trace: &Trace) -> Vec<NaturalLoop> {
        let mut latches_by_header: HashMap<Block, Vec<Block>> = HashMap::new();
        for &block in &self.reverse_postorder {
            let Some(def) = body.blocks.get(block) else {
                continue;
            };
            for &succ in &def.succs {
                if self.dominates(succ, block) {
                    latches_by_header.entry(succ).or_default().push(block);
                }
            }
        }

        let mut loops: Vec<NaturalLoop> = latches_by_header
            .into_iter()
            .map(|(header, mut latches)| {
                latches.sort_by_key(|block| block.index());
                let blocks = collect_loop_blocks(body, header, &latches);
                NaturalLoop {
                    header,
                    latches,
                    blocks,
                }
            })
            .collect();
        loops.sort_by_key(|natural_loop| natural_loop.header.index());
        trace
            .info(Stage::ControlFlow, "natural loop recognition complete")
            .field("loops", loops.len())
            .emit();
        loops
    }
}


fn collect_loop_blocks(body: &FunctionBody, header: Block, latches: &[Block]) -> Vec<Block> {
    let mut visited: HashSet<Block> = HashSet::new();
    visited.insert(header);

    let mut stack: Vec<Block> = latches.to_vec();
    while let Some(block) = stack.pop() {
        if !visited.insert(block) {
            continue;
        }
        let Some(def) = body.blocks.get(block) else {
            continue;
        };
        for &pred in &def.preds {
            if !visited.contains(&pred) {
                stack.push(pred);
            }
        }
    }

    let mut blocks: Vec<Block> = visited.into_iter().collect();
    blocks.sort_by_key(|block| block.index());
    blocks
}
