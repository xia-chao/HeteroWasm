use std::collections::HashSet;

use waffle::entity::EntityRef;
use waffle::{Block, FunctionBody};


pub(crate) fn reverse_postorder(body: &FunctionBody) -> Vec<Block> {

    let count = body.blocks.len();
    let mut postorder: Vec<Block> = Vec::with_capacity(count);
    let mut visited: HashSet<Block> = HashSet::with_capacity(count);
    let mut stack: Vec<(Block, usize)> = Vec::with_capacity(count);
    stack.push((body.entry, 0));
    visited.insert(body.entry);

    while let Some((block, next_successor)) = stack.pop() {
        let Some(def) = body.blocks.get(block) else {
            continue;
        };
        if next_successor < def.succs.len() {
            stack.push((block, next_successor + 1));
            let succ = def.succs[next_successor];
            if visited.insert(succ) {
                stack.push((succ, 0));
            }
        } else {
            postorder.push(block);
        }
    }

    postorder.reverse();
    postorder
}


pub(crate) fn compute_immediate_dominators(
    body: &FunctionBody,
    reverse_postorder: &[Block],
    position: &[Option<usize>],
    reachable: &[bool],
) -> Vec<Option<Block>> {
    let mut immediate: Vec<Option<Block>> = vec![None; body.blocks.len()];
    if let Some(slot) = immediate.get_mut(body.entry.index()) {
        *slot = Some(body.entry);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for &block in reverse_postorder.iter().skip(1) {
            let Some(def) = body.blocks.get(block) else {
                continue;
            };
            let mut candidate: Option<Block> = None;
            for &pred in &def.preds {
                if !reachable.get(pred.index()).copied().unwrap_or(false) {
                    continue;
                }

                if immediate.get(pred.index()).copied().flatten().is_none() {
                    continue;
                }
                candidate = Some(match candidate {
                    None => pred,
                    Some(current) => intersect(current, pred, &immediate, position),
                });
            }

            if let Some(new_dominator) = candidate {
                if immediate.get(block.index()).copied().flatten() != Some(new_dominator) {
                    if let Some(slot) = immediate.get_mut(block.index()) {
                        *slot = Some(new_dominator);
                    }
                    changed = true;
                }
            }
        }
    }

    immediate
}


fn intersect(
    first: Block,
    second: Block,
    immediate: &[Option<Block>],
    position: &[Option<usize>],
) -> Block {
    let mut finger_first = first;
    let mut finger_second = second;

    loop {
        let pos_first = position.get(finger_first.index()).copied().flatten();
        let pos_second = position.get(finger_second.index()).copied().flatten();
        let (Some(pos_first), Some(pos_second)) = (pos_first, pos_second) else {
            return finger_first;
        };
        if pos_first == pos_second {
            return finger_first;
        }
        if pos_first > pos_second {
            match immediate.get(finger_first.index()).copied().flatten() {
                Some(parent) if parent != finger_first => finger_first = parent,
                _ => return finger_first,
            }
        } else {
            match immediate.get(finger_second.index()).copied().flatten() {
                Some(parent) if parent != finger_second => finger_second = parent,
                _ => return finger_second,
            }
        }
    }
}
