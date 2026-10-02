pub mod liveness {
    use std::collections::HashSet;

    use heterowasm_scev::canonical_value;
    use waffle::{Block, FunctionBody, Terminator, Value, ValueDef};


    pub fn live_outs(body: &FunctionBody, members: &HashSet<Block>) -> Vec<Value> {
        let mut defined_inside: HashSet<Value> = HashSet::new();
        for block in members {
            let Some(definition) = body.blocks.get(*block) else {
                continue;
            };
            for (_, param) in &definition.params {
                defined_inside.insert(*param);
            }
            for inst in &definition.insts {
                defined_inside.insert(*inst);
            }
        }

        let mut found: Vec<Value> = Vec::new();
        for block in body.blocks.iter().collect::<Vec<_>>() {
            if members.contains(&block) {
                continue;
            }
            let Some(definition) = body.blocks.get(block) else {
                continue;
            };
            let mut referenced: Vec<Value> =
                definition.params.iter().map(|(_, value)| *value).collect();
            for inst in &definition.insts {

                match body.values.get(*inst) {
                    Some(ValueDef::Operator(_, args, _)) => {
                        referenced.extend(body.arg_pool[*args].iter().copied());
                    }
                    Some(ValueDef::Alias(inner)) => referenced.push(*inner),
                    Some(ValueDef::PickOutput(inner, _, _)) => referenced.push(*inner),
                    _ => {}
                }
            }

            referenced.extend(crate::ir::terminator_arguments(&definition.terminator));
            for value in referenced {
                let canonical = canonical_value(body, value);
                if defined_inside.contains(&canonical) && !found.contains(&value) {
                    found.push(value);
                }
            }
        }

        for block in members {
            let Some(definition) = body.blocks.get(*block) else {
                continue;
            };
            for target in terminator_targets(&definition.terminator) {
                if members.contains(&target.block) {
                    continue;
                }
                for value in &target.args {
                    let canonical = canonical_value(body, *value);
                    if defined_inside.contains(&canonical) && !found.contains(value) {
                        found.push(*value);
                    }
                }
            }
        }
        found
    }


    pub fn exit_edge_arguments(
        body: &FunctionBody,
        members: &HashSet<Block>,
        exit: Block,
    ) -> Result<Vec<Value>, String> {
        let mut edges: Vec<Vec<Value>> = Vec::new();
        for block in members {
            let Some(definition) = body.blocks.get(*block) else {
                continue;
            };
            for target in terminator_targets(&definition.terminator) {
                if target.block == exit {
                    edges.push(target.args.clone());
                }
            }
        }
        match edges.len() {
            0 => Ok(Vec::new()),
            1 => Ok(edges.remove(0)),
            count => Err(format!(
                "loop has {count} edges into the exit block; auto-rewrite does not support this yet"
            )),
        }
    }


    pub fn is_pass_through(
        body: &FunctionBody,
        header: Block,
        members: &HashSet<Block>,
        value: Value,
    ) -> bool {
        let canonical = canonical_value(body, value);
        let Some(ValueDef::BlockParam(block, index, _)) = body.values.get(canonical) else {
            return false;
        };
        if *block != header {
            return false;
        }
        let index = *index as usize;
        for member in members {
            let Some(definition) = body.blocks.get(*member) else {
                continue;
            };
            for target in terminator_targets(&definition.terminator) {
                if target.block != header {
                    continue;
                }
                let Some(arg) = target.args.get(index).copied() else {
                    return false;
                };
                if canonical_value(body, arg) != canonical {
                    return false;
                }
            }
        }
        true
    }


    fn terminator_targets(terminator: &Terminator) -> Vec<&waffle::BlockTarget> {
        match terminator {
            Terminator::Br { target } => vec![target],
            Terminator::CondBr {
                if_true, if_false, ..
            } => vec![if_true, if_false],
            _ => Vec::new(),
        }
    }
}

pub use liveness::{exit_edge_arguments, is_pass_through, live_outs};
