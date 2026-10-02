use waffle::{Block, FunctionBody, Terminator, Value, ValueDef};

use heterowasm_scev::canonical_value;


pub(crate) fn terminator_arguments(terminator: &Terminator) -> Vec<Value> {
    match terminator {
        Terminator::Br { target } => target.args.clone(),
        Terminator::CondBr {
            if_true, if_false, ..
        } => {
            let mut args = if_true.args.clone();
            args.extend(if_false.args.iter().copied());
            args
        }
        _ => Vec::new(),
    }
}


pub(crate) fn remap_terminator_arguments(
    terminator: &mut Terminator,
    replacements: &[(Value, Value)],
) {
    fn remap(args: &mut [Value], replacements: &[(Value, Value)]) {
        for arg in args.iter_mut() {
            if let Some((_, to)) = replacements.iter().find(|(from, _)| from == arg) {
                *arg = *to;
            }
        }
    }
    match terminator {
        Terminator::Br { target } => remap(&mut target.args, replacements),
        Terminator::CondBr {
            if_true, if_false, ..
        } => {
            remap(&mut if_true.args, replacements);
            remap(&mut if_false.args, replacements);
        }
        _ => {}
    }
}


pub(crate) fn targets_block(terminator: &Terminator, block: Block) -> bool {
    match terminator {
        Terminator::Br { target } => target.block == block,
        Terminator::CondBr {
            if_true, if_false, ..
        } => if_true.block == block || if_false.block == block,
        _ => false,
    }
}


pub(crate) fn header_parameter_indices(
    body: &FunctionBody,
    header: Block,
    fields: &[Value],
) -> Result<Vec<usize>, String> {
    fields
        .iter()
        .map(|field| {
            let canonical = canonical_value(body, *field);
            match body.values.get(canonical) {
                Some(ValueDef::BlockParam(block, index, _)) if *block == header => {
                    Ok(*index as usize)
                }
                _ => Err("field is not a loop-header block parameter".to_string()),
            }
        })
        .collect()
}
