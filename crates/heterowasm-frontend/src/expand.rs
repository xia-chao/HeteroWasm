use heterowasm_trace::{Stage, Trace};

use crate::error::LoadError;


pub fn expand_all_bodies(
    module: &mut waffle::Module<'_>,
    trace: &Trace,
) -> Result<usize, LoadError> {
    let _scope = trace.stage(Stage::Frontend, "expand_all_bodies");

    let funcs: Vec<waffle::Func> = module.funcs.iter().collect();
    let mut bodies = 0usize;
    for func in funcs {
        expand_one_body(module, func, trace)?;
        if module.funcs[func].body().is_some() {
            bodies += 1;
        }
    }
    trace
        .info(Stage::Frontend, "function body expand complete")
        .field("bodies", bodies)
        .emit();
    Ok(bodies)
}


fn expand_one_body(
    module: &mut waffle::Module<'_>,
    func: waffle::Func,
    trace: &Trace,
) -> Result<(), LoadError> {
    module.expand_func(func).map_err(|err| {
        let reason = format!("failed to expand function body: {err:#}");
        trace
            .error(Stage::Frontend, "function body expand failed")
            .field("reason", reason.as_str())
            .emit();
        LoadError::Malformed { reason }
    })?;
    Ok(())
}
