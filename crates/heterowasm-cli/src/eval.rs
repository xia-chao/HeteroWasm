use heterowasm_eval::evaluate;
use heterowasm_trace::{Level, Trace};

use crate::error::CliError;


pub(crate) fn run_eval(json: bool) -> Result<(), CliError> {
    let trace = Trace::to_stderr(Level::Info);
    let report = evaluate(&trace).map_err(|err| CliError::Message(err.to_string()))?;

    if json {
        println!("{}", crate::json::render_eval_json(&report));
    } else {
        crate::human::render_eval_human(&report);
    }
    Ok(())
}
