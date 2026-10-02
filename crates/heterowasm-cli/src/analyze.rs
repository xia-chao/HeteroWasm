use std::path::Path;

use heterowasm_address::AddressRecovery;
use heterowasm_cfg::ControlFlow;
use heterowasm_dependence::accesses_in_loop;
use heterowasm_frontend::waffle::entity::EntityRef;
use heterowasm_frontend::waffle::{FuncDecl, FunctionBody, Module};
use heterowasm_frontend::{expand_all_bodies, load_module, normalize_bulk_memory};
use heterowasm_legality::LoopLegal;
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::Stage;

use crate::cli::Options;
use crate::error::CliError;
use crate::types::{AnalysisReport, FunctionReport, LoopReport};


pub(crate) fn analyze(input: &Path, options: &Options) -> Result<(), CliError> {
    let trace = &options.trace;
    let _scope = trace.stage(Stage::Runtime, "analyze");
    let raw = std::fs::read(input).map_err(|source| CliError::Read {
        path: input.to_path_buf(),
        source,
    })?;

    let bytes = normalize_bulk_memory(&raw, trace);
    let mut module =
        load_module(&bytes, trace).map_err(|err| CliError::Message(err.to_string()))?;
    let bodies =
        expand_all_bodies(&mut module, trace).map_err(|err| CliError::Message(err.to_string()))?;

    let mut report = AnalysisReport::default();
    for decl in module.funcs.values() {
        let FuncDecl::Body(_, name, body) = decl else {
            continue;
        };
        if options.dump_ir {
            println!("--- WAFFLE IR for {name} ---");
            println!("{}", body.display_verbose("", Some(&module)));
        }
        report
            .functions
            .push(collect_function(name, body, &module, options));
    }

    if options.json {
        println!("{}", crate::json::render_json(&report));
    } else {
        println!("module loaded: {bodies} function bodies available for analysis");
        crate::human::render_human(&report);
    }
    Ok(())
}

fn collect_function(
    name: &str,
    body: &FunctionBody,
    module: &Module<'_>,
    options: &Options,
) -> FunctionReport {
    let trace = &options.trace;

    let flow = ControlFlow::analyze(body, trace);
    let loops = flow.natural_loops(body, trace);

    let mut loop_reports = Vec::with_capacity(loops.len());
    for natural_loop in &loops {
        let evolution = ScalarEvolution::analyze(body, natural_loop, trace);
        let recovery = AddressRecovery::analyze(body, &evolution, trace);
        let accesses = accesses_in_loop(recovery.accesses(), &natural_loop.blocks);
        let legal = LoopLegal::judge(body, natural_loop, &accesses, &evolution, module, trace);

        loop_reports.push(LoopReport {
            header: natural_loop.header.index(),
            outside_coverage: evolution.outside_coverage(),
            blocks: natural_loop.blocks.len(),
            induction_variables: evolution.induction_variables().len(),

            accesses_total: accesses.len(),
            accesses_affine: accesses.iter().filter(|access| access.is_affine()).count(),
            accesses_unknown: accesses.iter().filter(|access| !access.is_affine()).count(),
            obligations: legal.obligations().to_vec(),
            verdict: legal.verdict(),
            disposition: legal.disposition(),
            guards: legal.guards().to_vec(),
        });
    }

    FunctionReport {
        name: name.to_string(),
        blocks: body.blocks.len(),
        values: body.values.len(),
        loops: loop_reports,
    }
}
