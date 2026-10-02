use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module};
use heterowasm_trace::Trace;
use waffle::{FuncDecl, FunctionBody, Operator, Value, ValueDef};

use super::ScalarEvolution;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn with_body_and_loop<F>(case_name: &str, inspect: F) -> TestResult
where
    F: FnOnce(&FunctionBody, &ScalarEvolution),
{
    let case = CASES
        .iter()
        .find(|case| case.name == case_name)
        .ok_or_else(|| format!("case {case_name} is not registered"))?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let first = loops
        .first()
        .ok_or("case must contain at least one natural loop")?;
    let evolution = ScalarEvolution::analyze(body, first, &trace);
    inspect(body, &evolution);
    Ok(())
}


fn first_store_address(body: &FunctionBody) -> Option<Value> {
    for value in body.values.iter() {
        if let ValueDef::Operator(Operator::I32Store { .. }, args, _) = &body.values[value] {
            return body.arg_pool[*args].first().copied();
        }
    }
    None
}

#[test]
fn vector_add_should_have_exactly_one_induction_variable() -> TestResult {
    with_body_and_loop("pointwise/vector_add", |_, evolution| {
        assert_eq!(evolution.induction_variables().len(), 1);
    })
}

#[test]
fn vector_add_induction_variable_should_step_by_one() -> TestResult {
    with_body_and_loop("pointwise/vector_add", |_, evolution| {
        let variable = evolution.induction_variables()[0];
        assert_eq!(
            variable.step, 1,
            "loop variable increments by one each iteration"
        );
    })
}

#[test]
fn vector_add_store_address_should_be_base_plus_four_times_i() -> TestResult {
    with_body_and_loop("pointwise/vector_add", |body, evolution| {
        let address = first_store_address(body).expect("case must contain one i32.store");
        let affine = evolution
            .affine_of(body, address)
            .expect("address of C[i] must be affine, or the core positive case of spec §34 fails");

        assert_eq!(
            affine.offset, 0,
            "addresses in this case have no constant offset"
        );
        assert_eq!(affine.induction.len(), 1, "exactly one induction term");
        assert_eq!(affine.induction[0].1, 4, "i32 element stride is 4 bytes");
        assert_eq!(affine.invariants.len(), 1, "base is a loop invariant");
    })
}

#[test]
fn strided_copy_should_report_stride_eight() -> TestResult {
    with_body_and_loop("stride/strided_copy", |body, evolution| {
        let address = first_store_address(body).expect("case must contain one i32.store");
        let affine = evolution
            .affine_of(body, address)
            .expect("address of C[2i] is still affine");
        assert_eq!(
            affine.induction[0].1, 8,
            "skipping one i32 element is 8 bytes"
        );
    })
}


#[test]
fn indirect_index_should_contain_a_non_affine_load_address() -> TestResult {
    with_body_and_loop("indirect_index", |body, evolution| {
        let mut saw_non_affine = false;
        for value in body.values.iter() {
            let ValueDef::Operator(Operator::I32Load { .. }, args, _) = &body.values[value] else {
                continue;
            };
            let Some(&address) = body.arg_pool[*args].first() else {
                continue;
            };
            if evolution.affine_of(body, address).is_none() {
                saw_non_affine = true;
            }
        }
        assert!(
            saw_non_affine,
            "A[i] = B[C[i]] must contain an address depending on a load; it is not affine"
        );
    })
}

#[test]
fn two_loops_should_each_have_their_own_induction_variable() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.name == "control/two_loops")
        .ok_or("case control/two_loops is not registered")?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let body = module
        .funcs
        .values()
        .find_map(|decl| match decl {
            FuncDecl::Body(_, _, body) => Some(body),
            _ => None,
        })
        .ok_or("case must contain one expanded function body")?;

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    assert_eq!(loops.len(), 2);

    for natural_loop in &loops {
        let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
        assert_eq!(
            evolution.induction_variables().len(),
            1,
            "each of the two loops has its own independent loop variable"
        );
    }
    Ok(())
}


#[test]
fn optimized_shape_loop_should_not_be_marked_outside_coverage() -> TestResult {
    with_body_and_loop("pointwise/vector_add", |_, evolution| {
        assert!(
            !evolution.outside_coverage(),
            "SSA-shaped loops (induction already recovered) must not be marked out of coverage"
        );
    })
}

#[test]
fn debug_product_should_contain_at_least_one_outside_coverage_loop() -> TestResult {
    let compiled = heterowasm_corpus::compiled_cases()?;
    let debug_products: Vec<_> = compiled
        .iter()
        .filter(|case| case.opt_level == "0")
        .collect();
    if debug_products.is_empty() {
        eprintln!("note: no -O0 artifact found; run bash corpus/source-compiled/build.sh first");
        return Ok(());
    }

    let trace = Trace::silent();
    let mut outside = 0_usize;
    for product in debug_products {
        let mut module = load_module(&product.bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for decl in module.funcs.values() {
            let FuncDecl::Body(_, _, body) = decl else {
                continue;
            };
            let flow = ControlFlow::analyze(body, &trace);
            for natural_loop in flow.natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                if evolution.outside_coverage() {
                    outside += 1;
                }
            }
        }
    }

    assert!(
        outside > 0,
        "debug artifacts must mark at least one loop out of coverage, else this signal is always false"
    );
    Ok(())
}

#[test]
fn debug_products_should_recover_recurrences_from_memory() -> TestResult {
    let compiled = heterowasm_corpus::compiled_cases()?;
    let mut recovered = 0_usize;
    for product in &compiled {
        if product.opt_level != "0" {
            continue;
        }
        let mut module = load_module(&product.bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        for decl in module.funcs.values() {
            let FuncDecl::Body(_, _, body) = decl else {
                continue;
            };
            let flow = ControlFlow::analyze(body, &Trace::silent());
            for natural_loop in flow.natural_loops(body, &Trace::silent()) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &Trace::silent());
                recovered += evolution.recovered_from_memory();
            }
        }
    }
    assert!(
        recovered > 0,
        "debug artifacts must have at least one induction var recovered from a memory stack slot"
    );
    Ok(())
}


#[test]
fn loop_invariance_should_accept_the_slot_address_and_reject_the_stored_counter() -> TestResult {
    use std::collections::HashSet;

    use waffle::Block;

    use super::LoopInvariance;

    let compiled = heterowasm_corpus::compiled_cases()?;
    let mut checked = 0_usize;

    for product in &compiled {
        if product.opt_level != "0" {
            continue;
        }
        let trace = Trace::silent();
        let mut module = load_module(&product.bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for decl in module.funcs.values() {
            let FuncDecl::Body(_, _, body) = decl else {
                continue;
            };
            let flow = ControlFlow::analyze(body, &trace);
            for natural_loop in flow.natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                if evolution.recovered_from_memory() == 0 {
                    continue;
                }
                let members: HashSet<Block> = natural_loop.blocks.iter().copied().collect();
                let invariance =
                    LoopInvariance::new(body, &natural_loop, evolution.invariant_parameters());

                for value in body.values.iter() {
                    let ValueDef::Operator(Operator::I32Store { .. }, args, _) =
                        &body.values[value]
                    else {
                        continue;
                    };
                    if !members.contains(&body.value_blocks[value]) {
                        continue;
                    }
                    let operands = &body.arg_pool[*args];
                    let (Some(&address), Some(&stored)) = (operands.first(), operands.get(1))
                    else {
                        continue;
                    };

                    if !stores_a_load_of(body, address, stored) {
                        continue;
                    }

                    assert!(
                        invariance.value_is_invariant(address),
                        "store address must be classified loop-invariant, else memory recurrence cannot be recovered"
                    );
                    assert!(
                        !invariance.value_is_invariant(stored),
                        "counter stored in a slot changes every iteration; classifying it invariant is an unsafe false-negative"
                    );
                    checked += 1;
                }
            }
        }
    }

    assert!(
        checked > 0,
        "`-O0` artifacts must contain at least one memory-recurrence loop for this check, else the test is vacuously true"
    );
    Ok(())
}


fn stores_a_load_of(body: &FunctionBody, address: Value, stored: Value) -> bool {
    let Some(ValueDef::Operator(Operator::I32Add, args, _)) =
        body.values.get(super::canonical_value(body, stored))
    else {
        return false;
    };
    let operands: &[Value] = &body.arg_pool[*args];
    operands.iter().any(|operand| {
        let Some(ValueDef::Operator(Operator::I32Load { .. }, load_args, _)) =
            body.values.get(super::canonical_value(body, *operand))
        else {
            return false;
        };
        body.arg_pool[*load_args]
            .first()
            .is_some_and(|loaded_address| {
                super::canonical_value(body, *loaded_address)
                    == super::canonical_value(body, address)
            })
    })
}
