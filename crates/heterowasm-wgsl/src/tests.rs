use super::{
    inject_dispatch_import, plan_gpu_arity, rewrite_module_for_gpu, rewrite_module_fused,
    rewrite_module_two_dispatches,
};
use crate::fusion::{
    collect_region_uses, exclusive_intermediates, fusable_pairs, fuse_two_chain,
    same_iteration_space,
};
use crate::ir::header_parameter_indices;
use crate::single::{offload_loop, plan_first_gpu_loop};
use heterowasm_address::AddressRecovery;
use heterowasm_bounds::LoopExtent;
use heterowasm_cfg::ControlFlow;
use heterowasm_corpus::CASES;
use heterowasm_frontend::{expand_all_bodies, load_module, normalize_bulk_memory};
use heterowasm_scev::ScalarEvolution;
use heterowasm_trace::{Level, Sink, Stage, Trace};
use std::collections::HashSet;
use waffle::FuncDecl;

use super::{lower, Dispatch, Kernel, LowerError};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn lower_first_loop(case_name: &str) -> Result<Kernel, Box<dyn std::error::Error>> {
    let case = CASES
        .iter()
        .find(|case| case.name == case_name)
        .ok_or_else(|| format!("case {case_name} is not registered"))?;
    lower_bytes(&case.to_wasm()?)
}


fn lower_bytes(bytes: &[u8]) -> Result<Kernel, Box<dyn std::error::Error>> {
    let mut module = load_module(bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;

    for decl in module.funcs.values() {
        let FuncDecl::Body(_, _, body) = decl else {
            continue;
        };
        let trace = Trace::silent();
        let flow = ControlFlow::analyze(body, &trace);
        let loops = flow.natural_loops(body, &trace);
        let Some(natural_loop) = loops.first() else {
            continue;
        };
        let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
        let recovery = AddressRecovery::analyze(body, &evolution, &trace);
        let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
        let accesses: Vec<_> = recovery
            .accesses()
            .iter()
            .filter(|access| members.contains(&access.block))
            .cloned()
            .collect();
        let extent = LoopExtent::analyze(body, natural_loop, &evolution);
        return lower(
            body,
            natural_loop,
            &accesses,
            &evolution,
            extent,
            64,
            &trace,
        )
        .map_err(Into::into);
    }

    Err("case must contain at least one natural loop".into())
}


#[test]
fn fewer_than_32_shader_multiplies_stay_off_the_gpu() -> TestResult {
    for name in [
        "intensity/stencil5",
        "intensity/conv3",
        "intensity/horner4",
        "intensity/horner8",
        "intensity/horner16",
    ] {
        let kernel = lower_first_loop(name)?;
        assert!(
            kernel.shader_multiplies_still_slower_than_cpu(),
            "{name} has {} multiplies and must stay on the CPU",
            kernel.source.matches(" * ").count()
        );
        let case = CASES
            .iter()
            .find(|case| case.name == name)
            .ok_or_else(|| format!("case {name} is not registered"))?;
        let bytes = case.to_wasm()?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let error = rewrite_module_for_gpu(&mut module, &Trace::silent())
            .expect_err("a light loop must not be rewritten");
        assert!(
            error.contains("no extractable loop"),
            "{name} rewrite returned {error}"
        );
    }
    for name in ["intensity/horner32", "intensity/poly64"] {
        let kernel = lower_first_loop(name)?;
        assert!(
            !kernel.shader_multiplies_still_slower_than_cpu(),
            "{name} has {} multiplies and must stay eligible",
            kernel.source.matches(" * ").count()
        );
    }
    Ok(())
}


#[test]
fn constant_base_should_emit_expected_kernel() -> TestResult {
    let kernel = lower_first_loop("integer/constant_base")?;
    assert_eq!(kernel.workgroup_size, 64);
    assert_eq!(kernel.dispatch, Dispatch::Fixed(8));
    assert!(
        kernel.dispatch.fixed_trip_is_slower_than_cpu(),
        "a fixed trip of 8 is below the measured crossover"
    );

    let expected = "\
struct Params {
  p0 : u32,
};

@group(0) @binding(0) var<storage, read_write> mem : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= 8u) {
    return;
  }
  mem[i] = params.p0;
}
";
    assert_eq!(kernel.source, expected);
    Ok(())
}


#[test]
fn runtime_trip_should_emit_expected_kernel() -> TestResult {
    let kernel = lower_first_loop("pointwise/vector_add")?;
    assert_eq!(kernel.dispatch, Dispatch::FromField(0));

    let expected = "\
struct Params {
  p0 : u32,
  p1 : u32,
  p2 : u32,
  p3 : u32,
};

@group(0) @binding(0) var<storage, read_write> mem : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= params.p0) {
    return;
  }
  mem[params.p1 / 4u + i] = mem[params.p2 / 4u + i] + mem[params.p3 / 4u + i];
}
";
    assert_eq!(kernel.source, expected);
    Ok(())
}


#[test]
fn nested_product_should_keep_precedence() -> TestResult {
    let kernel = lower_first_loop("intensity/horner4")?;

    let expected = "\
struct Params {
  p0 : u32,
  p1 : u32,
  p2 : u32,
};

@group(0) @binding(0) var<storage, read_write> mem : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= params.p0) {
    return;
  }
  mem[params.p1 / 4u + i] = (((1u * mem[params.p2 / 4u + i] + 3u) * mem[params.p2 / 4u + i] + 2u) * mem[params.p2 / 4u + i] + 1u) * mem[params.p2 / 4u + i] + 0u;
}
";
    assert_eq!(kernel.source, expected);
    Ok(())
}


#[test]
fn non_extractable_trip_must_be_rejected() {
    let reason = lower_first_loop("unsupported/nested_loop")
        .err()
        .map(|error| error.to_string());
    assert_eq!(reason.as_deref(), Some(LowerError::NoExtent.as_str()));
}


#[test]
fn non_affine_address_must_be_rejected() {
    let reason = lower_first_loop("indirect_index")
        .err()
        .map(|error| error.to_string());
    assert_eq!(
        reason.as_deref(),
        Some(LowerError::UnsupportedBase.as_str())
    );
}


#[test]
fn load_in_value_should_emit_expected_kernel() -> TestResult {
    let kernel = lower_first_loop("integer/inplace_scale_constant_trip")?;
    assert_eq!(kernel.dispatch, Dispatch::Fixed(8));
    assert!(
        kernel.dispatch.fixed_trip_is_slower_than_cpu(),
        "a fixed trip of 8 is below the measured crossover"
    );

    let expected = "\
struct Params {
  p0 : u32,
};

@group(0) @binding(0) var<storage, read_write> mem : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= 8u) {
    return;
  }
  mem[params.p0 / 4u + i] = mem[params.p0 / 4u + i] * 2u;
}
";
    assert_eq!(kernel.source, expected);
    Ok(())
}


#[test]
fn out_of_subset_operator_must_be_rejected() {
    let reason = lower_first_loop("integer/inplace_mask_constant_trip")
        .err()
        .map(|error| error.to_string());

    assert_eq!(
        reason.as_deref(),
        Some(LowerError::UnsupportedOperator.as_str())
    );
}


#[test]
fn emission_should_not_be_constant_across_cases() {
    let emitted = lower_first_loop("integer/constant_base").is_ok()
        && lower_first_loop("integer/inplace_scale_constant_trip").is_ok()
        && lower_first_loop("pointwise/vector_add").is_ok();
    let rejected = lower_first_loop("indirect_index").is_err()
        && lower_first_loop("integer/inplace_mask_constant_trip").is_err();
    assert!(
        emitted,
        "loops with const/var iteration bounds inside the subset should be emittable"
    );
    assert!(
        rejected,
        "loops with non-affine addresses or out-of-subset ops must be rejected"
    );
}


#[test]
fn kernel_must_carry_the_invocation_contract() -> TestResult {
    let kernel = lower_first_loop("integer/constant_base")?;
    assert!(kernel.source.contains("@builtin(global_invocation_id)"));
    assert!(kernel.source.contains("let i : u32 = gid.x;"));
    assert!(kernel.source.contains("if (i >= 8u)"));
    Ok(())
}

#[test]
fn emitted_kernels_should_match_wasm_execution() -> TestResult {
    use super::conformance::{compare, Comparison, Subject};

    let seed: Vec<u8> = (0..256).map(|index| (index * 37 + 11) as u8).collect();

    let cases: [(&str, &[&[i32]]); 4] = [
        ("integer/constant_base", &[&[7], &[-559038737]]),
        ("integer/inplace_scale_constant_trip", &[&[0, 0], &[0, 0]]),
        ("integer/image_invert", &[&[0, 64, 8]]),
        ("pointwise/vector_add", &[&[0, 64, 128, 8]]),
    ];

    let mut compared = 0_usize;
    let mut skipped = Vec::new();

    for (name, argument_sets) in cases {
        let case = CASES
            .iter()
            .find(|case| case.name == name)
            .ok_or("case is not registered")?;
        let bytes = case.to_wasm()?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let memory = module
            .memories
            .iter()
            .next()
            .ok_or("case must contain memory")?;

        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            let kernel = lower(
                body,
                natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            )?;
            let subject = Subject {
                kernel: &kernel,
                body,
                module: &module,
                func,
                memory,
                seed: &seed,
            };

            let inputs: Vec<Vec<i32>> = argument_sets.iter().map(|args| args.to_vec()).collect();
            match compare(&subject, &inputs) {
                Ok(Comparison::Compared(outcome)) => {
                    assert_eq!(
                        outcome.mismatches, 0,
                        "{name}: memory mismatch in {}/{} input sets",
                        outcome.mismatches, outcome.inputs
                    );
                    assert!(outcome.inputs > 0, "{name}: compared zero input sets");
                    println!(
                        "  {name}: {} input sets, {} bytes each, all match",
                        outcome.inputs,
                        outcome.bytes_compared / outcome.inputs
                    );
                    compared += 1;
                }
                Ok(Comparison::Skipped(reason)) => skipped.push((name.to_string(), reason)),
                Err(error) => panic!("{name}: {error}"),
            }
            break;
        }
    }

    println!("checked {compared} cases, skipped {:?}", skipped);
    assert!(
        compared > 0,
        "at least one case must have been truly checked"
    );
    Ok(())
}

#[test]
fn corrupted_kernel_must_be_detected() -> TestResult {
    use super::conformance::{compare, Comparison, Subject};

    let case = CASES
        .iter()
        .find(|case| case.name == "integer/constant_base")
        .ok_or("case is not registered")?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let memory = module
        .memories
        .iter()
        .next()
        .ok_or("case must contain memory")?;
    let func = module
        .funcs
        .iter()
        .find(|func| matches!(&module.funcs[*func], waffle::FuncDecl::Body(..)))
        .ok_or("case must contain a function body")?;
    let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
        return Err("failed to get function body".into());
    };

    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let loops = flow.natural_loops(body, &trace);
    let natural_loop = loops.first().ok_or("case must contain a natural loop")?;
    let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
    let recovery = AddressRecovery::analyze(body, &evolution, &trace);
    let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
    let accesses: Vec<_> = recovery
        .accesses()
        .iter()
        .filter(|access| members.contains(&access.block))
        .cloned()
        .collect();
    let extent = LoopExtent::analyze(body, natural_loop, &evolution);
    let kernel = lower(
        body,
        natural_loop,
        &accesses,
        &evolution,
        extent,
        64,
        &trace,
    )?;

    let mut corrupted = kernel.clone();
    corrupted.source = corrupted.source.replace("params.p0", "123u");
    assert_ne!(
        corrupted.source, kernel.source,
        "corruption action had no effect"
    );

    let seed: Vec<u8> = (0..256).map(|index| (index * 37 + 11) as u8).collect();
    let inputs = vec![vec![7_i32]];

    let subject = Subject {
        kernel: &corrupted,
        body,
        module: &module,
        func,
        memory,
        seed: &seed,
    };
    let outcome = compare(&subject, &inputs).map_err(|error| error.to_string())?;
    let Comparison::Compared(outcome) = outcome else {
        return Err("negative control must not be skipped".into());
    };
    assert!(
        outcome.mismatches > 0,
        "the broken kernel must be caught by the check, else this harness is a no-op"
    );
    Ok(())
}

#[test]
fn emitted_kernels_should_be_valid_wgsl() -> TestResult {
    let mut checked = 0_usize;
    for name in [
        "integer/constant_base",
        "integer/inplace_scale_constant_trip",
        "integer/image_invert",
        "pointwise/vector_add",
    ] {
        let kernel = lower_first_loop(name)?;
        let module = naga::front::wgsl::parse_str(&kernel.source)
            .map_err(|error| format!("{name}: naga parse failed: {error:?}"))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|error| format!("{name}: naga validation failed: {error:?}"))?;
        checked += 1;
    }
    assert!(
        checked > 0,
        "at least one case must have been validated by naga"
    );
    Ok(())
}


#[test]
fn naga_must_reject_broken_wgsl() {
    let broken = "this is not wgsl at all";
    assert!(
        naga::front::wgsl::parse_str(broken).is_err(),
        "naga must reject non-WGSL text, otherwise the check above is meaningless"
    );
}

#[test]
fn gpu_execution_should_match_wasm_execution() -> TestResult {
    use super::conformance::gpu;
    use waffle::{ConstVal, InterpContext};

    let seed: Vec<u8> = (0..65_536).map(|index| (index * 37 + 11) as u8).collect();
    let cases: [(&str, &[i32]); 4] = [
        ("integer/constant_base", &[7]),
        ("integer/inplace_scale_constant_trip", &[0, 0]),
        ("integer/image_invert", &[0, 64, 8]),
        ("pointwise/vector_add", &[0, 64, 128, 8]),
    ];

    let mut compared = 0_usize;
    for (name, arguments) in cases {
        let case = CASES
            .iter()
            .find(|case| case.name == name)
            .ok_or("case is not registered")?;
        let bytes = case.to_wasm()?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let memory = module
            .memories
            .iter()
            .next()
            .ok_or("case must contain memory")?;

        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            let kernel = lower(
                body,
                natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            )?;


            let mut context = InterpContext::new(&module)?;
            context.memories[memory].data.copy_from_slice(&seed);
            let values: Vec<ConstVal> =
                arguments.iter().map(|a| ConstVal::I32(*a as u32)).collect();
            context.call(&module, func, &values).ok()?;
            let expected = context.memories[memory].data.clone();


            let mapping = super::conformance::field_mapping(body, &kernel.fields)
                .ok_or("field does not trace back to a parameter")?;
            let params: Vec<u32> = mapping
                .iter()
                .map(|index| arguments.get(*index as usize).copied().unwrap_or(0) as u32)
                .collect();
            let execution = gpu::execute(&kernel, &seed, &params)
                .map_err(|error| format!("{name}: {error}"))?;

            assert_eq!(
                execution.memory.len(),
                expected.len(),
                "{name}: read-back memory length is wrong"
            );
            let mismatches = expected
                .iter()
                .zip(execution.memory.iter())
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(
                mismatches, 0,
                "{name}: GPU vs Wasm differ in {mismatches} bytes"
            );
            println!(
                "  {name}: GPU({}) matches Wasm for all {} bytes",
                execution.adapter,
                expected.len()
            );
            compared += 1;
            break;
        }
    }

    assert!(compared > 0, "at least one case must have truly run on GPU");
    Ok(())
}

#[test]
fn dispatch_boundaries_should_write_exactly_the_expected_words() -> TestResult {
    use super::conformance::gpu;
    use super::Dispatch;

    let seed: Vec<u8> = (0..1024).map(|index| (index * 37 + 11) as u8).collect();

    for limit in [0_u32, 1, 63, 64, 65, 128] {
        let source = boundary_kernel(limit);
        let kernel = Kernel {
            source,
            workgroup_size: 64,
            dispatch: Dispatch::Fixed(limit),
            fields: Vec::new(),

            min_constant_bytes: 0,

            max_constant_bytes: 0,
            max_stride_bytes: 4,
            index_field: None,
            launch_count: None,
            results: Vec::new(),
        };

        let execution = gpu::execute(&kernel, &seed, &[]).map_err(|error| error.to_string())?;
        let written = (0..seed.len() / 4)
            .filter(|index| {
                let offset = index * 4;
                execution.memory[offset..offset + 4] == [7, 0, 0, 0]
            })
            .count();
        assert_eq!(
            written, limit as usize,
            "N={limit}: must write exactly {limit} words"
        );


        for index in (limit as usize)..(seed.len() / 4) {
            let offset = index * 4;
            assert_eq!(
                execution.memory[offset..offset + 4],
                seed[offset..offset + 4],
                "N={limit}: word {index} was modified out of bounds"
            );
        }
    }
    Ok(())
}


fn boundary_kernel(limit: u32) -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> mem : array<u32>;\n\
             struct Params {{ p0 : u32, }};\n\
             @group(0) @binding(1) var<uniform> params : Params;\n\
             @compute @workgroup_size(64)\n\
             fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{\n\
               let i : u32 = gid.x;\n\
               if (i >= {limit}u) {{\n    return;\n  }}\n\
               mem[i] = 7u;\n\
             }}\n"
    )
}


const KERNEL_HEAD: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) ";
const KERNEL_TAIL: &str = ") (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";

fn kernel(expression: &str) -> String {
    format!("{KERNEL_HEAD}{expression}{KERNEL_TAIL}")
}

const SQUARES: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.mul (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4)))) (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4)))))) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";
const AXPY: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.add (i32.mul (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4)))) (i32.const 3)) (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4)))))) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";
const POLY3: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.add (i32.mul (i32.add (i32.mul (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4)))) (i32.const 3)) (i32.const 5)) (i32.const 7)) (i32.const 11))) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";
static DOT4_SOURCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let mut expression = String::new();
    for j in 0..4_usize {
        let load = format!("(i32.load (i32.add (local.get $a) (i32.mul (i32.add (i32.mul (local.get $i) (i32.const 4)) (i32.const {j})) (i32.const 4))))");
        let operand = format!("(i32.load (i32.add (local.get $b) (i32.mul (i32.add (i32.mul (local.get $i) (i32.const 4)) (i32.const {j})) (i32.const 4))))");
        let term = format!("(i32.mul {load} {operand})");
        expression = if j == 0 {
            term
        } else {
            format!("(i32.add {expression} {term})")
        };
    }
    kernel(&expression)
});

const STENCIL3: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.add (i32.add (i32.load (i32.add (local.get $a) (i32.mul (i32.sub (local.get $i) (i32.const 1)) (i32.const 4)))) (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))) (i32.load (i32.add (local.get $a) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))))) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";
const HORNER8: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.add (i32.mul (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4)))) (i32.const 3)) (i32.const 5)) (i32.const 7)) (i32.const 11)) (i32.const 13)) (i32.const 17)) (i32.const 19)) (i32.const 23)) (i32.const 29)) (i32.const 31)) (i32.const 37)) (i32.const 41)) (i32.const 43)) (i32.const 47)) (i32.const 53)) (i32.const 59)) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";
const GATHER: &str = "(module (memory (export \"mem\") 256) (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) (local $i i32) (block $exit (loop $loop (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) (i32.load (i32.add (local.get $a) (i32.mul (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4)))) (i32.const 4))))) (local.set $i (i32.add (local.get $i) (i32.const 1))) (br $loop)))) (export \"main\" (func $kern)))";


#[test]
fn crossover_scan_small_n() -> TestResult {
    use super::conformance::gpu;
    use std::time::Instant;

    const WAT: &str = r#"(module
  (memory (export "mem") 256)
  (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32)
    (local $i i32)
    (block $exit
      (loop $loop
        (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
        (i32.store
          (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
          (i32.add
            (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
            (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop))))
  (export "main" (func $kern)))"#;

    let bytes = wat::parse_str(WAT)?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;

    let mut kernel = None;
    for func in module.funcs.iter() {
        let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
            continue;
        };
        let trace = Trace::silent();
        let flow = ControlFlow::analyze(body, &trace);
        let loops = flow.natural_loops(body, &trace);
        let Some(natural_loop) = loops.first() else {
            continue;
        };
        let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
        let recovery = AddressRecovery::analyze(body, &evolution, &trace);
        let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
        let accesses: Vec<_> = recovery
            .accesses()
            .iter()
            .filter(|access| members.contains(&access.block))
            .cloned()
            .collect();
        let extent = LoopExtent::analyze(body, natural_loop, &evolution);
        kernel = Some((
            func,
            lower(
                body,
                natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            )?,
        ));
        break;
    }
    let (func, kernel) = kernel.ok_or("no emittable loop found")?;
    println!("\n{}", kernel.source);


    let engine = wasmtime::Engine::default();
    let compiled = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &compiled, &[])?;
    let entry = instance.get_typed_func::<(i32, i32, i32, i32), ()>(&mut store, "main")?;

    let mapping = super::conformance::field_mapping(body_of(&module, func)?, &kernel.fields)
        .ok_or("field does not trace back to a parameter")?;


    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    println!("backend: {}", context.adapter());

    println!(
        "{:>9}  {:>12}  {:>12}  {:>10}  {:>10}",
        "N", "CPU(us)", "GPU(us)", "GPU/CPU", "xfer(MB)"
    );


    for n in [8_i32, 256, 10_000, 100_000, 1_000_000] {
        let a = 0_i32;
        let b = n * 4;
        let c = n * 8;
        let used = (n as usize) * 12;
        let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();
        let arguments = [a, b, c, n];
        let params: Vec<u32> = mapping
            .iter()
            .map(|index| arguments[*index as usize] as u32)
            .collect();

        let memory_data = instance
            .get_memory(&mut store, "mem")
            .ok_or("exported memory not found")?;
        memory_data.write(&mut store, 0, &seed)?;


        let workspace = context
            .workspace(used, params.len() * 4)
            .map_err(|error| error.to_string())?;
        let mut cpu = Vec::new();
        for round in 0..12 {
            let started = Instant::now();
            entry.call(&mut store, (a, b, c, n))?;

            if round == 0 {
                let expected = memory_data.data(&store)[..used].to_vec();
                let actual = context
                    .run_in(&compiled, &kernel, &workspace, &seed, &params)
                    .map_err(|error| error.to_string())?;
                let mismatches = expected
                    .iter()
                    .zip(actual.iter())
                    .filter(|(left, right)| left != right)
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "N={n}: GPU vs CPU differ in {mismatches} bytes"
                );
            }
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            if round >= 2 {
                cpu.push(elapsed);
            }
        }
        cpu.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let mut gpu = Vec::new();
        for round in 0..12 {
            let started = Instant::now();
            context
                .run_in(&compiled, &kernel, &workspace, &seed, &params)
                .map_err(|error| error.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            if round >= 2 {
                gpu.push(elapsed);
            }
        }
        gpu.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let cpu_median = cpu[cpu.len() / 2];
        let gpu_median = gpu[gpu.len() / 2];
        println!(
            "{n:>9}  {cpu_median:>12.1}  {gpu_median:>12.1}  {:>9.2}x  {:>10.3}",
            gpu_median / cpu_median,
            used as f64 / (1024.0 * 1024.0)
        );
    }

    println!();
    Ok(())
}

fn body_of<'m>(
    module: &'m waffle::Module<'_>,
    func: waffle::Func,
) -> Result<&'m waffle::FunctionBody, Box<dyn std::error::Error>> {
    match &module.funcs[func] {
        waffle::FuncDecl::Body(_, _, body) => Ok(body),
        _ => Err("not a function body".into()),
    }
}

fn intensity_wat(k: usize) -> String {
    let load_a = "(i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))";
    let load_b = "(i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))";
    let mut expression = String::from(load_a);
    for term in 0..k {
        let (operand, constant) = if term % 2 == 0 {
            (load_b, term * 2 + 3)
        } else {
            (load_a, term * 2 + 5)
        };
        expression = format!("(i32.add {expression} (i32.mul {operand} (i32.const {constant})))");
    }
    format!(
            "(module (memory (export \"mem\") 256) \
             (func $kern (param $a i32) (param $b i32) (param $c i32) (param $n i32) \
               (local $i i32) \
               (block $exit (loop $loop \
                 (br_if $exit (i32.ge_s (local.get $i) (local.get $n))) \
                 (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))) {expression}) \
                 (local.set $i (i32.add (local.get $i) (i32.const 1))) \
                 (br $loop)))) (export \"main\" (func $kern)))"
        )
}


#[test]
fn crossover_scan_intensity() -> TestResult {
    use super::conformance::gpu;
    use std::time::Instant;

    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let engine = wasmtime::Engine::default();

    println!(
        "\nbackend: {}\n{:>4}  {:>9}  {:>12}  {:>12}  {:>10}",
        context.adapter(),
        "k",
        "N",
        "CPU(us)",
        "GPU(us)",
        "GPU/CPU"
    );


    for k in [1_usize, 4, 8, 12, 14, 16, 18, 20, 24, 32] {
        let bytes = wat::parse_str(intensity_wat(k))?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;

        let mut lowered = None;
        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            lowered = Some((
                func,
                lower(
                    body,
                    natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                )?,
            ));
            break;
        }
        let (func, kernel) = lowered.ok_or("no emittable loop found")?;

        let compiled = context
            .compile(&kernel)
            .map_err(|error| error.to_string())?;
        let compiled_wasm = wasmtime::Module::new(&engine, &bytes)?;
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &compiled_wasm, &[])?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32), ()>(&mut store, "main")?;
        let body = match &module.funcs[func] {
            waffle::FuncDecl::Body(_, _, body) => body,
            _ => return Err("not a function body".into()),
        };
        let mapping = super::conformance::field_mapping(body, &kernel.fields)
            .ok_or("field does not trace back to a parameter")?;

        for n in [100_000_i32, 1_000_000] {
            let (a, b, c) = (0_i32, n * 4, n * 8);
            let used = (n as usize) * 12;
            let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();
            let arguments = [a, b, c, n];
            let params: Vec<u32> = mapping
                .iter()
                .map(|index| arguments[*index as usize] as u32)
                .collect();

            let memory_data = instance
                .get_memory(&mut store, "mem")
                .ok_or("exported memory not found")?;
            memory_data.write(&mut store, 0, &seed)?;

            let mut cpu = Vec::new();
            for round in 0..12 {
                let started = Instant::now();
                entry.call(&mut store, (a, b, c, n))?;
                let elapsed = started.elapsed().as_secs_f64() * 1e6;
                if round >= 2 {
                    cpu.push(elapsed);
                }
            }
            cpu.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

            let workspace = context
                .workspace(used, params.len() * 4)
                .map_err(|error| error.to_string())?;
            let mut gpu = Vec::new();
            for round in 0..12 {
                let started = Instant::now();
                let actual = context
                    .run_in(&compiled, &kernel, &workspace, &seed, &params)
                    .map_err(|error| error.to_string())?;
                let elapsed = started.elapsed().as_secs_f64() * 1e6;
                if round == 0 {
                    let expected = memory_data.data(&store)[..used].to_vec();
                    let mismatches = expected
                        .iter()
                        .zip(actual.iter())
                        .filter(|(left, right)| left != right)
                        .count();
                    assert_eq!(
                        mismatches, 0,
                        "k={k} N={n}: GPU vs CPU differ in {mismatches} bytes"
                    );
                }
                if round >= 2 {
                    gpu.push(elapsed);
                }
            }
            gpu.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

            let cpu_median = cpu[cpu.len() / 2];
            let gpu_median = gpu[gpu.len() / 2];
            println!(
                "{k:>4}  {n:>9}  {cpu_median:>12.1}  {gpu_median:>12.1}  {:>9.2}x",
                gpu_median / cpu_median
            );
        }
    }

    println!();
    Ok(())
}

fn median_and_mad(values: &mut [f64]) -> (f64, f64) {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let median = values[values.len() / 2];
    let mut deviations: Vec<f64> = values.iter().map(|value| (value - median).abs()).collect();
    deviations.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    (median, deviations[deviations.len() / 2])
}


#[test]
fn crossover_formal_measurement() -> TestResult {
    use super::conformance::gpu;
    use std::time::Instant;

    const ROUNDS: usize = 31;
    const N: i32 = 1_000_000;

    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let engine = wasmtime::Engine::default();

    println!(
        "\nbackend: {}    N = {N}    {ROUNDS} rounds/point (drop first), CPU/GPU alternate\n\
             {:>4}  {:>11}  {:>11}  {:>11}  {:>11}  {:>9}  {:>20}",
        context.adapter(),
        "k",
        "CPU median(us)",
        "CPU MAD",
        "GPU median(us)",
        "GPU MAD",
        "ratio",
        "ratio band(has1?)"
    );

    for k in [1_usize, 4, 8, 10, 12, 13, 14, 16, 18, 20, 24, 32] {
        let bytes = wat::parse_str(intensity_wat(k))?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;

        let mut lowered = None;
        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            lowered = Some((
                func,
                lower(
                    body,
                    natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                )?,
            ));
            break;
        }
        let (func, kernel) = lowered.ok_or("no emittable loop found")?;
        let compiled = context
            .compile(&kernel)
            .map_err(|error| error.to_string())?;

        let compiled_wasm = wasmtime::Module::new(&engine, &bytes)?;
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &compiled_wasm, &[])?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32), ()>(&mut store, "main")?;
        let body = match &module.funcs[func] {
            waffle::FuncDecl::Body(_, _, body) => body,
            _ => return Err("not a function body".into()),
        };
        let mapping = super::conformance::field_mapping(body, &kernel.fields)
            .ok_or("field does not trace back to a parameter")?;

        let (a, b, c) = (0_i32, N * 4, N * 8);
        let used = (N as usize) * 12;
        let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();
        let arguments = [a, b, c, N];
        let params: Vec<u32> = mapping
            .iter()
            .map(|index| arguments[*index as usize] as u32)
            .collect();
        let memory_data = instance
            .get_memory(&mut store, "mem")
            .ok_or("exported memory not found")?;
        let workspace = context
            .workspace(used, params.len() * 4)
            .map_err(|error| error.to_string())?;

        let mut cpu = Vec::with_capacity(ROUNDS);
        let mut gpu = Vec::with_capacity(ROUNDS);

        for round in 0..ROUNDS {
            memory_data.write(&mut store, 0, &seed)?;
            let started = Instant::now();
            entry.call(&mut store, (a, b, c, N))?;
            let cpu_elapsed = started.elapsed().as_secs_f64() * 1e6;

            let started = Instant::now();
            let actual = context
                .run_in(&compiled, &kernel, &workspace, &seed, &params)
                .map_err(|error| error.to_string())?;
            let gpu_elapsed = started.elapsed().as_secs_f64() * 1e6;

            if round == 0 {
                let expected = memory_data.data(&store)[..used].to_vec();
                let mismatches = expected
                    .iter()
                    .zip(actual.iter())
                    .filter(|(left, right)| left != right)
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "k={k}: GPU vs CPU differ in {mismatches} bytes"
                );
            }
            if round > 0 {
                cpu.push(cpu_elapsed);
                gpu.push(gpu_elapsed);
            }
        }

        let (cpu_median, cpu_mad) = median_and_mad(&mut cpu);
        let (gpu_median, gpu_mad) = median_and_mad(&mut gpu);
        let ratio = gpu_median / cpu_median;

        let low = (gpu_median - gpu_mad) / (cpu_median + cpu_mad);
        let high = (gpu_median + gpu_mad) / (cpu_median - cpu_mad);
        let verdict = if low > 1.0 {
            "GPU slower"
        } else if high < 1.0 {
            "GPU faster"
        } else {
            "interval contains 1 (not significant)"
        };
        println!(
                "{k:>4}  {cpu_median:>11.1}  {cpu_mad:>11.1}  {gpu_median:>11.1}  {gpu_mad:>11.1}  {ratio:>9.2}x  {low:>8.2}..{high:<8.2} {verdict}"
            );
    }

    println!();
    Ok(())
}

#[test]
fn real_kernel_intensity_against_crossover() -> TestResult {
    let cases: [(&str, &str); 7] = [
        ("squares  C[i]=A[i]*A[i]", SQUARES),
        ("axpy     C[i]=3*A[i]+B[i]", AXPY),
        ("poly3    C[i]=((A[i]*3+5)*7+11)", POLY3),
        ("dot4     C[i]=sum A[4i+j]*B[4i+j]", DOT4_SOURCE.as_str()),
        ("stencil3 C[i]=A[i-1]+A[i]+A[i+1]", STENCIL3),
        ("horner8  C[i]=8-step Horner", HORNER8),
        ("gather   C[i]=A[B[i]]", GATHER),
    ];

    println!(
        "\n{}  {}  {}  {}  {}   {}",
        format_args!("{:>34}", "op"),
        format_args!("{:>10}", "emittable?"),
        format_args!("{:>7}", "mul"),
        format_args!("{:>7}", "add"),
        format_args!("{:>7}", "load"),
        "in crossover(k∈[12,16])?"
    );

    for (name, source) in cases {
        let bytes = match wat::parse_str(source) {
            Ok(bytes) => bytes,
            Err(error) => {
                println!(
                    "{name:>34}  {status:>10}  {dash:>7}  {dash:>7}  {dash:>7}   {error}",
                    status = "WAT error",
                    dash = "-"
                );
                continue;
            }
        };
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;

        let mut lowered = None;
        let mut failure = None;
        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            match lower(
                body,
                natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            ) {
                Ok(kernel) => lowered = Some(kernel),
                Err(error) => failure = Some(error.to_string()),
            }
            break;
        }

        let Some(kernel) = lowered else {
            println!(
                "{name:>34}  {status:>10}  {dash:>7}  {dash:>7}  {dash:>7}   {reason}",
                status = "rejected",
                dash = "-",
                reason = failure.unwrap_or_else(|| "no loop".to_string())
            );
            continue;
        };


        let line = kernel
            .source
            .lines()
            .find(|line| line.trim_start().starts_with("mem[") && line.contains(" = "))
            .ok_or("emit result has no assignment statement")?;
        let rhs = line
            .split_once(" = ")
            .ok_or("assignment statement format changed")?
            .1
            .trim_end_matches(';');

        let loads = rhs.matches("mem[").count();
        let mut value_only = String::new();
        let mut depth = 0_usize;
        for character in rhs.chars() {
            match character {
                '[' => depth += 1,
                ']' => depth = depth.saturating_sub(1),
                _ if depth == 0 => value_only.push(character),
                _ => {}
            }
        }
        let multiplies = value_only.matches('*').count();
        let adds = value_only.matches('+').count();
        let verdict = if (12..=16).contains(&multiplies) {
            "**within band**"
        } else if multiplies > 16 {
            "above band (GPU faster)"
        } else {
            "below band (GPU slower)"
        };
        println!(
            "{name:>34}  {status:>10}  {multiplies:>7}  {adds:>7}  {loads:>7}   {verdict}",
            status = "ok"
        );
    }

    println!();
    Ok(())
}

#[test]
fn artifact_round_trip_through_disk_runs_and_matches_wasm() -> TestResult {
    use super::conformance::gpu;
    use waffle::{ConstVal, InterpContext};

    let root = std::env::temp_dir().join(format!(
        "heterowasm-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    std::fs::create_dir_all(&root)?;

    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let engine = wasmtime::Engine::default();
    let mut checked = 0_usize;

    for (name, arguments) in [
        ("integer/constant_base", vec![7_i32]),
        ("pointwise/vector_add", vec![0, 64, 128, 8]),
    ] {
        let case = CASES
            .iter()
            .find(|case| case.name == name)
            .ok_or("case is not registered")?;
        let bytes = case.to_wasm()?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let memory = module
            .memories
            .iter()
            .next()
            .ok_or("case must contain memory")?;

        let mut found = None;
        for func in module.funcs.iter() {
            let waffle::FuncDecl::Body(_, _, body) = &module.funcs[func] else {
                continue;
            };
            let trace = Trace::silent();
            let flow = ControlFlow::analyze(body, &trace);
            let loops = flow.natural_loops(body, &trace);
            let Some(natural_loop) = loops.first() else {
                continue;
            };
            let evolution = ScalarEvolution::analyze(body, natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let members: HashSet<waffle::Block> = natural_loop.blocks.iter().copied().collect();
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| members.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, natural_loop, &evolution);
            let kernel = lower(
                body,
                natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            )?;
            found = Some((func, body, kernel));
            break;
        }
        let (func, body, kernel) = found.ok_or("no emittable loop found")?;


        let artifact = super::artifact(&kernel, body)?;
        let wgsl_path = root.join(format!("{}.wgsl", name.replace('/', "_")));
        let manifest_path = root.join(format!("{}.json", name.replace('/', "_")));
        std::fs::write(&wgsl_path, artifact.wgsl)?;
        std::fs::write(&manifest_path, &artifact.manifest)?;


        let wgsl = std::fs::read_to_string(&wgsl_path)?;
        let manifest = std::fs::read_to_string(&manifest_path)?;
        assert!(
            manifest.contains("\"schema\": \"heterowasm.kernel.v1\""),
            "{name}: manifest missing schema marker"
        );
        assert!(
            manifest.contains(&format!("\"workgroup_size\": {}", kernel.workgroup_size)),
            "{name}: manifest missing workgroup size"
        );

        let dispatch = match kernel.dispatch {
            Dispatch::Fixed(count) => {
                assert!(
                    manifest.contains(&format!("\"count\": {count}")),
                    "{name}: manifest missing dispatch"
                );
                Dispatch::Fixed(count)
            }
            Dispatch::FromField(field) => {
                assert!(
                    manifest.contains(&format!("\"field\": {field}")),
                    "{name}: manifest missing dispatch field"
                );
                Dispatch::FromField(field)
            }
            Dispatch::FromFieldMask(field, mask) => {
                assert!(
                    manifest.contains(&format!("\"field\": {field}")),
                    "{name}: manifest missing dispatch field"
                );
                assert!(
                    manifest.contains(&format!("\"mask\": {mask}")),
                    "{name}: manifest missing dispatch mask"
                );
                Dispatch::FromFieldMask(field, mask)
            }
            Dispatch::FromFieldMaskAdd(field, mask, addend) => {
                assert!(
                    manifest.contains("\"kind\": \"from_field_mask_add\""),
                    "{name}: manifest missing masked add bound"
                );
                assert!(
                    manifest.contains(&format!("\"field\": {field}")),
                    "{name}: manifest missing dispatch field"
                );
                assert!(
                    manifest.contains(&format!("\"mask\": {mask}")),
                    "{name}: manifest missing dispatch mask"
                );
                assert!(
                    manifest.contains(&format!("\"addend\": {addend}")),
                    "{name}: manifest missing dispatch addend"
                );
                Dispatch::FromFieldMaskAdd(field, mask, addend)
            }
            Dispatch::FromSum(left, right) => {
                assert!(
                    manifest.contains(&format!("\"left\": {left}")),
                    "{name}: manifest missing sum fields"
                );
                assert!(
                    manifest.contains(&format!("\"right\": {right}")),
                    "{name}: manifest missing sum fields"
                );
                Dispatch::FromSum(left, right)
            }
            Dispatch::FromSumShift(base, shifted, amount) => {
                assert!(
                    manifest.contains(&format!("\"base\": {base}")),
                    "{name}: manifest missing shifted base"
                );
                assert!(
                    manifest.contains(&format!("\"shifted\": {shifted}")),
                    "{name}: manifest missing shifted field"
                );
                assert!(
                    manifest.contains(&format!("\"amount\": {amount}")),
                    "{name}: manifest missing shift amount"
                );
                Dispatch::FromSumShift(base, shifted, amount)
            }
            Dispatch::FromSubShift(base, minuend, subtrahend, amount) => {
                assert!(
                    manifest.contains("\"kind\": \"from_sub_shift\""),
                    "{name}: manifest missing difference shift bound"
                );
                assert!(
                    manifest.contains(&format!("\"base\": {base}")),
                    "{name}: manifest missing difference base"
                );
                assert!(
                    manifest.contains(&format!("\"minuend\": {minuend}")),
                    "{name}: manifest missing minuend"
                );
                assert!(
                    manifest.contains(&format!("\"subtrahend\": {subtrahend}")),
                    "{name}: manifest missing subtrahend"
                );
                assert!(
                    manifest.contains(&format!("\"amount\": {amount}")),
                    "{name}: manifest missing shift amount"
                );
                Dispatch::FromSubShift(base, minuend, subtrahend, amount)
            }
            Dispatch::FromLoadedShiftMask(base, shifted, amount, mask) => {
                assert!(
                    manifest.contains("\"kind\": \"from_loaded_shift_mask\""),
                    "{name}: manifest missing loaded shift mask"
                );
                assert!(
                    manifest.contains(&format!("\"base\": {base}")),
                    "{name}: manifest missing loaded base"
                );
                assert!(
                    manifest.contains(&format!("\"shifted\": {shifted}")),
                    "{name}: manifest missing loaded shifted field"
                );
                assert!(
                    manifest.contains(&format!("\"amount\": {amount}")),
                    "{name}: manifest missing loaded shift amount"
                );
                assert!(
                    manifest.contains(&format!("\"mask\": {mask}")),
                    "{name}: manifest missing loaded mask"
                );
                Dispatch::FromLoadedShiftMask(base, shifted, amount, mask)
            }
            Dispatch::FromLoaded(slot, offset) => {
                assert!(
                    manifest.contains("\"kind\": \"from_loaded\""),
                    "{name}: manifest missing loaded bound"
                );
                assert!(
                    manifest.contains(&format!("\"field\": {slot}")),
                    "{name}: manifest missing loaded field"
                );
                assert!(
                    manifest.contains(&format!("\"offset\": {offset}")),
                    "{name}: manifest missing loaded offset"
                );
                Dispatch::FromLoaded(slot, offset)
            }
        };

        let mapping = super::conformance::field_mapping(body, &kernel.fields)
            .ok_or("field does not trace back to a parameter")?;
        for (slot, parameter) in mapping.iter().enumerate() {
            assert!(
                manifest.contains(&format!("\"slot\": {slot}, \"wasm_param\": {parameter}")),
                "{name}: manifest field {slot} mapping is wrong"
            );
        }

        let restored = Kernel {
            source: wgsl,
            workgroup_size: kernel.workgroup_size,
            dispatch,
            fields: Vec::new(),

            min_constant_bytes: 0,

            max_constant_bytes: 0,
            max_stride_bytes: 4,
            index_field: None,
            launch_count: None,
            results: Vec::new(),
        };
        let compiled = context
            .compile(&restored)
            .map_err(|error| error.to_string())?;


        let seed: Vec<u8> = (0..65_536).map(|index| (index * 37 + 11) as u8).collect();
        let mut interpreter = InterpContext::new(&module)?;
        interpreter.memories[memory].data.copy_from_slice(&seed);
        let values: Vec<ConstVal> = arguments.iter().map(|a| ConstVal::I32(*a as u32)).collect();
        interpreter.call(&module, func, &values).ok()?;
        let expected = interpreter.memories[memory].data.clone();

        let params: Vec<u32> = mapping
            .iter()
            .map(|index| arguments[*index as usize] as u32)
            .collect();
        let workspace = context
            .workspace(seed.len(), params.len() * 4)
            .map_err(|error| error.to_string())?;
        let actual = context
            .run_in(&compiled, &restored, &workspace, &seed, &params)
            .map_err(|error| error.to_string())?;

        let mismatches = expected
            .iter()
            .zip(actual.iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(
            mismatches, 0,
            "{name}: disk read-back vs Wasm differ in {mismatches} bytes"
        );


        let parsed = naga::front::wgsl::parse_str(&restored.source)
            .map_err(|error| format!("{name}: naga rejected read-back artifact: {error:?}"))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&parsed)
        .map_err(|error| format!("{name}: naga validation failed: {error:?}"))?;


        let compiled_wasm = wasmtime::Module::new(&engine, &bytes)?;
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &compiled_wasm, &[])?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32), ()>(&mut store, "main");
        let _ = entry;

        println!("  {name}: write → read back → naga ok → GPU matches Wasm byte-for-byte");
        checked += 1;
    }

    std::fs::remove_dir_all(&root).ok();
    assert!(checked >= 2, "at least two cases must complete end-to-end");
    Ok(())
}

#[test]
fn whole_program_split_across_cpu_and_gpu() -> TestResult {
    use super::conformance::gpu;


    const FULL: &str = r#"(module (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitA (loop $loopA
      (br_if $exitA (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopA)))
    (local.set $i (i32.const 0))
    (block $exitB (loop $loopB
      (br_if $exitB (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $d) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $d) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopB))))
  (export "main" (func $prog)))"#;


    const CPU_ONLY: &str = r#"(module (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitB (loop $loopB
      (br_if $exitB (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $d) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $d) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopB))))
  (export "main" (func $prog)))"#;

    let n = 512_i32;
    let a = 0_i32;
    let b = 4096;
    let c = 8192;
    let d = 12288;
    let arguments = (a, b, c, d, n);
    let used = 16384_usize;
    let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();


    let full_bytes = wat::parse_str(FULL)?;
    let mut full_module = load_module(&full_bytes, &Trace::silent())?;
    expand_all_bodies(&mut full_module, &Trace::silent())?;

    let mut gpu_loops: Vec<Kernel> = Vec::new();
    let mut cpu_loops = 0_usize;
    for func in full_module.funcs.iter() {
        let waffle::FuncDecl::Body(_, _, body) = &full_module.funcs[func] else {
            continue;
        };
        let trace = Trace::silent();
        for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            let legality = heterowasm_legality::LoopLegal::judge(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                &full_module,
                &trace,
            );
            if matches!(
                legality.disposition(),
                heterowasm_legality::Disposition::Gpu
                    | heterowasm_legality::Disposition::GpuAfterGuard
            ) {
                if let Ok(kernel) = lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                ) {
                    gpu_loops.push(kernel);
                    continue;
                }
            }
            cpu_loops += 1;
        }
    }
    println!(
        "\nwhole-program classify: {} loops on GPU, {cpu_loops} stay on CPU",
        gpu_loops.len()
    );
    assert_eq!(
        gpu_loops.len(),
        1,
        "stage A should and must only be judged GPU-eligible"
    );
    assert_eq!(
        cpu_loops, 1,
        "stage B has cross-iteration dependence; should stay on CPU"
    );


    let engine = wasmtime::Engine::default();
    let full_compiled = wasmtime::Module::new(&engine, &full_bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &full_compiled, &[])?;
    let entry = instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    entry.call(&mut store, arguments)?;
    let reference = memory.data(&store).to_vec();


    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let kernel = &gpu_loops[0];
    let compiled = context.compile(kernel).map_err(|error| error.to_string())?;


    let full_body = {
        let mut found = None;
        for func in full_module.funcs.iter() {
            if let waffle::FuncDecl::Body(_, _, body) = &full_module.funcs[func] {
                found = Some(body);
            }
        }
        found.ok_or("missing function body")?
    };
    let mapping = super::conformance::field_mapping(full_body, &kernel.fields)
        .ok_or("field does not trace back to a parameter")?;
    let all = [a, b, c, d, n];
    let params: Vec<u32> = mapping
        .iter()
        .map(|index| all[*index as usize] as u32)
        .collect();

    let workspace = context
        .workspace(used, params.len() * 4)
        .map_err(|error| error.to_string())?;
    let after_gpu = context
        .run_in(&compiled, kernel, &workspace, &seed, &params)
        .map_err(|error| error.to_string())?;


    let cpu_bytes = wat::parse_str(CPU_ONLY)?;
    let cpu_compiled = wasmtime::Module::new(&engine, &cpu_bytes)?;
    let mut cpu_store = wasmtime::Store::new(&engine, ());
    let cpu_instance = wasmtime::Instance::new(&mut cpu_store, &cpu_compiled, &[])?;
    let cpu_entry =
        cpu_instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut cpu_store, "main")?;
    let cpu_memory = cpu_instance
        .get_memory(&mut cpu_store, "mem")
        .ok_or("missing memory")?;
    cpu_memory.write(&mut cpu_store, 0, &after_gpu)?;
    cpu_entry.call(&mut cpu_store, arguments)?;
    let split = cpu_memory.data(&cpu_store).to_vec();


    let mismatches = reference
        .iter()
        .zip(split.iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "whole program: reference (pure CPU) {} bytes vs split (GPU A + CPU B) {} bytes → {} mismatches",
        reference.len(),
        split.len(),
        mismatches
    );
    assert_eq!(
        mismatches, 0,
        "split execution mismatches pure CPU execution"
    );


    let mut skip_store = wasmtime::Store::new(&engine, ());
    let skip_instance = wasmtime::Instance::new(&mut skip_store, &cpu_compiled, &[])?;
    let skip_entry =
        skip_instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut skip_store, "main")?;
    let skip_memory = skip_instance
        .get_memory(&mut skip_store, "mem")
        .ok_or("missing memory")?;
    skip_memory.write(&mut skip_store, 0, &seed)?;
    skip_entry.call(&mut skip_store, arguments)?;
    let without_gpu = skip_memory.data(&skip_store).to_vec();
    let differing = reference
        .iter()
        .zip(without_gpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert!(
        differing > 0,
        "skipping the GPU step still yields the same result — this check has no discriminating power"
    );
    println!("negative control: after skipping GPU step, {differing} bytes differ → check has discriminating power\n");
    Ok(())
}


const OFFLOAD_PROGRAM: &str = r#"(module
  (import "heterowasm" "__hw_dispatch" (func $dispatch (param i32 i32 i32 i32 i32)))
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitA (loop $loopA
      (br_if $exitA (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopA)))
    (local.set $i (i32.const 0))
    (block $exitB (loop $loopB
      (br_if $exitB (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $d) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $d) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopB))))
  (export "main" (func $prog)))"#;


#[test]
fn offload_loop_passes_encoding_with_live_out_params() -> TestResult {
    let offload_bytes = wat::parse_str(OFFLOAD_PROGRAM)?;
    let mut module = load_module(&offload_bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;

    let mut plan = None;
    let trace = Trace::silent();
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            let legality = heterowasm_legality::LoopLegal::judge(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                &module,
                &trace,
            );
            if !matches!(
                legality.disposition(),
                heterowasm_legality::Disposition::Gpu
                    | heterowasm_legality::Disposition::GpuAfterGuard
            ) {
                continue;
            }
            let kernel = lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            )?;

            let exit = natural_loop
                .blocks
                .iter()
                .filter_map(|block| body.blocks.get(*block))
                .flat_map(|definition| match &definition.terminator {
                    waffle::Terminator::Br { target } => vec![target.block],
                    waffle::Terminator::CondBr {
                        if_true, if_false, ..
                    } => vec![if_true.block, if_false.block],
                    _ => vec![],
                })
                .find(|block| !natural_loop.blocks.contains(block))
                .ok_or("natural loop has no exit block")?;
            plan = Some((
                func,
                natural_loop.header,
                exit,
                natural_loop.blocks.clone(),
                kernel.fields.clone(),
            ));
            break;
        }
        if plan.is_some() {
            break;
        }
    }
    let (func, header, exit, members, fields) = plan.ok_or("no extractable loop found")?;


    offload_loop(&mut module, func, header, exit, &members, &fields, 0, None)?;
    println!(
        "\nrewrite skeleton applied: header={header:?} exit={exit:?} members={members:?} \
             (back-edges removed, terminators rewritten, edge table rebuilt)"
    );


    let encoded = module.to_wasm_bytes()?;
    assert!(!encoded.is_empty(), "encode result is empty");
    println!(
        "encode ok: {} bytes (live-outs passed as exit-block parameters)",
        encoded.len()
    );
    let engine = wasmtime::Engine::default();
    wasmtime::Module::new(&engine, &encoded)?;
    Ok(())
}


const REFERENCE_PROGRAM: &str = r#"(module
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitA (loop $loopA
      (br_if $exitA (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopA)))
    (local.set $i (i32.const 0))
    (block $exitB (loop $loopB
      (br_if $exitB (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $d) (i32.mul (i32.add (local.get $i) (i32.const 1)) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $d) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopB))))
  (export "main" (func $prog)))"#;


#[test]
fn rewritten_wasm_runs_on_gpu_and_matches_cpu() -> TestResult {
    use super::conformance::gpu;

    let n = 512_i32;
    let (a, b, c, d) = (0_i32, 4096, 8192, 12288);
    let arguments = (a, b, c, d, n);
    let used = 16384_usize;
    let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();
    let engine = wasmtime::Engine::default();


    let reference_bytes = wat::parse_str(REFERENCE_PROGRAM)?;
    let reference_module = wasmtime::Module::new(&engine, &reference_bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &reference_module, &[])?;
    let entry = instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    entry.call(&mut store, arguments)?;
    let reference = memory.data(&store).to_vec();


    let offload_bytes = wat::parse_str(OFFLOAD_PROGRAM)?;
    let mut module = load_module(&offload_bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let plan = plan_first_gpu_loop(&module, &Trace::silent())?;
    offload_loop(
        &mut module,
        plan.func,
        plan.header,
        plan.exit,
        &plan.members,
        &plan.kernel.fields,
        0,
        None,
    )?;
    let kernel = plan.kernel;
    let rewritten = module.to_wasm_bytes()?;
    println!(
        "\nauto-rewrite: before {} bytes → after {} bytes",
        offload_bytes.len(),
        rewritten.len()
    );


    struct Host {
        context: gpu::GpuContext,
        compiled: gpu::Compiled,
        kernel: Kernel,
    }
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;

    let mut linker = wasmtime::Linker::new(&engine);
    linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |mut caller: wasmtime::Caller<'_, Host>,
         p0: i32,
         p1: i32,
         p2: i32,
         p3: i32,
         _kernel: i32|
         -> Result<(), wasmtime::Error> {
            let memory = caller
                .get_export("mem")
                .and_then(|item| item.into_memory())
                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
            let data = memory.data(&caller).to_vec();
            let host = caller.data();
            let params = [p0 as u32, p1 as u32, p2 as u32, p3 as u32];
            let workspace = host
                .context
                .workspace(data.len(), params.len() * 4)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            let produced = host
                .context
                .run_in(&host.compiled, &host.kernel, &workspace, &data, &params)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            memory.write(&mut caller, 0, &produced)?;
            Ok(())
        },
    )?;

    let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;
    let mut gpu_store = wasmtime::Store::new(
        &engine,
        Host {
            context,
            compiled,
            kernel,
        },
    );
    let gpu_instance = linker.instantiate(&mut gpu_store, &rewritten_module)?;
    let gpu_entry =
        gpu_instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut gpu_store, "main")?;
    let gpu_memory = gpu_instance
        .get_memory(&mut gpu_store, "mem")
        .ok_or("missing memory")?;
    gpu_memory.write(&mut gpu_store, 0, &seed)?;
    gpu_entry.call(&mut gpu_store, arguments)?;
    let produced = gpu_memory.data(&gpu_store).to_vec();

    let mismatches = reference
        .iter()
        .zip(produced.iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "auto-rewritten module: vs pure CPU reference {} bytes → {} mismatches",
        reference.len(),
        mismatches
    );
    assert_eq!(
        mismatches, 0,
        "auto-rewrite + GPU offload result mismatches pure CPU execution"
    );


    let mut idle_linker = wasmtime::Linker::new(&engine);
    idle_linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |_p0: i32, _p1: i32, _p2: i32, _p3: i32, _kernel: i32| {},
    )?;
    let idle_module = wasmtime::Module::new(&engine, &rewritten)?;
    let mut idle_store = wasmtime::Store::new(&engine, ());
    let idle_instance = idle_linker.instantiate(&mut idle_store, &idle_module)?;
    let idle_entry =
        idle_instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut idle_store, "main")?;
    let idle_memory = idle_instance
        .get_memory(&mut idle_store, "mem")
        .ok_or("missing memory")?;
    idle_memory.write(&mut idle_store, 0, &seed)?;
    idle_entry.call(&mut idle_store, arguments)?;
    let idle = idle_memory.data(&idle_store).to_vec();
    let differing = reference
        .iter()
        .zip(idle.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert!(
        differing > 0,
        "no-op offload call still yields the same result — this check has no discriminating power"
    );
    println!("negative control: after no-op __hw_dispatch, {differing} bytes differ → check has discriminating power\n");
    Ok(())
}


#[test]
fn bounds_fallback_matches_wasm_trapping() -> TestResult {
    let engine = wasmtime::Engine::default();
    let reference_bytes = wat::parse_str(REFERENCE_PROGRAM)?;
    let module = wasmtime::Module::new(&engine, &reference_bytes)?;
    let seed: Vec<u8> = (0..16384).map(|index| (index * 37 + 11) as u8).collect();
    let n = 512_i32;

    let buffer_bytes = 16 * 65_536_u64;


    let offload_bytes = wat::parse_str(OFFLOAD_PROGRAM)?;
    let mut parsed = load_module(&offload_bytes, &Trace::silent())?;
    expand_all_bodies(&mut parsed, &Trace::silent())?;
    let plan = plan_first_gpu_loop(&parsed, &Trace::silent())?;
    let kernel = plan.kernel;

    for (label, c, expect_gpu) in [
        ("in-bounds", 8192_i32, true),
        ("out of bounds", 2_000_000, false),
    ] {
        let arguments = (0_i32, 4096, c, 12288, n);

        let params: Vec<u32> = vec![n as u32, c as u32, 0, 4096];

        let decision = crate::decide_dispatch(&kernel, &params, buffer_bytes);
        let on_gpu = matches!(decision, crate::DispatchDecision::Gpu);


        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
        let memory = instance
            .get_memory(&mut store, "mem")
            .ok_or("missing memory")?;
        memory.write(&mut store, 0, &seed)?;
        let trapped = entry.call(&mut store, arguments).is_err();

        match &decision {
            crate::DispatchDecision::Gpu => {
                println!("verdict {label}: GPU (Wasm trapped = {trapped})");
            }
            crate::DispatchDecision::CpuFallback { reason } => {
                println!("verdict {label}: CPU fallback — {reason} (Wasm trapped = {trapped})");
            }
        }
        assert_eq!(on_gpu, expect_gpu, "{label}: verdict direction is wrong");

        assert_eq!(
            !on_gpu, trapped,
            "{label}: verdict disagrees with Wasm trap behavior — Wasm must not trap when choosing GPU, and must trap on fallback"
        );
    }
    Ok(())
}


const SINGLE_LOOP: &str = r#"(module
  (import "heterowasm" "__hw_dispatch" (func $dispatch (param i32 i32 i32 i32 i32)))
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitA (loop $loopA
      (br_if $exitA (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopA))))
  (export "main" (func $prog)))"#;


const SINGLE_LOOP_CPU: &str = r#"(module
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $c i32) (param $d i32) (param $n i32)
    (local $i i32)
    (block $exitA (loop $loopA
      (br_if $exitA (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $c) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $loopA))))
  (export "main" (func $prog)))"#;


struct CpuFallback {
    module: wasmtime::Module,
    engine: wasmtime::Engine,
}

impl CpuFallback {

    fn run(
        &self,
        memory: &[u8],
        arguments: (i32, i32, i32, i32, i32),
    ) -> Result<Vec<u8>, wasmtime::Error> {
        let mut store = wasmtime::Store::new(&self.engine, ());
        let instance = wasmtime::Instance::new(&mut store, &self.module, &[])?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
        let target = instance
            .get_memory(&mut store, "mem")
            .ok_or_else(|| wasmtime::Error::msg("missing memory"))?;
        target.write(&mut store, 0, memory)?;
        entry.call(&mut store, arguments)?;
        Ok(target.data(&store).to_vec())
    }
}


#[test]
fn bounds_fallback_is_wired_to_cpu_execution() -> TestResult {
    use super::conformance::gpu;

    let engine = wasmtime::Engine::default();


    let offload_bytes = wat::parse_str(SINGLE_LOOP)?;
    let mut parsed = load_module(&offload_bytes, &Trace::silent())?;
    expand_all_bodies(&mut parsed, &Trace::silent())?;
    let plan = plan_first_gpu_loop(&parsed, &Trace::silent())?;
    let kernel = plan.kernel.clone();
    let (func, header, exit, members) = (plan.func, plan.header, plan.exit, plan.members.clone());

    let indices = header_parameter_indices(
        parsed.funcs[func].body().ok_or("missing function body")?,
        header,
        &kernel.fields,
    )?;
    offload_loop(
        &mut parsed,
        func,
        header,
        exit,
        &members,
        &kernel.fields,
        0,
        None,
    )?;
    let rewritten = parsed.to_wasm_bytes()?;
    println!(
        "\nauto-rewrite: before {} bytes → after {} bytes",
        offload_bytes.len(),
        rewritten.len()
    );
    let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;

    struct Host {
        context: gpu::GpuContext,
        compiled: gpu::Compiled,
        kernel: Kernel,
        fallback: CpuFallback,
        buffer_bytes: u64,

        positions: Vec<usize>,
        used_gpu: std::cell::Cell<bool>,
        used_cpu: std::cell::Cell<bool>,
    }

    let mut linker = wasmtime::Linker::new(&engine);
    linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |mut caller: wasmtime::Caller<'_, Host>,
         p0: i32,
         p1: i32,
         p2: i32,
         p3: i32,
         _kernel: i32|
         -> Result<(), wasmtime::Error> {
            let memory = caller
                .get_export("mem")
                .and_then(|item| item.into_memory())
                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
            let data = memory.data(&caller).to_vec();
            let host = caller.data();
            let params = [p0 as u32, p1 as u32, p2 as u32, p3 as u32];

            match crate::decide_dispatch(&host.kernel, &params, host.buffer_bytes) {
                crate::DispatchDecision::Gpu => {
                    host.used_gpu.set(true);
                    let workspace = host
                        .context
                        .workspace(data.len(), params.len() * 4)
                        .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                    let produced = host
                        .context
                        .run_in(&host.compiled, &host.kernel, &workspace, &data, &params)
                        .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                    memory.write(&mut caller, 0, &produced)?;
                    Ok(())
                }
                crate::DispatchDecision::CpuFallback { reason } => {
                    eprintln!("fall back to CPU: {reason}");
                    host.used_cpu.set(true);

                    let mut full = [0_i32; 5];
                    for (slot, position) in host.positions.iter().enumerate() {
                        full[*position] = params[slot] as i32;
                    }
                    let produced = host
                        .fallback
                        .run(&data, (full[0], full[1], full[2], full[3], full[4]))?;
                    memory.write(&mut caller, 0, &produced)?;
                    Ok(())
                }
            }
        },
    )?;

    for (label, c, expect_gpu, expect_trap) in [
        ("in-bounds", 8192_i32, true, false),
        ("out of bounds", 2_000_000, false, true),
    ] {
        let arguments = (0_i32, 4096, c, 12288, 512_i32);
        let seed: Vec<u8> = (0..16384).map(|index| (index * 37 + 11) as u8).collect();


        let reference = if expect_trap {
            None
        } else {
            let mut store = wasmtime::Store::new(&engine, ());
            let module = wasmtime::Module::new(&engine, &wat::parse_str(SINGLE_LOOP_CPU)?)?;
            let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
            let entry =
                instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
            let memory = instance
                .get_memory(&mut store, "mem")
                .ok_or("missing memory")?;
            memory.write(&mut store, 0, &seed)?;
            entry.call(&mut store, arguments)?;
            Some(memory.data(&store).to_vec())
        };

        let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
        let compiled = context
            .compile(&kernel)
            .map_err(|error| error.to_string())?;
        let mut store = wasmtime::Store::new(
            &engine,
            Host {
                context,
                compiled,
                kernel: kernel.clone(),
                fallback: CpuFallback {
                    module: wasmtime::Module::new(&engine, &wat::parse_str(SINGLE_LOOP_CPU)?)?,
                    engine: engine.clone(),
                },
                buffer_bytes: 16 * 65_536,
                positions: indices.clone(),
                used_gpu: std::cell::Cell::new(false),
                used_cpu: std::cell::Cell::new(false),
            },
        );
        let instance = linker.instantiate(&mut store, &rewritten_module)?;
        let entry = instance.get_typed_func::<(i32, i32, i32, i32, i32), ()>(&mut store, "main")?;
        let memory = instance
            .get_memory(&mut store, "mem")
            .ok_or("missing memory")?;
        memory.write(&mut store, 0, &seed)?;
        let outcome = entry.call(&mut store, arguments);

        let used_gpu = store.data().used_gpu.get();
        let used_cpu = store.data().used_cpu.get();
        println!(
            "{label}: result={} used_gpu={used_gpu} used_cpu_fallback={used_cpu}",
            if outcome.is_ok() {
                "normal return"
            } else {
                "trap"
            }
        );

        assert_eq!(
            used_gpu, expect_gpu,
            "{label}: GPU branch usage does not match expectation"
        );
        assert_eq!(
            used_cpu, !expect_gpu,
            "{label}: CPU fallback branch usage does not match expectation"
        );
        assert_eq!(
            outcome.is_err(),
            expect_trap,
            "{label}: trap behavior does not match expectation (with correct fallback wiring, OOB should trap like Wasm)"
        );

        if let Some(reference) = reference {
            let produced = memory.data(&store).to_vec();
            let mismatches = reference
                .iter()
                .zip(produced.iter())
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(mismatches, 0, "{label}: mismatches pure CPU reference");
            println!(
                "{label}: vs pure CPU reference {} bytes → 0 mismatches",
                reference.len()
            );
        }
    }
    println!();
    Ok(())
}


#[test]
fn corpus_cases_survive_automatic_rewrite() -> TestResult {

    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {

            corpus_cases_survive_automatic_rewrite_inner().map_err(|error| error.to_string())
        })
        .map_err(|error| format!("failed to spawn thread: {error}"))?;
    handle
        .join()
        .map_err(|_| "deep-recursion test thread panicked".to_string())?
        .map_err(|error| -> Box<dyn std::error::Error> { error.into() })
}

fn corpus_cases_survive_automatic_rewrite_inner() -> TestResult {
    use super::conformance::gpu;

    let engine = wasmtime::Engine::default();
    let mut rewritten_cases = 0_usize;
    let mut skipped: Vec<(&str, String)> = Vec::new();

    for case in CASES {

        let original_bytes = case.to_wasm()?;

        let normalized = normalize_bulk_memory(&original_bytes, &Trace::silent());
        let mut probe = load_module(&normalized, &Trace::silent())?;
        expand_all_bodies(&mut probe, &Trace::silent())?;
        let arity = match plan_first_gpu_loop(&probe, &Trace::silent()) {
            Ok(plan) => (plan.kernel.fields.len(), plan.kernel.results.len()),
            Err(reason) => {
                skipped.push((case.name, reason));
                continue;
            }
        };
        let (arity, result_slots) = arity;


        let source_bytes = wat::parse_str(case.wat)?;
        let injected_bytes =
            inject_dispatch_import(&source_bytes, arity, result_slots, &Trace::silent())?;
        let mut module = load_module(&injected_bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;


        let plan = plan_first_gpu_loop(&module, &Trace::silent())?;
        let kernel = plan.kernel.clone();

        let positions = header_parameter_indices(
            module.funcs[plan.func]
                .body()
                .ok_or("missing function body")?,
            plan.header,
            &kernel.fields,
        )
        .unwrap_or_default();
        let (func, header, exit, members) =
            (plan.func, plan.header, plan.exit, plan.members.clone());

        let wasm_params = {
            let body = module.funcs[func].body().ok_or("missing function body")?;
            super::conformance::field_mapping(body, &kernel.fields)
                .ok_or("field does not trace back to a function parameter")?
        };
        match offload_loop(
            &mut module,
            func,
            header,
            exit,
            &members,
            &kernel.fields,
            0,
            None,
        ) {
            Ok(()) => {}
            Err(reason) => {
                skipped.push((case.name, format!("rewrite rejected: {reason}")));
                continue;
            }
        }
        let rewritten = match module.to_wasm_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                skipped.push((case.name, format!("encode failed: {error}")));
                continue;
            }
        };


        let param_count = module.funcs[func]
            .body()
            .map(|body| body.n_params)
            .unwrap_or(0);
        let mut arguments = vec![0_i32; param_count];
        for (slot, parameter) in wasm_params.iter().enumerate() {
            if (*parameter as usize) < arguments.len() {

                arguments[*parameter as usize] = if slot == 0 { 8 } else { (slot as i32) * 64 };
            }
        }
        let seed: Vec<u8> = (0..65_536).map(|index| (index * 37 + 11) as u8).collect();


        let reference_module = wasmtime::Module::new(&engine, &original_bytes)?;
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &reference_module, &[])?;

        let Some(entry) = instance.get_func(&mut store, "main") else {
            skipped.push((case.name, "case does not export main".to_string()));
            continue;
        };
        let values: Vec<wasmtime::Val> = arguments
            .iter()
            .take(entry.ty(&store).params().len())
            .map(|argument| wasmtime::Val::I32(*argument))
            .collect();
        let Some(memory) = instance.get_memory(&mut store, "mem") else {
            skipped.push((case.name, "case has no memory".to_string()));
            continue;
        };
        memory.write(&mut store, 0, &seed)?;
        if entry.call(&mut store, &values, &mut []).is_err() {
            skipped.push((case.name, "reference execution trapped".to_string()));
            continue;
        }
        let reference = memory.data(&store).to_vec();


        struct Host {
            context: gpu::GpuContext,
            compiled: gpu::Compiled,
            kernel: Kernel,
        }
        let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
        let compiled = context
            .compile(&kernel)
            .map_err(|error| error.to_string())?;
        let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;


        let mut linker = wasmtime::Linker::new(&engine);
        let dispatch_type = wasmtime::FuncType::new(
            &engine,
            (0..arity + 1).map(|_| wasmtime::ValType::I32),
            (0..result_slots).map(|_| wasmtime::ValType::I32),
        );
        linker.func_new(
            "heterowasm",
            "__hw_dispatch",
            dispatch_type,
            |mut caller: wasmtime::Caller<'_, Host>,
             args: &[wasmtime::Val],
             results: &mut [wasmtime::Val]|
             -> Result<(), wasmtime::Error> {
                let memory = caller
                    .get_export("mem")
                    .and_then(|item| item.into_memory())
                    .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
                let data = memory.data(&caller).to_vec();
                let host = caller.data();
                let mut params: Vec<u32> = args
                    .iter()
                    .map(|value| value.i32().unwrap_or(0) as u32)
                    .collect();

                params.pop();
                let workspace = host
                    .context
                    .workspace_with_results(data.len(), params.len() * 4, host.kernel.results.len())
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                let (produced, exit_values) = host
                    .context
                    .run_in_with_exit(&host.compiled, &host.kernel, &workspace, &data, &params)
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                memory.write(&mut caller, 0, &produced)?;
                for (slot, value) in results.iter_mut().zip(exit_values.iter()) {
                    *slot = wasmtime::Val::I32(*value as i32);
                }
                Ok(())
            },
        )?;

        let mut store = wasmtime::Store::new(
            &engine,
            Host {
                context,
                compiled,
                kernel,
            },
        );
        let instance = linker.instantiate(&mut store, &rewritten_module)?;
        let entry = instance
            .get_func(&mut store, "main")
            .ok_or("rewritten module lost the main export")?;
        let rewritten_values: Vec<wasmtime::Val> = arguments
            .iter()
            .take(entry.ty(&store).params().len())
            .map(|argument| wasmtime::Val::I32(*argument))
            .collect();
        let memory = instance
            .get_memory(&mut store, "mem")
            .ok_or("missing memory")?;
        memory.write(&mut store, 0, &seed)?;
        entry.call(&mut store, &rewritten_values, &mut [])?;
        let produced = memory.data(&store).to_vec();

        let mismatches = reference
            .iter()
            .zip(produced.iter())
            .filter(|(left, right)| left != right)
            .count();
        if mismatches > 0 {
            let offsets: Vec<usize> = reference
                .iter()
                .zip(produced.iter())
                .enumerate()
                .filter(|(_, (left, right))| left != right)
                .map(|(index, _)| index)
                .take(6)
                .collect();
            println!(
                "  DIAG {}: positions={positions:?} arguments={arguments:?} diff offsets={offsets:?}",
                case.name
            );
        }
        assert_eq!(
            mismatches, 0,
            "{}: after auto-rewrite, {mismatches} byte mismatches vs pure CPU reference",
            case.name
        );
        println!(
            "  {}: rewrite {} → {} bytes, matches pure CPU byte-for-byte",
            case.name,
            injected_bytes.len(),
            rewritten.len()
        );
        rewritten_cases += 1;
    }

    println!(
        "\ncases through auto-rewrite: {rewritten_cases}; skipped {}",
        skipped.len()
    );
    for (name, reason) in skipped.iter().take(12) {
        println!("  skip {name}: {reason}");
    }
    assert!(
        rewritten_cases > 0,
        "at least one case must complete auto-rewrite"
    );
    Ok(())
}


#[test]
fn fusion_feasibility_analysis() -> TestResult {
    use super::conformance::gpu;
    use std::time::Instant;

    const N: i32 = 1_000_000;
    const ROUNDS: usize = 15;

    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let engine = wasmtime::Engine::default();
    let used = (N as usize) * 12;
    let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();


    let params = [N as u32, 0, N as u32 * 4, N as u32 * 8];

    let mut fused_pair = None;
    let mut single_pair = None;
    let mut break_even: Option<usize> = None;

    println!(
        "\nbackend: {}   N = {N}   median of {ROUNDS} rounds/point\n\
             {:>6}  {:>13}  {:>13}  {:>13}  {:>11}  {:>12}",
        context.adapter(),
        "chain L",
        "CPU(us)",
        "fused GPU(us)",
        "unfused GPU(us)",
        "fused/CPU",
        "unfused/CPU"
    );

    for length in [1_usize, 2, 4, 8, 16, 32] {

        if fused_pair.is_none() {
            let bytes = wat::parse_str(intensity_wat(length))?;
            let mut module = load_module(&bytes, &Trace::silent())?;
            expand_all_bodies(&mut module, &Trace::silent())?;
            let plan = plan_first_gpu_loop(&module, &Trace::silent())?;
            let kernel = plan.kernel;
            let compiled = context
                .compile(&kernel)
                .map_err(|error| error.to_string())?;
            fused_pair = Some((kernel, compiled));
        }
        let (_, _) = fused_pair.as_ref().ok_or("fused kernel not ready")?;


        let fused_bytes = wat::parse_str(intensity_wat(length))?;
        let mut fused_module = load_module(&fused_bytes, &Trace::silent())?;
        expand_all_bodies(&mut fused_module, &Trace::silent())?;
        let fused_kernel = plan_first_gpu_loop(&fused_module, &Trace::silent())?.kernel;
        let fused_compiled = context
            .compile(&fused_kernel)
            .map_err(|error| error.to_string())?;
        let fused_workspace = context
            .workspace(used, params.len() * 4)
            .map_err(|error| error.to_string())?;


        if single_pair.is_none() {
            let bytes = wat::parse_str(intensity_wat(1))?;
            let mut module = load_module(&bytes, &Trace::silent())?;
            expand_all_bodies(&mut module, &Trace::silent())?;
            let kernel = plan_first_gpu_loop(&module, &Trace::silent())?.kernel;
            let compiled = context
                .compile(&kernel)
                .map_err(|error| error.to_string())?;
            single_pair = Some((kernel, compiled));
        }
        let (single_kernel, single_compiled) = single_pair.as_ref().ok_or("single-op not ready")?;
        let single_workspace = context
            .workspace(used, params.len() * 4)
            .map_err(|error| error.to_string())?;


        let cpu_module = wasmtime::Module::new(&engine, &fused_bytes)?;
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &cpu_module, &[])?;
        let entry = instance
            .get_func(&mut store, "main")
            .ok_or("missing main")?;
        let values: Vec<wasmtime::Val> = vec![
            wasmtime::Val::I32(0),
            wasmtime::Val::I32(N * 4),
            wasmtime::Val::I32(N * 8),
            wasmtime::Val::I32(N),
        ];

        let mut cpu = Vec::new();
        let mut fused = Vec::new();
        let mut unfused = Vec::new();
        for round in 0..ROUNDS {
            let started = Instant::now();
            entry.call(&mut store, &values, &mut [])?;
            if round > 0 {
                cpu.push(started.elapsed().as_secs_f64() * 1e6);
            }

            let started = Instant::now();
            context
                .run_in(
                    &fused_compiled,
                    &fused_kernel,
                    &fused_workspace,
                    &seed,
                    &params,
                )
                .map_err(|error| error.to_string())?;
            if round > 0 {
                fused.push(started.elapsed().as_secs_f64() * 1e6);
            }

            let started = Instant::now();
            for _ in 0..length {
                context
                    .run_in(
                        single_compiled,
                        single_kernel,
                        &single_workspace,
                        &seed,
                        &params,
                    )
                    .map_err(|error| error.to_string())?;
            }
            if round > 0 {
                unfused.push(started.elapsed().as_secs_f64() * 1e6);
            }
        }
        let median = |mut values: Vec<f64>| {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values[values.len() / 2]
        };
        let cpu_median = median(cpu);
        let fused_median = median(fused);
        let unfused_median = median(unfused);

        println!(
            "{length:>6}  {cpu_median:>13.1}  {fused_median:>13.1}  {unfused_median:>13.1}  \
                 {:>10.2}x  {:>11.2}x",
            fused_median / cpu_median,
            unfused_median / cpu_median
        );
        if break_even.is_none() && fused_median < cpu_median {
            break_even = Some(length);
        }
    }

    println!();
    match break_even {
        Some(length) => {
            println!("fusion break-even: for chain length L ≥ {length}, **fused** GPU beats CPU\n");
        }
        None => println!("no fusion break-even found within measured chain lengths\n"),
    }
    Ok(())
}


const FUSABLE_CHAIN: &str = r#"(module
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $x i32) (param $y i32) (param $n i32)
    (local $i i32)
    (block $e1 (loop $l1
      (br_if $e1 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l1)))
    (local.set $i (i32.const 0))
    (block $e2 (loop $l2
      (br_if $e2 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $y) (i32.mul (local.get $i) (i32.const 4)))
        (i32.mul
          (i32.load (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4))))
          (i32.const 2)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l2))))
  (export "main" (func $prog)))"#;

const SHARED_INTERMEDIATE: &str = r#"(module
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $x i32) (param $y i32) (param $n i32)
    (local $i i32) (local $z i32)
    (block $e1 (loop $l1
      (br_if $e1 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l1)))
    (local.set $i (i32.const 0))
    (block $e2 (loop $l2
      (br_if $e2 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $y) (i32.mul (local.get $i) (i32.const 4)))
        (i32.mul
          (i32.load (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4))))
          (i32.const 2)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l2)))
    (local.set $i (i32.const 0))
    (block $e3 (loop $l3
      (br_if $e3 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $z) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4))))
          (i32.const 1)))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l3))))
  (export "main" (func $prog)))"#;

const SINGLE_ONLY: &str = r#"(module
  (memory (export "mem") 16)
  (func $prog (param $a i32) (param $b i32) (param $x i32) (param $y i32) (param $n i32)
    (local $i i32)
    (block $e1 (loop $l1
      (br_if $e1 (i32.ge_s (local.get $i) (local.get $n)))
      (i32.store (i32.add (local.get $x) (i32.mul (local.get $i) (i32.const 4)))
        (i32.add
          (i32.load (i32.add (local.get $a) (i32.mul (local.get $i) (i32.const 4))))
          (i32.load (i32.add (local.get $b) (i32.mul (local.get $i) (i32.const 4))))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $l1))))
  (export "main" (func $prog)))"#;


#[test]
fn exclusive_intermediate_is_decidable() -> TestResult {
    for (label, wat_source, expected) in [
        ("FUSABLE   X=A+B → Y=X*2", FUSABLE_CHAIN, 1_usize),
        (
            "SHARED    X has two readers (Y and Z)",
            SHARED_INTERMEDIATE,
            0,
        ),
        ("SINGLE    only one loop", SINGLE_ONLY, 0),
    ] {
        let bytes = wat::parse_str(wat_source)?;
        let mut module = load_module(&bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        let trace = Trace::silent();

        let mut reported = 0_usize;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let loops = collect_region_uses(body, &trace);
            let found = exclusive_intermediates(&loops);
            println!(
                "  {label}: {} loops, found {} pure intermediates {:?}",
                loops.len(),
                found.len(),
                found
                    .iter()
                    .map(|(region, producer, consumer)| format!(
                        "{region} (loop{producer}→loop{consumer})"
                    ))
                    .collect::<Vec<_>>()
            );
            reported = found.len();
        }
        assert_eq!(
            reported,
            expected,
            "{}: criterion reports {reported}, expected {expected} — criterion must separate positive/negative cases",
            label.split_whitespace().next().unwrap_or("")
        );
    }
    println!();
    Ok(())
}

#[test]
fn corpus_has_how_many_fusable_pairs() -> TestResult {
    let trace = Trace::silent();
    let mut with_loops = 0_usize;
    let mut multi_loop = 0_usize;
    let mut total_exclusive = 0_usize;
    let mut total_fusable = 0_usize;

    println!(
        "\n{:>36}  {:>5}  {:>9}  {:>9}",
        "case", "loops", "exclusive intermediates", "fusable pairs"
    );

    for case in CASES {
        let bytes = case.to_wasm()?;

        let normalized = normalize_bulk_memory(&bytes, &trace);
        let mut module = load_module(&normalized, &trace)?;
        expand_all_bodies(&mut module, &trace)?;

        let mut loops = 0_usize;
        let mut exclusive = 0_usize;
        let mut fusable = 0_usize;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let collected = collect_region_uses(body, &trace);
            loops += collected.len();
            exclusive += exclusive_intermediates(&collected).len();
            fusable += fusable_pairs(body, &collected).len();
        }

        if loops > 0 {
            with_loops += 1;
        }
        if loops > 1 {
            multi_loop += 1;
        }
        total_exclusive += exclusive;
        total_fusable += fusable;

        let expected = match case.name {
            "fusable/chain2" => Some((2_usize, 1_usize, 1_usize)),
            "fusable/chain3" => Some((3, 2, 2)),
            "fusion/extra_reader" => Some((3, 0, 0)),

            "fusion/cycle_swap" => Some((2, 2, 0)),
            _ => None,
        };
        if let Some((want_loops, want_exclusive, want_fusable)) = expected {
            assert_eq!(
                    (loops, exclusive, fusable),
                    (want_loops, want_exclusive, want_fusable),
                    "{}: criterion reports ({loops}, {exclusive}, {fusable}), expected ({want_loops}, {want_exclusive}, {want_fusable})",
                    case.name
                );
        }
        if loops > 1 || exclusive > 0 {
            println!(
                "{:>36}  {loops:>5}  {exclusive:>9}  {fusable:>9}",
                case.name
            );
        }

        if exclusive > 0 && fusable == 0 {
            for func in module.funcs.iter() {
                let Some(body) = module.funcs[func].body() else {
                    continue;
                };
                let collected = collect_region_uses(body, &trace);
                let candidates = exclusive_intermediates(&collected);
                let edges: Vec<(usize, usize)> = candidates
                    .iter()
                    .map(|(_, producer, consumer)| (*producer, *consumer))
                    .collect();
                for (region, producer, consumer) in candidates {
                    let space_ok =
                        same_iteration_space(body, &collected[producer].1, &collected[consumer].1);
                    let cyclic = edges.iter().any(|(a, b)| *a == consumer && *b == producer);
                    println!(
                        "    DIAG {region} loop{producer}→loop{consumer}: \
                             same iteration space={space_ok} cyclic dep={cyclic}"
                    );
                }
            }
        }
    }

    println!(
        "\nstats: {} cases have loops, of which {multi_loop} are **multi-loop**; \
             {total_exclusive} exclusive intermediates, **{total_fusable} fusable pairs**",
        with_loops
    );
    if total_fusable == 0 {
        println!(
            "→ **this corpus has nothing fusable.** Fusion priority must be reassessed:\n  \
                 fusing on single-loop-only cases has nothing verifiable."
        );
    } else {
        println!("→ criterion also passed per-case asserts on the **accept** side (positives recognized, negatives rejected).");
    }
    println!();
    Ok(())
}


#[test]
fn fusion_closes_the_loop_with_evidence() -> TestResult {
    use super::conformance::gpu;
    use std::time::Instant;

    const N: i32 = 1_000_000;
    const ROUNDS: usize = 15;


    let case = CASES
        .iter()
        .find(|case| case.name == "fusable/chain2")
        .ok_or("case is not registered")?;
    let bytes = case.to_wasm()?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let trace = Trace::silent();

    let mut kernels = Vec::new();
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        let collected = collect_region_uses(body, &trace);
        let fusable = fusable_pairs(body, &collected);
        println!(
            "\ncriterion: {} loops, {} fusable pairs {:?}",
            collected.len(),
            fusable.len(),
            fusable
                .iter()
                .map(|(region, producer, consumer)| format!("{region} loop{producer}→{consumer}"))
                .collect::<Vec<_>>()
        );
        assert!(
            !fusable.is_empty(),
            "criterion must recognize chain2's fusable pair"
        );

        for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            if let Ok(kernel) = lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            ) {
                kernels.push(kernel);
            }
        }
        break;
    }
    assert_eq!(kernels.len(), 2, "chain2 should emit two kernels");

    let (producer, consumer) = (&kernels[0], &kernels[1]);
    println!(
        "producer field count {}, consumer field count {}",
        producer.fields.len(),
        consumer.fields.len()
    );
    let plan = fuse_two_chain(producer, consumer, producer.fields.len(), "X")?;
    println!(
        "fusion eliminates intermediate {}\n--- fused kernel ---\n{}",
        plan.removed_intermediate, plan.fused.source
    );


    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let producer_compiled = context
        .compile(producer)
        .map_err(|error| error.to_string())?;
    let consumer_compiled = context
        .compile(consumer)
        .map_err(|error| error.to_string())?;
    let fused_compiled = context
        .compile(&plan.fused)
        .map_err(|error| error.to_string())?;

    let used = (N as usize) * 16;
    let seed: Vec<u8> = (0..used).map(|index| (index * 37 + 11) as u8).collect();

    let a = 0_u32;
    let b = (N as u32) * 4;
    let x = (N as u32) * 8;
    let y = (N as u32) * 8;


    let producer_params = [N as u32, x, a, b];

    let consumer_params = [N as u32, y, x];

    let fused_params = [N as u32, y, a, b];

    let producer_workspace = context
        .workspace(used, producer_params.len() * 4)
        .map_err(|error| error.to_string())?;
    let consumer_workspace = context
        .workspace(used, consumer_params.len() * 4)
        .map_err(|error| error.to_string())?;
    let fused_workspace = context
        .workspace(used, fused_params.len() * 4)
        .map_err(|error| error.to_string())?;


    let after_producer = context
        .run_in(
            &producer_compiled,
            producer,
            &producer_workspace,
            &seed,
            &producer_params,
        )
        .map_err(|error| error.to_string())?;
    let unfused = context
        .run_in(
            &consumer_compiled,
            consumer,
            &consumer_workspace,
            &after_producer,
            &consumer_params,
        )
        .map_err(|error| error.to_string())?;
    let fused = context
        .run_in(
            &fused_compiled,
            &plan.fused,
            &fused_workspace,
            &seed,
            &fused_params,
        )
        .map_err(|error| error.to_string())?;

    let mismatches = unfused
        .iter()
        .zip(fused.iter())
        .filter(|(l, r)| l != r)
        .count();
    println!(
        "\nresult check: unfused (2 dispatches) vs fused (1) {} bytes → {mismatches} mismatches",
        unfused.len()
    );
    assert_eq!(mismatches, 0, "fused result mismatches unfused");


    let mut broken = plan.fused.clone();
    broken.source = broken.source.replace("* 3u", "* 5u");
    assert_ne!(
        broken.source, plan.fused.source,
        "corruption action had no effect"
    );
    let broken_compiled = context
        .compile(&broken)
        .map_err(|error| error.to_string())?;
    let broken_workspace = context
        .workspace(used, fused_params.len() * 4)
        .map_err(|error| error.to_string())?;
    let broken_out = context
        .run_in(
            &broken_compiled,
            &broken,
            &broken_workspace,
            &seed,
            &fused_params,
        )
        .map_err(|error| error.to_string())?;
    let differing = unfused
        .iter()
        .zip(broken_out.iter())
        .filter(|(l, r)| l != r)
        .count();
    assert!(
        differing > 0,
        "the broken fused kernel must be caught by the check"
    );
    println!("negative control: after changing one constant, {differing} bytes differ → check has discriminating power");


    let median = |mut values: Vec<f64>| {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values[values.len() / 2]
    };
    let mut unfused_times = Vec::new();
    let mut fused_times = Vec::new();
    for round in 0..ROUNDS {
        let started = Instant::now();
        let mid = context
            .run_in(
                &producer_compiled,
                producer,
                &producer_workspace,
                &seed,
                &producer_params,
            )
            .map_err(|error| error.to_string())?;
        context
            .run_in(
                &consumer_compiled,
                consumer,
                &consumer_workspace,
                &mid,
                &consumer_params,
            )
            .map_err(|error| error.to_string())?;
        if round > 0 {
            unfused_times.push(started.elapsed().as_secs_f64() * 1e6);
        }

        let started = Instant::now();
        context
            .run_in(
                &fused_compiled,
                &plan.fused,
                &fused_workspace,
                &seed,
                &fused_params,
            )
            .map_err(|error| error.to_string())?;
        if round > 0 {
            fused_times.push(started.elapsed().as_secs_f64() * 1e6);
        }
    }
    let unfused_median = median(unfused_times);
    let fused_median = median(fused_times);
    println!(
        "time: unfused {unfused_median:.1} µs vs fused {fused_median:.1} µs → fused {:.2}× faster\n",
        unfused_median / fused_median
    );
    assert!(
        fused_median < unfused_median,
        "fused should be faster than unfused (two transfers vs one)"
    );
    Ok(())
}

#[test]
fn binary_wasm_input_survives_automatic_rewrite() -> TestResult {
    use super::conformance::gpu;
    use waffle::{ConstVal, InterpContext};

    let engine = wasmtime::Engine::default();
    let mut checked = 0_usize;


    for name in ["pointwise/vector_add"] {
        let case = CASES
            .iter()
            .find(|case| case.name == name)
            .ok_or("case is not registered")?;


        let original_binary = case.to_wasm()?;
        assert!(
            std::str::from_utf8(&original_binary)
                .map(|text| !text.trim_start().starts_with("(module"))
                .unwrap_or(true),
            "{name}: this must be a binary, otherwise we are not testing the binary path"
        );


        let disassembled = wasmprinter::print_bytes(&original_binary)?;
        assert!(
            disassembled.contains("(module"),
            "{name}: disassembly failed"
        );


        let arity = {
            let mut probe = load_module(&original_binary, &Trace::silent())?;
            expand_all_bodies(&mut probe, &Trace::silent())?;
            plan_gpu_arity(&probe, &Trace::silent())?.fields
        };

        let source_bytes = wat::parse_str(&disassembled)?;
        let injected_bytes = inject_dispatch_import(&source_bytes, arity, 0, &Trace::silent())?;
        let mut module = load_module(&injected_bytes, &Trace::silent())?;
        expand_all_bodies(&mut module, &Trace::silent())?;
        plan_first_gpu_loop(&module, &Trace::silent())?;

        let mut kernels = Vec::new();
        let mut target = None;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            let trace = Trace::silent();
            for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                let recovery = AddressRecovery::analyze(body, &evolution, &trace);
                let accesses: Vec<_> = recovery
                    .accesses()
                    .iter()
                    .filter(|access| natural_loop.blocks.contains(&access.block))
                    .cloned()
                    .collect();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
                if let Ok(kernel) = lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                ) {
                    let indices =
                        header_parameter_indices(body, natural_loop.header, &kernel.fields)?;
                    let members = natural_loop.blocks.clone();
                    let header = natural_loop.header;
                    kernels.push((func, header, members, indices, kernel));
                    target = Some(target.unwrap_or((func, header)));
                }
            }
            break;
        }
        assert!(!kernels.is_empty(), "{name}: no emittable loop found");

        rewrite_module_for_gpu(&mut module, &Trace::silent())?;
        let rewritten = module.to_wasm_bytes()?;
        let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;


        let mut params = vec![8_u32; kernels[0].4.fields.len()];
        for (slot, value) in params.iter_mut().enumerate().skip(1) {
            *value = (slot as u32) * 64;
        }


        let memory_bytes = 65_536_usize;
        let seed: Vec<u8> = (0..memory_bytes)
            .map(|index| (index * 37 + 11) as u8)
            .collect();
        let mut original_module = load_module(&original_binary, &Trace::silent())?;
        expand_all_bodies(&mut original_module, &Trace::silent())?;


        let (entry_func, param_count) = original_module
            .funcs
            .iter()
            .find_map(|f| {
                original_module.funcs[f]
                    .body()
                    .map(|body| (f, body.n_params))
            })
            .ok_or("missing function body")?;
        let mut full_args = vec![ConstVal::I32(0); param_count];
        for (index, value) in params.iter().enumerate() {
            if index + 1 < full_args.len() {
                full_args[index + 1] = ConstVal::I32(*value);
            }
        }
        if let Some(last) = full_args.last_mut() {
            *last = ConstVal::I32(8);
        }

        let mut context = InterpContext::new(&original_module)?;
        let memory = original_module
            .memories
            .iter()
            .next()
            .ok_or("case must contain memory")?;
        context.memories[memory].data.copy_from_slice(&seed);
        let call_result = context.call(&original_module, entry_func, &full_args);
        if call_result.ok().is_err() {
            println!("  {name}: original binary trapped, skipped");
            continue;
        }
        let reference = context.memories[memory].data.clone();


        struct Host {
            context: gpu::GpuContext,
            compiled: gpu::Compiled,
            kernel: Kernel,
        }
        let kernel = kernels[0].4.clone();
        let gpu_context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
        let compiled = gpu_context
            .compile(&kernel)
            .map_err(|error| error.to_string())?;

        let mut linker = wasmtime::Linker::new(&engine);
        linker.func_new(
            "heterowasm",
            "__hw_dispatch",
            wasmtime::FuncType::new(
                &engine,
                (0..arity + 1).map(|_| wasmtime::ValType::I32),
                std::iter::empty::<wasmtime::ValType>(),
            ),
            |mut caller: wasmtime::Caller<'_, Host>,
             incoming: &[wasmtime::Val],
             _results: &mut [wasmtime::Val]|
             -> Result<(), wasmtime::Error> {
                let memory = caller
                    .get_export("mem")
                    .and_then(|item| item.into_memory())
                    .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
                let data = memory.data(&caller).to_vec();
                let host = caller.data();
                let mut params: Vec<u32> = incoming
                    .iter()
                    .map(|value| value.i32().unwrap_or(0) as u32)
                    .collect();

                params.pop();
                let workspace = host
                    .context
                    .workspace(data.len(), params.len() * 4)
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                let produced = host
                    .context
                    .run_in(&host.compiled, &host.kernel, &workspace, &data, &params)
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                memory.write(&mut caller, 0, &produced)?;
                Ok(())
            },
        )?;

        let mut store = wasmtime::Store::new(
            &engine,
            Host {
                context: gpu_context,
                compiled,
                kernel,
            },
        );
        let instance = linker.instantiate(&mut store, &rewritten_module)?;
        let entry = instance
            .get_func(&mut store, "main")
            .ok_or("missing main")?;
        let memory = instance
            .get_memory(&mut store, "mem")
            .ok_or("missing memory")?;
        memory.write(&mut store, 0, &seed)?;
        let values: Vec<wasmtime::Val> = full_args
            .iter()
            .map(|value| {
                let raw = match value {
                    ConstVal::I32(v) => *v as i32,
                    _ => 0,
                };
                wasmtime::Val::I32(raw)
            })
            .collect();
        entry.call(&mut store, &values, &mut [])?;
        let produced = memory.data(&store).to_vec();

        let mismatches = reference
            .iter()
            .zip(produced.iter())
            .filter(|(left, right)| left != right)
            .count();
        println!(
                "  {name}: binary {} bytes → disasm → inject → rewrite → {} bytes; {mismatches} mismatches vs original binary",
                original_binary.len(),
                rewritten.len()
            );
        assert_eq!(
            mismatches, 0,
            "{name}: binary-path result mismatches original execution"
        );
        checked += 1;
    }

    assert!(checked > 0, "at least one binary input must succeed");
    Ok(())
}

#[test]
fn fused_rewrite_matches_original_program() -> TestResult {
    use super::conformance::gpu;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use waffle::{ConstVal, InterpContext};

    let case = CASES
        .iter()
        .find(|case| case.name == "fusable/chain2")
        .ok_or("case is not registered")?;
    let engine = wasmtime::Engine::default();


    let reference_binary = case.to_wasm()?;
    let mut reference_module = load_module(&reference_binary, &Trace::silent())?;
    expand_all_bodies(&mut reference_module, &Trace::silent())?;
    let param_count = reference_module
        .funcs
        .iter()
        .find_map(|f| reference_module.funcs[f].body().map(|body| body.n_params))
        .ok_or("missing function body")?;
    let entry_func = reference_module
        .funcs
        .iter()
        .find(|f| reference_module.funcs[*f].body().is_some())
        .ok_or("missing function body")?;

    let mut full_args = vec![ConstVal::I32(0); param_count];
    for (index, value) in [0_u32, 256, 512, 768, 1024, 1280].iter().enumerate() {
        if index < full_args.len() {
            full_args[index] = ConstVal::I32(*value);
        }
    }
    if let Some(last) = full_args.last_mut() {
        *last = ConstVal::I32(8);
    }

    let mut context = InterpContext::new(&reference_module)?;
    let memory = reference_module
        .memories
        .iter()
        .next()
        .ok_or("case must contain memory")?;
    let seed: Vec<u8> = (0..context.memories[memory].data.len())
        .map(|index| (index * 37 + 11) as u8)
        .collect();
    context.memories[memory].data.copy_from_slice(&seed);
    context
        .call(&reference_module, entry_func, &full_args)
        .ok()?;
    let reference = context.memories[memory].data.clone();


    let disassembled = wasmprinter::print_bytes(&reference_binary)?;
    let arity = {
        let mut probe = load_module(&reference_binary, &Trace::silent())?;
        expand_all_bodies(&mut probe, &Trace::silent())?;
        plan_gpu_arity(&probe, &Trace::silent())?.fields
    };

    let source_bytes = wat::parse_str(&disassembled)?;
    let injected_bytes = inject_dispatch_import(&source_bytes, arity, 0, &Trace::silent())?;
    let mut module = load_module(&injected_bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;


    let trace = Trace::silent();
    let mut kernels = Vec::new();
    for func in module.funcs.iter() {
        let Some(body) = module.funcs[func].body() else {
            continue;
        };
        for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
            let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
            let recovery = AddressRecovery::analyze(body, &evolution, &trace);
            let accesses: Vec<_> = recovery
                .accesses()
                .iter()
                .filter(|access| natural_loop.blocks.contains(&access.block))
                .cloned()
                .collect();
            let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
            if let Ok(kernel) = lower(
                body,
                &natural_loop,
                &accesses,
                &evolution,
                extent,
                64,
                &trace,
            ) {
                kernels.push(kernel);
            }
        }
        break;
    }
    assert_eq!(kernels.len(), 2, "chain2 should emit two kernels");
    let plan = fuse_two_chain(&kernels[0], &kernels[1], kernels[0].fields.len(), "X")?;

    let outcome = rewrite_module_fused(&mut module, &trace)?;
    println!(
        "\nrewrite: extracted {} loops, fused = {}",
        outcome.loops_removed, outcome.fused
    );
    assert_eq!(outcome.loops_removed, 2);
    assert!(outcome.fused);

    let rewritten_module = wasmtime::Module::new(&engine, &module.to_wasm_bytes()?)?;

    struct Host {
        context: gpu::GpuContext,
        compiled: gpu::Compiled,
        kernel: Kernel,
        calls: AtomicUsize,
    }
    let gpu_context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = gpu_context
        .compile(&plan.fused)
        .map_err(|error| error.to_string())?;

    let mut linker = wasmtime::Linker::new(&engine);
    linker.func_new(
        "heterowasm",
        "__hw_dispatch",
        wasmtime::FuncType::new(
            &engine,
            (0..arity + 1).map(|_| wasmtime::ValType::I32),
            std::iter::empty::<wasmtime::ValType>(),
        ),
        |mut caller: wasmtime::Caller<'_, Host>,
         incoming: &[wasmtime::Val],
         _results: &mut [wasmtime::Val]|
         -> Result<(), wasmtime::Error> {
            caller.data().calls.fetch_add(1, Ordering::SeqCst);
            let memory = caller
                .get_export("mem")
                .and_then(|item| item.into_memory())
                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
            let data = memory.data(&caller).to_vec();
            let host = caller.data();
            let mut params: Vec<u32> = incoming
                .iter()
                .map(|value| value.i32().unwrap_or(0) as u32)
                .collect();

            params.pop();
            let workspace = host
                .context
                .workspace(data.len(), params.len() * 4)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            let produced = host
                .context
                .run_in(&host.compiled, &host.kernel, &workspace, &data, &params)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            memory.write(&mut caller, 0, &produced)?;
            Ok(())
        },
    )?;

    let mut store = wasmtime::Store::new(
        &engine,
        Host {
            context: gpu_context,
            compiled,
            kernel: plan.fused.clone(),
            calls: AtomicUsize::new(0),
        },
    );
    let instance = linker.instantiate(&mut store, &rewritten_module)?;
    let entry = instance
        .get_func(&mut store, "main")
        .ok_or("missing main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    let values: Vec<wasmtime::Val> = full_args
        .iter()
        .map(|value| match value {
            ConstVal::I32(raw) => wasmtime::Val::I32(*raw as i32),
            _ => wasmtime::Val::I32(0),
        })
        .collect();
    entry.call(&mut store, &values, &mut [])?;
    let produced = memory.data(&store).to_vec();
    let call_count = store.data().calls.load(Ordering::SeqCst);

    let differing: Vec<usize> = reference
        .iter()
        .zip(produced.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .map(|(index, _)| index)
        .collect();
    println!(
        "host import called {call_count} times; vs original {} bytes → {} mismatches",
        reference.len(),
        differing.len()
    );
    if let (Some(first), Some(last)) = (differing.first(), differing.last()) {
        println!("  mismatch range {first}..{last}; 64-byte block {:?}", {
            let mut buckets = std::collections::BTreeSet::new();
            for index in &differing {
                buckets.insert(index / 64);
            }
            buckets.iter().take(8).copied().collect::<Vec<_>>()
        });
    }
    for (label, base) in [("X(768)", 768_usize), ("Y(1024)", 1024)] {
        println!(
            "  {label} reference {:?} got {:?}",
            &reference[base..base + 8],
            &produced[base..base + 8]
        );
    }
    assert_eq!(call_count, 1, "fusion must produce exactly one dispatch");


    let intermediate = 768_usize..800_usize;
    let outside: Vec<usize> = differing
        .iter()
        .copied()
        .filter(|index| !intermediate.contains(index))
        .collect();
    println!(
        "  intermediate {} differs in {} places (not written); **remaining {} places**",
        intermediate.len(),
        differing.len(),
        outside.len()
    );
    assert!(
        outside.is_empty(),
        "must match everywhere except the intermediate; actually {} diffs: {:?}",
        outside.len(),
        &outside[..outside.len().min(8)]
    );
    assert_eq!(
        differing.len(),
        intermediate.len(),
        "difference must be **exactly** the intermediate segment — fewer means it was written; more means other corruption"
    );
    assert_eq!(
        &produced[intermediate.clone()],
        &seed[intermediate],
        "the intermediate segment must keep its initial value (truly not written)"
    );
    Ok(())
}

#[test]
fn end_to_end_fusion_versus_unfused() -> TestResult {
    use super::conformance::gpu;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use waffle::ConstVal;


    const N: u32 = 40_000;
    const ROUNDS: usize = 11;
    let engine = wasmtime::Engine::default();

    let case = CASES
        .iter()
        .find(|case| case.name == "fusable/chain2")
        .ok_or("case is not registered")?;
    let original_binary = case.to_wasm()?;
    let disassembled = wasmprinter::print_bytes(&original_binary)?;
    let arity = {
        let mut probe = load_module(&original_binary, &Trace::silent())?;
        expand_all_bodies(&mut probe, &Trace::silent())?;
        plan_gpu_arity(&probe, &Trace::silent())?.fields
    };

    let source_bytes = wat::parse_str(&disassembled)?;
    let injected_bytes = inject_dispatch_import(&source_bytes, arity, 0, &Trace::silent())?;


    let memory_sink = std::sync::Arc::new(heterowasm_trace::MemorySink::default());

    let memory_erased: std::sync::Arc<dyn Sink> =
        std::sync::Arc::<heterowasm_trace::MemorySink>::clone(&memory_sink);
    let jsonl_path = std::env::temp_dir().join("heterowasm-two-dispatch.jsonl");
    let jsonl_sink = std::sync::Arc::new(
        heterowasm_trace::JsonlSink::create(&jsonl_path).map_err(|e| e.to_string())?,
    );
    let jsonl_erased: std::sync::Arc<dyn Sink> = jsonl_sink;
    let trace = Trace::with_sinks(vec![memory_erased, jsonl_erased], Level::Debug);
    let sink = memory_sink;
    println!("evidence written: {}", jsonl_path.display());


    let mut kernels = Vec::new();
    {
        let mut module = load_module(&injected_bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                let recovery = AddressRecovery::analyze(body, &evolution, &trace);
                let accesses: Vec<_> = recovery
                    .accesses()
                    .iter()
                    .filter(|access| natural_loop.blocks.contains(&access.block))
                    .cloned()
                    .collect();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
                if let Ok(kernel) = lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                ) {
                    kernels.push(kernel);
                }
            }
            break;
        }
    }
    assert_eq!(kernels.len(), 2, "chain2 should emit two kernels");
    let fused_plan = fuse_two_chain(&kernels[0], &kernels[1], kernels[0].fields.len(), "X")?;


    let mut fused_module = load_module(&injected_bytes, &trace)?;
    expand_all_bodies(&mut fused_module, &trace)?;
    let outcome = rewrite_module_fused(&mut fused_module, &trace)?;
    assert!(
        outcome.fused && outcome.loops_removed == 2,
        "path B must use fusion"
    );
    let fused_wasm = fused_module.to_wasm_bytes()?;


    let mut unfused_module = load_module(&injected_bytes, &trace)?;
    expand_all_bodies(&mut unfused_module, &trace)?;
    let unfused_outcome = rewrite_module_two_dispatches(&mut unfused_module, &trace)?;
    assert_eq!(
        unfused_outcome.loops_removed, 2,
        "baseline must also extract both loops"
    );
    let unfused_wasm = unfused_module.to_wasm_bytes()?;


    struct Host {
        context: gpu::GpuContext,

        compiled: Vec<gpu::Compiled>,
        kernels: Vec<Kernel>,
        calls: AtomicUsize,

        trace: Trace,
    }


    let mut full_args = vec![ConstVal::I32(0); 7];
    for (index, value) in [0_u32, N * 4, N * 8, N * 12, N * 16, N * 20]
        .iter()
        .enumerate()
    {
        full_args[index] = ConstVal::I32(*value);
    }
    full_args[6] = ConstVal::I32(N);
    let call_values: Vec<wasmtime::Val> = full_args
        .iter()
        .map(|value| match value {
            ConstVal::I32(raw) => wasmtime::Val::I32(*raw as i32),
            _ => wasmtime::Val::I32(0),
        })
        .collect();
    let seed: Vec<u8> = (0..(1 << 20))
        .map(|index| (index * 37 + 11) as u8)
        .collect();


    let measure = |wasm: &[u8],
                   kernels: Vec<Kernel>,
                   expect_calls: usize,
                   label: &str,
                   rounds: usize|
     -> Result<Vec<f64>, String> {
        let module = wasmtime::Module::new(&engine, wasm).map_err(|e| e.to_string())?;

        trace
            .info(Stage::Runtime, "measured module loaded")
            .subject(label.to_string())
            .field("wasm_bytes", wasm.len())
            .field("expect_calls", expect_calls)
            .emit();
        let mut times = Vec::new();
        let mut observed = 0_usize;
        for round in 0..rounds {

            let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
            let compiled: Vec<gpu::Compiled> = kernels
                .iter()
                .map(|kernel| context.compile(kernel).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            let mut linker = wasmtime::Linker::new(&engine);
            linker
                .func_new(
                    "heterowasm",
                    "__hw_dispatch",
                    wasmtime::FuncType::new(
                        &engine,
                        (0..arity + 1).map(|_| wasmtime::ValType::I32),
                        std::iter::empty::<wasmtime::ValType>(),
                    ),
                    move |mut caller: wasmtime::Caller<'_, Host>,
                          incoming: &[wasmtime::Val],
                          _results: &mut [wasmtime::Val]|
                          -> Result<(), wasmtime::Error> {
                        let index = caller.data().calls.fetch_add(1, Ordering::SeqCst);
                        let memory = caller
                            .get_export("mem")
                            .and_then(|item| item.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;
                        let data = memory.data(&caller).to_vec();
                        let all: Vec<u32> = incoming
                            .iter()
                            .map(|value| value.i32().unwrap_or(0) as u32)
                            .collect();
                        let host = caller.data();
                        let slot = index.min(host.compiled.len() - 1);

                        let fields = host.kernels[slot].fields.len();
                        let params: Vec<u32> = all[..fields.min(all.len())].to_vec();

                        host.trace
                            .info(Stage::Runtime, "host dispatch entered")
                            .subject("__hw_dispatch")
                            .field("call_index", index)
                            .field("slot", slot)
                            .field("incoming", format!("{all:?}"))
                            .field("sliced", format!("{params:?}"))
                            .field("kernel_fields", fields)
                            .field("memory_bytes", data.len())
                            .emit();
                        let workspace = host
                            .context
                            .workspace(data.len(), params.len() * 4)
                            .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                        let produced = host
                            .context
                            .run_in(
                                &host.compiled[slot],
                                &host.kernels[slot],
                                &workspace,
                                &data,
                                &params,
                            )
                            .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                        memory.write(&mut caller, 0, &produced)?;
                        Ok(())
                    },
                )
                .map_err(|error| error.to_string())?;
            let mut store = wasmtime::Store::new(
                &engine,
                Host {
                    context,
                    compiled,
                    kernels: kernels.clone(),
                    calls: AtomicUsize::new(0),
                    trace: {
                        let erased: std::sync::Arc<dyn Sink> =
                            std::sync::Arc::<heterowasm_trace::MemorySink>::clone(&sink);
                        Trace::with_sink(erased, Level::Debug)
                    },
                },
            );
            let instance = linker
                .instantiate(&mut store, &module)
                .map_err(|e| e.to_string())?;
            let entry = instance
                .get_func(&mut store, "main")
                .ok_or("missing main")?;
            let memory = instance
                .get_memory(&mut store, "mem")
                .ok_or("missing memory")?;
            memory
                .write(&mut store, 0, &seed)
                .map_err(|e| e.to_string())?;


            let started = Instant::now();
            let outcome = entry.call(&mut store, &call_values, &mut []);
            let elapsed = started.elapsed().as_secs_f64() * 1e6;

            match &outcome {
                Ok(()) => {
                    trace
                        .info(Stage::Runtime, "dispatch call returned")
                        .subject(label.to_string())
                        .field("round", round)
                        .field("elapsed_us", format!("{elapsed:.0}"))
                        .field("ok", true)
                        .emit();
                }
                Err(error) => {
                    trace
                        .error(Stage::Runtime, "dispatch call failed")
                        .subject(label.to_string())
                        .field("round", round)
                        .field("ok", false)
                        .field("error_debug", format!("{error:?}"))
                        .field("error_display", format!("{error}"))
                        .field("calls_so_far", store.data().calls.load(Ordering::SeqCst))
                        .emit();
                }
            }
            outcome.map_err(|e| e.to_string())?;

            observed = store.data().calls.load(Ordering::SeqCst);
            if observed != expect_calls {
                return Err(format!(
                    "{label}: observed {observed} dispatches, expected {expect_calls}"
                ));
            }
            if round > 0 {
                times.push(elapsed);
            }
        }
        println!("{label}: {observed} dispatches");
        Ok(times)
    };

    let unfused_times = measure(
        &unfused_wasm,
        vec![kernels[0].clone(), kernels[1].clone()],
        2,
        "path A (unfused: one dispatch per loop)",
        ROUNDS,
    )?;
    let fused_times = measure(
        &fused_wasm,
        vec![fused_plan.fused.clone()],
        1,
        "path B (fused: two loops become one call)",
        ROUNDS,
    )?;

    let median = |mut values: Vec<f64>| {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values[values.len() / 2]
    };
    let unfused = median(unfused_times);
    let fused = median(fused_times);
    println!(
        "\nend-to-end (N = {N}, median of {ROUNDS} rounds):\n  \
             path A (unfused) {unfused:.0} µs — **2 dispatches** (one per loop)\n  \
             path B (fused)   {fused:.0} µs — **1 dispatch** (both loops in one)\n  \
             fused/unfused = {:.3} (<1 means fused is faster)\n",
        fused / unfused
    );


    let events = sink.events();
    println!("=== evidence bus: {} events ===", events.len());
    for event in &events {
        let level = event.level.as_str();
        let stage = event.stage.as_str();
        let subject = event.subject.as_deref().unwrap_or("-");
        let fields = event
            .fields
            .iter()
            .map(|(name, value)| format!("{name}={value:?}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "  [{level}][{stage}] {} | {subject} | {fields}",
            event.message
        );
    }


    let planned: Vec<&heterowasm_trace::Event> = events
        .iter()
        .filter(|event| event.subject.as_deref() == Some("two_dispatch"))
        .collect();
    assert!(
        !planned.is_empty(),
        "evidence bus has no two_dispatch event — data never arrived; any conclusion is vacuous"
    );
    for event in &planned {
        if let Some(actual) = event.int_field("call_sites") {
            assert_eq!(
                actual, 2,
                "rewritten artifact Call count is not 2 — insertion failed, and this must be visible in the log"
            );
        }
    }
    let host_events = events
        .iter()
        .filter(|event| event.subject.as_deref() == Some("__hw_dispatch"))
        .count();
    println!(
        "  host call events {host_events}; dispatch result events {}",
        events
            .iter()
            .filter(|event| event.message.contains("dispatch call"))
            .count()
    );
    assert!(
        host_events > 0,
        "evidence bus has no host-call event — data never arrived"
    );
    Ok(())
}

#[test]
fn locate_two_dispatch_trap() -> TestResult {
    use super::conformance::gpu;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use waffle::{ConstVal, InterpContext};

    const N: u32 = 8;
    let engine = wasmtime::Engine::default();
    let case = CASES
        .iter()
        .find(|case| case.name == "fusable/chain2")
        .ok_or("case is not registered")?;
    let original_binary = case.to_wasm()?;
    let disassembled = wasmprinter::print_bytes(&original_binary)?;
    let arity = {
        let mut probe = load_module(&original_binary, &Trace::silent())?;
        expand_all_bodies(&mut probe, &Trace::silent())?;
        plan_gpu_arity(&probe, &Trace::silent())?.fields
    };

    let source_bytes = wat::parse_str(&disassembled)?;
    let injected_bytes = inject_dispatch_import(&source_bytes, arity, 0, &Trace::silent())?;
    let trace = Trace::silent();


    let mut reference_module = load_module(&original_binary, &trace)?;
    expand_all_bodies(&mut reference_module, &trace)?;
    let entry_func = reference_module
        .funcs
        .iter()
        .find(|f| reference_module.funcs[*f].body().is_some())
        .ok_or("missing function body")?;

    let stride = N * 4;
    let mut full_args = vec![ConstVal::I32(0); 7];
    for (index, value) in [
        0_u32,
        stride,
        stride * 2,
        stride * 3,
        stride * 4,
        stride * 5,
    ]
    .iter()
    .enumerate()
    {
        full_args[index] = ConstVal::I32(*value);
    }
    full_args[6] = ConstVal::I32(N);
    let mut context = InterpContext::new(&reference_module)?;
    let memory = reference_module
        .memories
        .iter()
        .next()
        .ok_or("case must contain memory")?;

    let seed: Vec<u8> = (0..context.memories[memory].data.len())
        .map(|index| (index * 37 + 11) as u8)
        .collect();
    context.memories[memory].data.copy_from_slice(&seed);
    context
        .call(&reference_module, entry_func, &full_args)
        .ok()?;
    let reference = context.memories[memory].data.clone();


    let mut kernels = Vec::new();
    {
        let mut module = load_module(&injected_bytes, &trace)?;
        expand_all_bodies(&mut module, &trace)?;
        for func in module.funcs.iter() {
            let Some(body) = module.funcs[func].body() else {
                continue;
            };
            for natural_loop in ControlFlow::analyze(body, &trace).natural_loops(body, &trace) {
                let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
                let recovery = AddressRecovery::analyze(body, &evolution, &trace);
                let accesses: Vec<_> = recovery
                    .accesses()
                    .iter()
                    .filter(|access| natural_loop.blocks.contains(&access.block))
                    .cloned()
                    .collect();
                let extent = LoopExtent::analyze(body, &natural_loop, &evolution);
                if let Ok(kernel) = lower(
                    body,
                    &natural_loop,
                    &accesses,
                    &evolution,
                    extent,
                    64,
                    &trace,
                ) {
                    kernels.push(kernel);
                }
            }
            break;
        }
    }
    assert_eq!(kernels.len(), 2);


    let mut module = load_module(&injected_bytes, &trace)?;
    expand_all_bodies(&mut module, &trace)?;
    let outcome = rewrite_module_two_dispatches(&mut module, &trace)?;
    assert_eq!(outcome.loops_removed, 2);
    let rewritten = module.to_wasm_bytes()?;
    std::fs::write("/tmp/hw-two-dispatch.wasm", &rewritten)?;


    let rewritten_wat = wasmprinter::print_bytes(&rewritten)?;
    println!("\n=== rewritten wasm (two dispatches) ===\n{rewritten_wat}");

    let module = wasmtime::Module::new(&engine, &rewritten)?;

    struct Host {
        context: gpu::GpuContext,
        compiled: Vec<gpu::Compiled>,
        kernels: Vec<Kernel>,
        calls: AtomicUsize,
        trace: std::cell::RefCell<Vec<String>>,
    }
    let gpu_context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled: Vec<gpu::Compiled> = kernels
        .iter()
        .map(|kernel| {
            gpu_context
                .compile(kernel)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut linker = wasmtime::Linker::new(&engine);
    linker
        .func_new(
            "heterowasm",
            "__hw_dispatch",
            wasmtime::FuncType::new(
                &engine,
                (0..arity + 1).map(|_| wasmtime::ValType::I32),
                std::iter::empty::<wasmtime::ValType>(),
            ),
            |mut caller: wasmtime::Caller<'_, Host>,
             incoming: &[wasmtime::Val],
             _results: &mut [wasmtime::Val]|
             -> Result<(), wasmtime::Error> {
                let index = caller.data().calls.fetch_add(1, Ordering::SeqCst);
                let memory = caller
                    .get_export("mem")
                    .and_then(|item| item.into_memory())
                    .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;
                let data = memory.data(&caller).to_vec();
                let all: Vec<u32> = incoming
                    .iter()
                    .map(|value| value.i32().unwrap_or(0) as u32)
                    .collect();
                let host = caller.data();
                let slot = index.min(host.compiled.len() - 1);
                let params: Vec<u32> =
                    all[..host.kernels[slot].fields.len().min(all.len())].to_vec();
                host.trace
                    .borrow_mut()
                    .push(format!("call #{index} → kernel {slot}, params {params:?}"));
                let workspace = host
                    .context
                    .workspace(data.len(), params.len() * 4)
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                let produced = host
                    .context
                    .run_in(
                        &host.compiled[slot],
                        &host.kernels[slot],
                        &workspace,
                        &data,
                        &params,
                    )
                    .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
                memory.write(&mut caller, 0, &produced)?;
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;

    let mut store = wasmtime::Store::new(
        &engine,
        Host {
            context: gpu_context,
            compiled,
            kernels,
            calls: AtomicUsize::new(0),
            trace: std::cell::RefCell::new(Vec::new()),
        },
    );
    let instance = linker.instantiate(&mut store, &module)?;
    let entry = instance
        .get_func(&mut store, "main")
        .ok_or("missing main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    let values: Vec<wasmtime::Val> = full_args
        .iter()
        .map(|value| match value {
            ConstVal::I32(raw) => wasmtime::Val::I32(*raw as i32),
            _ => wasmtime::Val::I32(0),
        })
        .collect();
    let result = entry.call(&mut store, &values, &mut []);
    let calls = store.data().calls.load(Ordering::SeqCst);
    let logged = store.data().trace.borrow().clone();
    println!("host call log: {logged:?}");
    println!("host was called {calls} times");
    match result {
        Ok(()) => {
            let produced = memory.data(&store).to_vec();
            let mismatches = reference
                .iter()
                .zip(produced.iter())
                .filter(|(left, right)| left != right)
                .count();
            let where_: Vec<usize> = reference
                .iter()
                .zip(produced.iter())
                .enumerate()
                .filter(|(_, (left, right))| left != right)
                .map(|(index, _)| index)
                .collect();
            println!("**ran successfully**; {mismatches} mismatches vs original → at {where_:?}");
            println!(
                "  X(48) reference {:?} got {:?}\n  Y(64) reference {:?} got {:?}",
                &reference[48..56],
                &produced[48..56],
                &reference[64..72],
                &produced[64..72]
            );
        }
        Err(error) => {
            panic!("**two-dispatch wasm trapped**: {error:?}");
        }
    }
    Ok(())
}


#[test]
fn injection_must_renumber_every_function_reference() -> TestResult {
    use waffle::wasmparser::{Operator, Parser, Payload};


    let source = wat::parse_str(
        r#"(module
             (func $callee (result i32) (i32.const 7))
             (func $caller (result i32) (call $callee))
             (export "callee" (func $callee)))"#,
    )?;
    let injected = inject_dispatch_import(&source, 3, 0, &Trace::silent())?;

    let mut import_names: Vec<(String, String, usize)> = Vec::new();
    let mut calls: Vec<u32> = Vec::new();
    let mut exports: Vec<u32> = Vec::new();
    let mut defined = 0usize;
    for payload in Parser::new(0).parse_all(&injected) {
        match payload? {
            Payload::ImportSection(reader) => {
                for imports in reader {
                    match imports? {
                        waffle::wasmparser::Imports::Single(_, import) => {
                            if let waffle::wasmparser::TypeRef::Func(index) = import.ty {
                                import_names.push((
                                    import.module.to_string(),
                                    import.name.to_string(),
                                    index as usize,
                                ));
                            }
                        }
                        other => panic!("unexpected import encoding shape: {other:?}"),
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for entry in reader {
                    entry?;
                    defined += 1;
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export?;
                    if let waffle::wasmparser::ExternalKind::Func = export.kind {
                        exports.push(export.index);
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let mut reader = body.get_operators_reader()?;
                while !reader.eof() {
                    if let Operator::Call { function_index } = reader.read()? {
                        calls.push(function_index);
                    }
                }
            }
            _ => {}
        }
    }

    assert_eq!(
        import_names.len(),
        1,
        "after injection there should be only our one import"
    );
    assert_eq!(import_names[0].0, "heterowasm");
    assert_eq!(import_names[0].1, "__hw_dispatch");
    assert_eq!(
        defined, 2,
        "both functions of the original module must still be present"
    );
    assert_eq!(
        exports,
        vec![1],
        "export table must point at the shifted index (old callee 0 ⇒ now 1)"
    );
    assert_eq!(
        calls,
        vec![1],
        "`call` must point at the shifted index (old callee 0 ⇒ now 1); without renumbering this stays 0"
    );
    Ok(())
}


#[test]
fn kernel_results_must_match_the_emitted_wgsl() -> TestResult {
    let mut accepted = 0_usize;
    let mut with_results = 0_usize;
    for case in CASES {
        let Ok(kernel) = lower_first_loop(case.name) else {
            continue;
        };
        accepted += 1;
        let declares = kernel.source.contains("var<storage, read_write> results");
        let writes = kernel.source.contains("results[");
        assert_eq!(
            kernel.results.is_empty(),
            !declares,
            "{}: `results` has {} entries, but WGSL {}declares bindings",
            case.name,
            kernel.results.len(),
            if declares { "yes" } else { "no" }
        );
        assert_eq!(
            declares, writes,
            "{}: if a binding is declared it must be written; if written it must be declared",
            case.name
        );
        if !kernel.results.is_empty() {
            with_results += 1;

            for position in 0..kernel.results.len() {
                assert!(
                    kernel.source.contains(&format!("results[{position}] = ")),
                    "{}: exit value at position {position} was not written out",
                    case.name
                );
            }
        }
    }
    assert!(
        accepted > 0,
        "case has no emittable loop — this test would become a no-op"
    );
    println!("case has {accepted} emittable loops, of which {with_results} carry exit values");
    Ok(())
}


#[test]
fn value_and_address_must_agree_on_the_induction_scaling() -> TestResult {
    let kernel = lower_bytes(&wat::parse_str(
        r#"(module
             (memory 1 1)
             (func $running_store (param $seed i32)
               (local $i i32)
               (local $acc i32)
               (local.set $acc (local.get $seed))
               (block $exit
                 (loop $loop
                   (br_if $exit (i32.ge_s (local.get $i) (i32.const 8)))
                   (i32.store
                     (i32.mul (local.get $i) (i32.const 4))
                     (local.get $acc))
                   (local.set $acc (i32.add (local.get $acc) (i32.const 1)))
                   (local.set $i (i32.add (local.get $i) (i32.const 1)))
                   (br $loop)
                 )
               )
             )
             (export "main" (func $running_store))
             (export "mem" (memory 0)))"#,
    )?)?;


    fn coefficient_of_i(text: &str) -> Option<i64> {
        if let Some(rest) = text.split("i * ").nth(1) {
            let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
            return digits.parse().ok();
        }

        text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|token| token == "i")
            .then_some(1)
    }

    let store = kernel
        .source
        .lines()
        .find(|line| line.trim_start().starts_with("mem["))
        .ok_or("emitted WGSL has no store")?
        .trim()
        .to_string();
    let (address, rhs) = store.split_once(" = ").ok_or("no equals sign in store")?;
    let address_coefficient =
        coefficient_of_i(address).ok_or_else(|| format!("address side has no i: {address}"))?;
    let value_coefficient =
        coefficient_of_i(rhs).ok_or_else(|| format!("value side has no i: {rhs}"))?;

    assert_eq!(
        address_coefficient, 1,
        "address `i * 4` should fold to stride 1 word, got: {address}"
    );
    assert_eq!(
        value_coefficient, 1,
        "loop-carried `acc` (+1 each iter) should fold to coefficient 1 of i, got: {rhs}"
    );
    assert_eq!(
        value_coefficient, address_coefficient,
        "value and address must fold the same induction var consistently, else the written value lands wrong:\n  address {address}\n  value   {rhs}"
    );
    Ok(())
}


#[test]
fn manifest_numbers_must_match_the_kernel() -> TestResult {
    let kernel = lower_first_loop("integer/constant_base")?;
    let body_owner = wat::parse_str(
        r#"(module (memory 1 1) (func (param i32) (i32.store (i32.const 0) (local.get 0))))"#,
    )?;
    let mut module = load_module(&body_owner, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let FuncDecl::Body(_, _, body) = module
        .funcs
        .values()
        .find(|decl| matches!(decl, FuncDecl::Body(..)))
        .ok_or("no function body")?
    else {
        return Err("no function body".into());
    };
    let artifact = super::artifact(&kernel, body)?;

    for (label, expected) in [
        ("workgroup_size", kernel.workgroup_size as i64),
        ("min_offset_bytes", kernel.min_constant_bytes),
        ("max_constant_bytes", kernel.max_constant_bytes),
        ("max_stride_bytes", kernel.max_stride_bytes),
        ("result_slots", kernel.results.len() as i64),
    ] {
        assert!(
            artifact
                .manifest
                .contains(&format!("\"{label}\": {expected}")),
            "manifest {label} should be {expected} — wrong arg slots do not fail to compile, they silently write wrong\n{}",
            artifact.manifest
        );
    }
    Ok(())
}


#[test]
fn scale_running_pointer_gpu_matches_cpu_including_exit_value() -> TestResult {
    use super::conformance::gpu;

    let case = CASES
        .iter()
        .find(|case| case.name == "integer/scale_running_pointer")
        .ok_or("case integer/scale_running_pointer is not registered")?;
    let source_bytes = case.to_wasm()?;
    let a = 0_i32;
    let used = 1024_usize;
    let mut seed = vec![0_u8; used];
    for index in 0..8 {
        let word = (index + 1) as i32;
        seed[index * 4..(index + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }
    let engine = wasmtime::Engine::default();


    let reference_module = wasmtime::Module::new(&engine, &source_bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &reference_module, &[])?;
    let entry = instance.get_typed_func::<i32, ()>(&mut store, "main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    entry.call(&mut store, a)?;
    let reference = memory.data(&store)[..used].to_vec();
    let expected_pointer = i32::from_le_bytes(reference[512..516].try_into()?);
    assert_eq!(
        expected_pointer, 32,
        "CPU baseline final pointer must be a + 8*4 = 32, got {expected_pointer}"
    );


    let kernel = lower_bytes(&source_bytes)?;
    assert!(
        !kernel.results.is_empty(),
        "scale_running_pointer must emit non-empty results (induction live-out)"
    );
    assert!(
        kernel.source.contains("results[0] = "),
        "kernel must write the exit value; source was:\n{}",
        kernel.source
    );
    let injected = inject_dispatch_import(
        &source_bytes,
        kernel.fields.len().max(1),
        kernel.results.len(),
        &Trace::silent(),
    )?;
    let mut module = load_module(&injected, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let plan = plan_first_gpu_loop(&module, &Trace::silent())?;
    assert_eq!(
        plan.kernel.results.len(),
        1,
        "plan must see one exit value; got {}",
        plan.kernel.results.len()
    );
    offload_loop(
        &mut module,
        plan.func,
        plan.header,
        plan.exit,
        &plan.members,
        &plan.kernel.fields,
        0,
        None,
    )?;
    let rewritten = module.to_wasm_bytes()?;
    let kernel = plan.kernel;


    struct Host {
        context: gpu::GpuContext,
        compiled: gpu::Compiled,
        kernel: Kernel,
    }
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;

    let mut linker = wasmtime::Linker::new(&engine);
    linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |mut caller: wasmtime::Caller<'_, Host>,
         p0: i32,
         _kernel: i32|
         -> Result<i32, wasmtime::Error> {
            let memory = caller
                .get_export("mem")
                .and_then(|item| item.into_memory())
                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
            let data = memory.data(&caller).to_vec();
            let host = caller.data();
            let params = [p0 as u32];
            let workspace = host
                .context
                .workspace_with_results(data.len(), params.len() * 4, host.kernel.results.len())
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            let (produced, exit_values) = host
                .context
                .run_in_with_exit(&host.compiled, &host.kernel, &workspace, &data, &params)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            memory.write(&mut caller, 0, &produced)?;
            let exit = *exit_values
                .first()
                .ok_or_else(|| wasmtime::Error::msg("GPU returned no exit value"))?;
            Ok(exit as i32)
        },
    )?;

    let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;
    let mut gpu_store = wasmtime::Store::new(
        &engine,
        Host {
            context,
            compiled,
            kernel,
        },
    );
    let gpu_instance = linker.instantiate(&mut gpu_store, &rewritten_module)?;
    let gpu_entry = gpu_instance.get_typed_func::<i32, ()>(&mut gpu_store, "main")?;
    let gpu_memory = gpu_instance
        .get_memory(&mut gpu_store, "mem")
        .ok_or("missing memory")?;
    gpu_memory.write(&mut gpu_store, 0, &seed)?;
    gpu_entry.call(&mut gpu_store, a)?;
    let produced = gpu_memory.data(&gpu_store)[..used].to_vec();

    let mismatches = reference
        .iter()
        .zip(produced.iter())
        .filter(|(left, right)| left != right)
        .count();
    let gpu_pointer = i32::from_le_bytes(produced[512..516].try_into()?);
    println!(
        "\nscale_running_pointer: CPU vs GPU → {mismatches} mismatches; \
         final pointer CPU={expected_pointer} GPU={gpu_pointer}; \
         scaled words CPU={:?} GPU={:?}",
        (0..8)
            .map(|index| i32::from_le_bytes(
                reference[index * 4..index * 4 + 4].try_into().unwrap()
            ))
            .collect::<Vec<_>>(),
        (0..8)
            .map(|index| i32::from_le_bytes(produced[index * 4..index * 4 + 4].try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        mismatches, 0,
        "auto-rewrite + GPU offload with exit-value return mismatches pure CPU"
    );
    assert_eq!(gpu_pointer, 32, "GPU final pointer must be 32");


    let mut idle_linker = wasmtime::Linker::new(&engine);
    idle_linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |_p0: i32, _kernel: i32| -> i32 { 0 },
    )?;
    let idle_module = wasmtime::Module::new(&engine, &rewritten)?;
    let mut idle_store = wasmtime::Store::new(&engine, ());
    let idle_instance = idle_linker.instantiate(&mut idle_store, &idle_module)?;
    let idle_entry = idle_instance.get_typed_func::<i32, ()>(&mut idle_store, "main")?;
    let idle_memory = idle_instance
        .get_memory(&mut idle_store, "mem")
        .ok_or("missing memory")?;
    idle_memory.write(&mut idle_store, 0, &seed)?;
    idle_entry.call(&mut idle_store, a)?;
    let idle = idle_memory.data(&idle_store)[..used].to_vec();
    let differing = reference
        .iter()
        .zip(idle.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert!(
        differing > 0,
        "no-op offload still matches CPU — check has no discriminating power"
    );
    println!("negative control: {differing} bytes differ after no-op __hw_dispatch\n");
    Ok(())
}


#[test]
fn running_pointer_gpu_return_matches_cpu() -> TestResult {
    use super::conformance::gpu;

    let case = CASES
        .iter()
        .find(|case| case.name == "integer/running_pointer")
        .ok_or("case integer/running_pointer is not registered")?;
    let source_bytes = case.to_wasm()?;
    let (value, n) = (9_i32, 4_i32);
    let used = 64_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();

    let reference_module = wasmtime::Module::new(&engine, &source_bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &reference_module, &[])?;
    let entry = instance.get_typed_func::<(i32, i32), i32>(&mut store, "main")?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    let cpu_pointer = entry.call(&mut store, (value, n))?;
    let reference = memory.data(&store)[..used].to_vec();
    assert_eq!(cpu_pointer, n * 4, "CPU exit pointer must be n*4");

    let kernel = lower_bytes(&source_bytes)?;
    assert_eq!(
        kernel.results.len(),
        1,
        "running_pointer must emit the exit pointer as a result, got {}",
        kernel.results.len()
    );
    let injected = inject_dispatch_import(
        &source_bytes,
        kernel.fields.len().max(1),
        kernel.results.len(),
        &Trace::silent(),
    )?;
    let mut module = load_module(&injected, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let plan = plan_first_gpu_loop(&module, &Trace::silent())?;
    offload_loop(
        &mut module,
        plan.func,
        plan.header,
        plan.exit,
        &plan.members,
        &plan.kernel.fields,
        0,
        None,
    )?;
    let rewritten = module.to_wasm_bytes()?;
    let kernel = plan.kernel;

    struct Host {
        context: gpu::GpuContext,
        compiled: gpu::Compiled,
        kernel: Kernel,
    }
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let mut linker = wasmtime::Linker::new(&engine);
    linker.func_wrap(
        "heterowasm",
        "__hw_dispatch",
        |mut caller: wasmtime::Caller<'_, Host>,
         p0: i32,
         p1: i32,
         _kernel: i32|
         -> Result<i32, wasmtime::Error> {
            let memory = caller
                .get_export("mem")
                .and_then(|item| item.into_memory())
                .ok_or_else(|| wasmtime::Error::msg("missing memory export"))?;
            let data = memory.data(&caller).to_vec();
            let host = caller.data();
            let params = [p0 as u32, p1 as u32];
            let workspace = host
                .context
                .workspace_with_results(data.len(), params.len() * 4, host.kernel.results.len())
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            let (produced, exit_values) = host
                .context
                .run_in_with_exit(&host.compiled, &host.kernel, &workspace, &data, &params)
                .map_err(|error| wasmtime::Error::msg(error.to_string()))?;
            memory.write(&mut caller, 0, &produced)?;
            let exit = *exit_values
                .first()
                .ok_or_else(|| wasmtime::Error::msg("GPU returned no exit value"))?;
            Ok(exit as i32)
        },
    )?;
    let rewritten_module = wasmtime::Module::new(&engine, &rewritten)?;
    let mut gpu_store = wasmtime::Store::new(
        &engine,
        Host {
            context,
            compiled,
            kernel,
        },
    );
    let gpu_instance = linker.instantiate(&mut gpu_store, &rewritten_module)?;
    let gpu_entry = gpu_instance.get_typed_func::<(i32, i32), i32>(&mut gpu_store, "main")?;
    let gpu_memory = gpu_instance
        .get_memory(&mut gpu_store, "mem")
        .ok_or("missing memory")?;
    gpu_memory.write(&mut gpu_store, 0, &seed)?;
    let gpu_pointer = gpu_entry.call(&mut gpu_store, (value, n))?;
    let produced = gpu_memory.data(&gpu_store)[..used].to_vec();
    let mismatches = reference
        .iter()
        .zip(produced.iter())
        .filter(|(left, right)| left != right)
        .count();
    println!(
        "\nrunning_pointer: CPU pointer={cpu_pointer} GPU pointer={gpu_pointer} mismatches={mismatches}"
    );
    assert_eq!(mismatches, 0, "memory mismatches pure CPU");
    assert_eq!(gpu_pointer, cpu_pointer, "GPU return value mismatches CPU");
    Ok(())
}


#[test]
fn nonzero_index_start_matches_cpu_without_dividing_the_index() -> TestResult {
    use super::conformance::gpu;
    const INDEXED: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store
              (i32.mul (local.get $i) (i32.const 4))
              (i32.add (i32.load (i32.mul (local.get $i) (i32.const 4))) (i32.const 1)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))))"#;
    let bytes = wat::parse_str(INDEXED)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        !kernel.source.contains("/ 4u"),
        "index start must not be divided by 4; source was:\n{}",
        kernel.source
    );
    assert!(
        kernel.source.contains("gid.x"),
        "stride must use the zero-based trip; source was:\n{}",
        kernel.source
    );
    let start = 8_i32;
    let limit = 12_i32;
    let used = 64_usize;
    let mut seed = vec![0_u8; used];
    for index in 0..16 {
        seed[index * 4..index * 4 + 4].copy_from_slice(&(index as i32).to_le_bytes());
    }
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let gpu = context
        .run(&compiled, &kernel, &seed, &[start as u32, limit as u32])
        .map_err(|error| error.to_string())?;
    let mismatches = gpu[..used]
        .iter()
        .zip(cpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(mismatches, 0, "GPU memory mismatches CPU");
    Ok(())
}


#[test]
fn nonzero_start_with_separate_byte_base_matches_cpu() -> TestResult {
    use super::conformance::gpu;
    const SOURCE: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $base i32) (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store
              (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4)))
              (i32.const 9))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))))"#;
    let bytes = wat::parse_str(SOURCE)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        kernel.source.contains("params.p2 / 4u + i"),
        "byte base must add the absolute index; source was:\n{}",
        kernel.source
    );
    let base = 16_i32;
    let start = 2_i32;
    let limit = 6_i32;
    let used = 128_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (base, start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let gpu = context
        .run(
            &compiled,
            &kernel,
            &seed,
            &[start as u32, limit as u32, base as u32],
        )
        .map_err(|error| error.to_string())?;
    let mismatches = gpu[..used]
        .iter()
        .zip(cpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(
        mismatches, 0,
        "field start with a separate byte base mismatches CPU"
    );

    const CONSTANT_START: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $base i32)
        (local $i i32)
        (local.set $i (i32.const 3))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (i32.const 7)))
            (i32.store
              (i32.add (local.get $base) (i32.mul (local.get $i) (i32.const 4)))
              (i32.const 9))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))))"#;
    let bytes = wat::parse_str(CONSTANT_START)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        kernel.source.contains("params.p0 / 4u + i"),
        "constant start must keep the absolute index; source was:\n{}",
        kernel.source
    );
    let base = 16_i32;
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    let seed = vec![0_u8; used];
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<i32, ()>(&mut store, "main")?
        .call(&mut store, base)?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let gpu = context
        .run(&compiled, &kernel, &seed, &[base as u32])
        .map_err(|error| error.to_string())?;
    let mismatches = gpu[..used]
        .iter()
        .zip(cpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(
        mismatches, 0,
        "constant start with a separate byte base mismatches CPU"
    );
    Ok(())
}


#[test]
fn stored_index_with_nonzero_start_matches_cpu() -> TestResult {
    use super::conformance::gpu;
    const SOURCE: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store
              (i32.mul (local.get $i) (i32.const 4))
              (local.get $i))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))))"#;
    let bytes = wat::parse_str(SOURCE)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        !kernel.source.contains("params.p0 + i"),
        "stored index must not add the start twice; source was:\n{}",
        kernel.source
    );
    let start = 8_i32;
    let limit = 12_i32;
    let used = 64_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let gpu = context
        .run(&compiled, &kernel, &seed, &[start as u32, limit as u32])
        .map_err(|error| error.to_string())?;
    let mismatches = gpu[..used]
        .iter()
        .zip(cpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(mismatches, 0, "stored index mismatches CPU");
    Ok(())
}


#[test]
fn pointer_value_with_nonzero_index_start_matches_cpu() -> TestResult {
    use super::conformance::gpu;
    const SOURCE: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $base i32) (param $s i32) (param $n i32)
        (local $i i32)
        (local $p i32)
        (local.set $i (local.get $s))
        (local.set $p (local.get $base))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $p))
            (local.set $p (i32.add (local.get $p) (i32.const 4)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))
        (i32.store (i32.const 256) (local.get $p))))"#;
    let bytes = wat::parse_str(SOURCE)?;
    let kernel = lower_bytes(&bytes)?;
    let base = 20_i32;
    let start = 2_i32;
    let limit = 6_i32;
    let used = 512_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (base, start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let workspace = context
        .workspace_with_results(used, kernel.fields.len().max(1) * 4, kernel.results.len())
        .map_err(|error| error.to_string())?;
    let (gpu, exit) = context
        .run_in_with_exit(
            &compiled,
            &kernel,
            &workspace,
            &seed,
            &[start as u32, limit as u32, base as u32],
        )
        .map_err(|error| error.to_string())?;
    let cpu_final = i32::from_le_bytes(cpu[256..260].try_into().unwrap());
    assert_eq!(
        exit,
        vec![cpu_final as u32],
        "GPU exit pointer mismatches CPU"
    );
    let mut mismatches = 0_usize;
    for index in 0..16 {
        let cpu_word = i32::from_le_bytes(cpu[index * 4..index * 4 + 4].try_into().unwrap());
        let gpu_word = i32::from_le_bytes(gpu[index * 4..index * 4 + 4].try_into().unwrap());
        if cpu_word != gpu_word {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "pointer values stored at the index mismatch CPU"
    );
    Ok(())
}


#[test]
fn step_two_index_stops_at_the_limit() -> TestResult {
    use super::conformance::gpu;
    const STEPPED: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $i))
            (local.set $i (i32.add (local.get $i) (i32.const 2)))
            (br $loop)))))"#;
    let bytes = wat::parse_str(STEPPED)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        kernel
            .source
            .contains("params.p0 + 2u * gid.x >= params.p1"),
        "guard must advance by the induction step; source was:\n{}",
        kernel.source
    );
    let start = 4_i32;
    let limit = 12_i32;
    let used = 64_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let gpu = context
        .run(&compiled, &kernel, &seed, &[start as u32, limit as u32])
        .map_err(|error| error.to_string())?;
    let mismatches = gpu[..used]
        .iter()
        .zip(cpu.iter())
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(mismatches, 0, "step-2 index mismatches CPU");
    Ok(())
}


#[test]
fn step_two_exit_value_matches_cpu() -> TestResult {
    use super::conformance::gpu;
    const SOURCE: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $i))
            (local.set $i (i32.add (local.get $i) (i32.const 2)))
            (br $loop)))
        (i32.store (i32.const 256) (local.get $i))))"#;
    let bytes = wat::parse_str(SOURCE)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        kernel
            .source
            .contains("params.p0 + 2u * gid.x + 2u >= params.p1"),
        "last iteration must advance by the induction step; source was:\n{}",
        kernel.source
    );
    let start = 4_i32;
    let limit = 12_i32;
    let used = 512_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let cpu_final = i32::from_le_bytes(cpu[256..260].try_into().unwrap());
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let workspace = context
        .workspace_with_results(used, kernel.fields.len().max(1) * 4, kernel.results.len())
        .map_err(|error| error.to_string())?;
    let (gpu, exit) = context
        .run_in_with_exit(
            &compiled,
            &kernel,
            &workspace,
            &seed,
            &[start as u32, limit as u32],
        )
        .map_err(|error| error.to_string())?;
    let mut mismatches = 0_usize;
    for index in 0..16 {
        let cpu_word = i32::from_le_bytes(cpu[index * 4..index * 4 + 4].try_into().unwrap());
        let gpu_word = i32::from_le_bytes(gpu[index * 4..index * 4 + 4].try_into().unwrap());
        if cpu_word != gpu_word {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "step-2 memory mismatches CPU");
    assert_eq!(
        exit,
        vec![cpu_final as u32],
        "step-2 exit value mismatches CPU"
    );
    Ok(())
}


#[test]
fn step_two_exit_matches_cpu_when_limit_is_not_a_multiple() -> TestResult {
    use super::conformance::gpu;
    const SOURCE: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $i))
            (local.set $i (i32.add (local.get $i) (i32.const 2)))
            (br $loop)))
        (i32.store (i32.const 256) (local.get $i))))"#;
    let bytes = wat::parse_str(SOURCE)?;
    let kernel = lower_bytes(&bytes)?;
    let start = 4_i32;
    let limit = 13_i32;
    let used = 512_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
    let memory = instance
        .get_memory(&mut store, "mem")
        .ok_or("missing memory")?;
    memory.write(&mut store, 0, &seed)?;
    instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
        .call(&mut store, (start, limit))?;
    let cpu = memory.data(&store)[..used].to_vec();
    let cpu_final = i32::from_le_bytes(cpu[256..260].try_into().unwrap());
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let workspace = context
        .workspace_with_results(used, kernel.fields.len().max(1) * 4, kernel.results.len())
        .map_err(|error| error.to_string())?;
    let (gpu, exit) = context
        .run_in_with_exit(
            &compiled,
            &kernel,
            &workspace,
            &seed,
            &[start as u32, limit as u32],
        )
        .map_err(|error| error.to_string())?;
    let mut mismatches = 0_usize;
    for index in 0..16 {
        let cpu_word = i32::from_le_bytes(cpu[index * 4..index * 4 + 4].try_into().unwrap());
        let gpu_word = i32::from_le_bytes(gpu[index * 4..index * 4 + 4].try_into().unwrap());
        if cpu_word != gpu_word {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "unaligned step-2 memory mismatches CPU");
    assert_eq!(
        exit,
        vec![cpu_final as u32],
        "unaligned step-2 exit value mismatches CPU"
    );
    Ok(())
}


#[test]
fn downward_step_matches_cpu() -> TestResult {
    use super::conformance::gpu;
    const FIELD: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main") (param $s i32) (param $n i32)
        (local $i i32)
        (local.set $i (local.get $s))
        (block $exit
          (loop $loop
            (br_if $exit (i32.le_s (local.get $i) (local.get $n)))
            (local.set $i (i32.sub (local.get $i) (i32.const 2)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $i))
            (br $loop)))
        (i32.store (i32.const 256) (local.get $i))))"#;
    let bytes = wat::parse_str(FIELD)?;
    let kernel = lower_bytes(&bytes)?;
    assert!(
        kernel
            .source
            .contains("params.p0 < 2u * gid.x || params.p0 - 2u * gid.x <= params.p1"),
        "downward guard must subtract the step; source was:\n{}",
        kernel.source
    );
    assert!(
        kernel
            .source
            .contains("params.p0 - 2u * gid.x - 2u <= params.p1"),
        "downward exit must use the value after the subtract; source was:\n{}",
        kernel.source
    );
    let start = 12_i32;
    let used = 512_usize;
    let seed = vec![0_u8; used];
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes)?;
    let context = gpu::GpuContext::new().map_err(|error| error.to_string())?;
    let compiled = context
        .compile(&kernel)
        .map_err(|error| error.to_string())?;
    let workspace = context
        .workspace_with_results(used, kernel.fields.len().max(1) * 4, kernel.results.len())
        .map_err(|error| error.to_string())?;

    for limit in [4_i32, 5] {
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
        let memory = instance
            .get_memory(&mut store, "mem")
            .ok_or("missing memory")?;
        memory.write(&mut store, 0, &seed)?;
        instance
            .get_typed_func::<(i32, i32), ()>(&mut store, "main")?
            .call(&mut store, (start, limit))?;
        let cpu = memory.data(&store)[..used].to_vec();
        let cpu_final = i32::from_le_bytes(cpu[256..260].try_into().unwrap());
        let (gpu, exit) = context
            .run_in_with_exit(
                &compiled,
                &kernel,
                &workspace,
                &seed,
                &[start as u32, limit as u32],
            )
            .map_err(|error| error.to_string())?;
        let mismatches = gpu[..64]
            .iter()
            .zip(cpu[..64].iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(
            mismatches, 0,
            "downward step mismatches CPU at limit {limit}"
        );
        assert_eq!(
            exit,
            vec![cpu_final as u32],
            "downward exit value mismatches CPU at limit {limit}"
        );
    }
    Ok(())
}


#[test]
fn downward_subtract_records_negative_step() -> TestResult {
    use heterowasm_cfg::ControlFlow;
    use heterowasm_frontend::{expand_all_bodies, load_module};
    use heterowasm_scev::ScalarEvolution;
    use heterowasm_trace::Trace;
    use waffle::{FuncDecl, Operator, ValueDef};

    const CONSTANT: &str = r#"(module
      (memory (export "mem") 1 1)
      (func (export "main")
        (local $i i32)
        (local.set $i (i32.const 12))
        (block $exit
          (loop $loop
            (br_if $exit (i32.le_s (local.get $i) (i32.const 4)))
            (local.set $i (i32.sub (local.get $i) (i32.const 2)))
            (i32.store (i32.mul (local.get $i) (i32.const 4)) (local.get $i))
            (br $loop)))))"#;
    let bytes = wat::parse_str(CONSTANT)?;
    let mut module = load_module(&bytes, &Trace::silent())?;
    expand_all_bodies(&mut module, &Trace::silent())?;
    let FuncDecl::Body(_, _, body) = module
        .funcs
        .values()
        .find(|decl| matches!(decl, FuncDecl::Body(..)))
        .ok_or("missing function body")?
    else {
        return Err("missing function body".into());
    };
    let trace = Trace::silent();
    let flow = ControlFlow::analyze(body, &trace);
    let natural_loop = flow
        .natural_loops(body, &trace)
        .into_iter()
        .next()
        .ok_or("missing natural loop")?;
    let evolution = ScalarEvolution::analyze(body, &natural_loop, &trace);
    let variables = evolution.induction_variables();
    assert_eq!(variables.len(), 1, "subtract update must be one induction");
    assert_eq!(variables[0].step, -2, "subtract of 2 must be step -2");

    let header = body
        .blocks
        .get(natural_loop.header)
        .ok_or("missing header")?;
    let mut saw_subtract = false;
    for &pred in &header.preds {
        let Some(block) = body.blocks.get(pred) else {
            continue;
        };
        let waffle::Terminator::Br { target } = &block.terminator else {
            continue;
        };
        if !natural_loop.blocks.contains(&pred) {
            continue;
        }
        for arg in &target.args {
            let Some(ValueDef::Operator(Operator::I32Sub, _, _)) = body.values.get(*arg) else {
                continue;
            };
            saw_subtract = true;
        }
    }
    assert!(saw_subtract, "back-edge update must be I32Sub");
    Ok(())
}
