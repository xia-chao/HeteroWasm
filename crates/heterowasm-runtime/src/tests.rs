use std::path::PathBuf;

use heterowasm_trace::{Level, Trace};
use wasmtime::Val;

use crate::{DispatchSpec, Runtime, RuntimeError, Shader};


fn fixture(tag: &str, kernel_wgsl: &str, kernel_json: &str, wat_source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("heterowasm-runtime-{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    std::fs::write(dir.join("kernels.json"), KERNELS_INDEX).expect("write index");
    std::fs::write(dir.join("kernel.wgsl"), kernel_wgsl).expect("write wgsl");
    std::fs::write(dir.join("kernel.json"), kernel_json).expect("write manifest");
    let wasm = wat::parse_str(wat_source).expect("assemble fixture wasm");
    std::fs::write(dir.join("rewritten.wasm"), wasm).expect("write wasm");
    dir
}


const KERNELS_INDEX: &str = r#"{ "schema": "heterowasm.kernels.v1", "kernels": ["kernel"] }"#;


const KERNEL_MANIFEST: &str = r#"{
  "schema": "heterowasm.kernel.v1",
  "wgsl": "kernel.wgsl",
  "workgroup_size": 64,
  "dispatch": { "kind": "fixed", "count": 8 },
  "uniform_size": 16,
  "min_offset_bytes": 0,
  "max_constant_bytes": 0,
  "max_stride_bytes": 4,
  "storage_binding": 0,
  "params_binding": 1,
  "fields_resolved": true,
  "fields": [
    { "slot": 0, "wasm_param": 0, "name": "params.p0" }
  ]
}"#;


const KERNEL_WGSL: &str = r#"struct Params {
  p0 : u32,
};
@group(0) @binding(0) var<storage, read_write> mem : array<u32>;
@group(0) @binding(1) var<uniform> params : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i : u32 = gid.x;
  if (i >= 8u) { return; }
  mem[params.p0 / 4u + i] = 42u;
}
"#;


const REWRITTEN_WAT: &str = r#"(module
  (import "heterowasm" "__hw_dispatch" (func $dispatch (param i32 i32)))
  (memory (export "mem") 1)
  (func (export "main")
    i32.const 0
    i32.const 0
    call $dispatch)
)"#;

fn quiet() -> Trace {

    Trace::to_stderr(Level::Error)
}

#[test]
fn runs_rewritten_wasm_without_any_glue() -> Result<(), Box<dyn std::error::Error>> {
    let dir = fixture("basic", KERNEL_WGSL, KERNEL_MANIFEST, REWRITTEN_WAT);
    let mut runtime = Runtime::load(&dir, quiet())?;
    runtime.run("main", &[])?;

    let memory = runtime.memory()?;

    for word in 0..8 {
        let at = word * 4;
        let value =
            u32::from_le_bytes([memory[at], memory[at + 1], memory[at + 2], memory[at + 3]]);
        assert_eq!(
            value, 42,
            "word {word} should be 42 (kernel really wrote memory)"
        );
    }

    let at = 8 * 4;
    let value = u32::from_le_bytes([memory[at], memory[at + 1], memory[at + 2], memory[at + 3]]);
    assert_eq!(
        value, 0,
        "word 8 is outside the dispatch range and must not be written"
    );

    assert_eq!(runtime.dispatches(), 1);
    Ok(())
}

#[test]
fn second_run_reuses_everything_and_dispatches_again() -> Result<(), Box<dyn std::error::Error>> {
    let dir = fixture("reuse", KERNEL_WGSL, KERNEL_MANIFEST, REWRITTEN_WAT);
    let mut runtime = Runtime::load(&dir, quiet())?;
    runtime.run("main", &[])?;
    runtime.run("main", &[])?;

    assert_eq!(runtime.dispatches(), 2);
    Ok(())
}

#[test]
fn missing_index_is_reported_not_silently_skipped() {
    let dir = std::env::temp_dir().join("heterowasm-runtime-empty");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create directory");

    let error = match Runtime::load(&dir, quiet()) {
        Ok(_) => panic!("missing kernels.json must error"),
        Err(error) => error,
    };
    assert!(
        matches!(error, RuntimeError::Read { .. }),
        "expected a read failure, got {error}"
    );
}

#[test]
fn manifest_parsing_reads_every_field() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from("kernel.json");
    let spec = crate::manifest::parse_kernel(KERNEL_MANIFEST, &path, "kernel", String::new())?;
    assert_eq!(spec.workgroup_size, 64);
    assert_eq!(spec.uniform_size, 16);
    assert_eq!(spec.dispatch, DispatchSpec::Fixed(8));
    assert_eq!(spec.fields.len(), 1);
    assert_eq!(spec.fields[0].wasm_param, 0);
    Ok(())
}

#[test]
fn malformed_manifest_reports_what_is_missing() {
    let path = PathBuf::from("bad.json");
    let broken = r#"{ "schema": "heterowasm.kernel.v1" }"#;
    let error = crate::manifest::parse_kernel(broken, &path, "bad", String::new())
        .expect_err("missing workgroup_size must error");
    let text = error.to_string();
    assert!(
        text.contains("workgroup_size"),
        "error must say what is missing: {text}"
    );
}


#[test]
fn overlapping_ranges_must_be_refused() {

    let shader = crate::types::Shader {
        source: KERNEL_WGSL,
        workgroup_size: 64,
        dispatch: DispatchSpec::Fixed(8),

        min_offset_bytes: 0,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: None,
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let error = crate::dispatch::check(&shader, &[0, 8], 4096)
        .expect_err("partially overlapping ranges must be rejected");
    let text = error.to_string();
    assert!(
        text.contains("overlap"),
        "reject reason must mention overlap: {text}"
    );


    let all_upload = crate::dispatch::Span::pieces(
        &shader,
        &[0, 8],
        4096,
        crate::dispatch::AccessDirection::Upload,
    )
    .expect("pieces");
    let all_readback = crate::dispatch::Span::pieces(
        &shader,
        &[0, 8],
        4096,
        crate::dispatch::AccessDirection::Readback,
    )
    .expect("pieces");
    assert_eq!(
        all_upload, all_readback,
        "unclassified shader must keep both directions identical"
    );
    let classified = crate::types::Shader {
        read_slots: vec![0],
        write_slots: vec![1],
        ..shader.clone()
    };
    let upload = crate::dispatch::Span::pieces(
        &classified,
        &[0, 8],
        4096,
        crate::dispatch::AccessDirection::Upload,
    )
    .expect("pieces");
    let readback = crate::dispatch::Span::pieces(
        &classified,
        &[0, 8],
        4096,
        crate::dispatch::AccessDirection::Readback,
    )
    .expect("pieces");
    assert_eq!(
        upload,
        vec![(0, 36)],
        "write-only slot must not be uploaded"
    );
    assert_eq!(
        readback,
        vec![(8, 44)],
        "read-only slot must not be read back"
    );


    crate::dispatch::check(&shader, &[0, 64], 4096).expect("disjoint ranges should be allowed");


    crate::dispatch::check(&shader, &[0, 0], 4096)
        .expect("in-place element-wise should be allowed");
}


#[test]
fn negative_offset_needs_lower_bound_slack() {
    let shader = crate::types::Shader {
        source: KERNEL_WGSL,
        workgroup_size: 64,
        dispatch: DispatchSpec::Fixed(8),

        min_offset_bytes: -8,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: None,
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };


    let error = crate::dispatch::check(&shader, &[0, 64], 4096)
        .expect_err("base 0 has no headroom; must reject");
    assert!(
        error.to_string().contains("lower-bound"),
        "reason must explain it is a lower-bound issue: {error}"
    );


    crate::dispatch::check(&shader, &[8, 64], 4096).expect("exact headroom should be allowed");


    crate::dispatch::check(&shader, &[64, 512], 4096).expect("ample headroom should be allowed");


    let positive = crate::types::Shader {
        min_offset_bytes: 0,
        ..shader
    };
    crate::dispatch::check(&positive, &[0, 64], 4096)
        .expect("with no negative offset, base 0 should be allowed");
}


#[test]
fn strided_access_must_account_for_the_real_stride() {

    let shader = crate::types::Shader {
        source: KERNEL_WGSL,
        workgroup_size: 64,
        dispatch: DispatchSpec::Fixed(8192),
        min_offset_bytes: 0,
        max_constant_bytes: 0,

        max_stride_bytes: 8,
        index_field: None,
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };


    let error = crate::dispatch::check(&shader, &[0], 65536)
        .expect_err("out-of-bounds access with stride 8 must be rejected");
    assert!(
        error.to_string().contains("65540"),
        "reason must include the computed byte count: {error}"
    );


    let unit_stride = crate::types::Shader {
        max_stride_bytes: 4,
        source: shader.source,
        workgroup_size: shader.workgroup_size,
        dispatch: shader.dispatch,
        min_offset_bytes: shader.min_offset_bytes,
        max_constant_bytes: shader.max_constant_bytes,
        index_field: shader.index_field,
        launch_count: shader.launch_count,
        read_slots: shader.read_slots.clone(),
        write_slots: shader.write_slots.clone(),
    };
    crate::dispatch::check(&unit_stride, &[0], 40000)
        .expect("with unit stride, 40000 is enough — counterexample holds");
    crate::dispatch::check(&shader, &[0], 40000)
        .expect_err("at the same size, real stride 8 must be rejected");


    crate::dispatch::check(&shader, &[0], 200000).expect("sufficient buffer should be allowed");
}


#[test]
fn downward_index_start_extends_the_copy_range() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: -8,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let (_lo, hi) = crate::dispatch::touched_range(&shader, &[12, 4, 64], 65536).expect("range");
    assert!(hi >= 108, "copy must cover the highest store, hi={hi}");
}


#[test]
fn index_start_is_not_an_alias_base() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: -16,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    crate::dispatch::check(&shader, &[20, 12, 16], 65536)
        .expect("the start slot is an index, not a second pointer");
    let (_lo, hi) = crate::dispatch::touched_range(&shader, &[20, 12, 16], 65536).expect("range");
    assert!(hi >= 84, "copy must cover byte 80, hi={hi}");
}


#[test]
fn upward_byte_pointer_writeback_includes_destination() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: 0,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let params = [83240_u32, 83268, 83280, 0, 0];
    crate::dispatch::check(&shader, &params, 16_777_216)
        .expect("destination before source must not be refused");
    let pieces = crate::dispatch::Span::pieces(
        &shader,
        &params,
        16_777_216,
        crate::dispatch::AccessDirection::Readback,
    )
    .expect("pieces");
    assert!(
        pieces.iter().any(|(lo, hi)| *lo <= 83240 && *hi >= 83268),
        "destination bytes must be written back, pieces={pieces:?}"
    );
    assert!(
        pieces.iter().all(|(_, hi)| *hi <= 83316),
        "span must follow the byte limit, not the raw dispatch count, pieces={pieces:?}"
    );
}


#[test]
fn downward_index_range_includes_the_low_stores() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: -8,
        max_constant_bytes: 0,
        max_stride_bytes: 8,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let (lo, hi) = crate::dispatch::touched_range(&shader, &[40, 4], 65536).expect("range");
    assert!(
        lo <= 16,
        "low store at byte 16 must be inside the copy, lo={lo}"
    );
    assert!(
        hi >= 156,
        "high store at byte 152 must be inside the copy, hi={hi}"
    );
}


#[test]
fn downward_trip_count_exceeds_the_limit_field() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: -8,
        max_constant_bytes: 0,
        max_stride_bytes: 8,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    assert_eq!(crate::dispatch::invocation_count(&shader, &[200, 4], 4), 98);
    assert_eq!(crate::dispatch::invocation_count(&shader, &[133, 4], 4), 65);
    let upward = crate::dispatch::invocation_count(&shader, &[4, 12], 12);
    assert_eq!(upward, 12, "an increasing loop keeps the limit dispatch");
}


#[test]
fn upward_byte_guard_launches_the_byte_span() {
    let source = "if (params.p0 + 4u * gid.x >= params.p1) { return; }\n";
    let shader = crate::types::Shader {
        source,
        workgroup_size: 64,
        dispatch: DispatchSpec::FromField(1),
        min_offset_bytes: 0,
        max_constant_bytes: 0,
        max_stride_bytes: 4,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    assert_eq!(
        crate::dispatch::invocation_count(&shader, &[83240, 83268], 83268),
        7,
        "the byte guard must not launch the end address as a thread count"
    );
}


#[test]
fn constant_downward_launch_uses_the_trip_count() {
    let shader = crate::types::Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::Fixed(4),
        min_offset_bytes: 0,
        max_constant_bytes: 792,
        max_stride_bytes: 8,
        index_field: None,
        launch_count: Some(98),
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    assert_eq!(crate::dispatch::scheduled_invocations(&shader, &[], 4), 98);
    let mut kept = shader;
    kept.launch_count = None;
    assert_eq!(crate::dispatch::scheduled_invocations(&kept, &[], 4), 4);
}

#[test]
fn dispatch_beyond_kernel_count_is_refused() {

    let two_calls = r#"(module
      (import "heterowasm" "__hw_dispatch" (func $dispatch (param i32 i32)))
      (memory (export "mem") 1)
      (func (export "main")
        i32.const 0
        i32.const 0
        call $dispatch
        i32.const 0
        i32.const 1
        call $dispatch)
    )"#;
    let dir = fixture("mismatch", KERNEL_WGSL, KERNEL_MANIFEST, two_calls);
    let mut runtime = match Runtime::load(&dir, quiet()) {
        Ok(runtime) => runtime,
        Err(error) => panic!("load should succeed, got {error}"),
    };
    let error = match runtime.run("main", &[]) {
        Ok(_) => panic!("second call has no matching kernel; must error"),
        Err(error) => error,
    };
    assert!(
        matches!(error, RuntimeError::Execute { .. }),
        "expected a runtime error, got {error}"
    );
}


#[test]
fn resolved_empty_fields_are_zero_parameters() {
    const TEXT: &str = r#"{
      "workgroup_size": 64,
      "dispatch": { "kind": "fixed", "count": 4 },
      "uniform_size": 16,
      "min_offset_bytes": 0,
      "max_constant_bytes": 40,
      "max_stride_bytes": 8,
      "result_slots": 1,
      "fields_resolved": true,
      "fields": []
    }"#;
    let spec =
        super::manifest::parse_kernel(TEXT, std::path::Path::new("k.json"), "k", String::new())
            .expect("parse zero-field manifest");
    let specs = std::slice::from_ref(&spec);
    assert_eq!(super::host::import_arity(specs), 1);
    assert_eq!(super::host::import_results(specs), 1);
}


#[test]
fn unresolved_uniform_padding_uses_the_wasm_import() {
    const TEXT: &str = r#"{
      "workgroup_size": 64,
      "dispatch": { "kind": "fixed", "count": 4 },
      "uniform_size": 16,
      "min_offset_bytes": 0,
      "max_constant_bytes": 0,
      "max_stride_bytes": 4,
      "result_slots": 2,
      "fields_resolved": false,
      "fields": []
    }"#;
    let wat = r#"(module
      (import "heterowasm" "__hw_dispatch" (func $dispatch (param i32 i32 i32 i32) (result i32 i32)))
      (memory (export "mem") 1)
      (func (export "main")
        i32.const 0
        i32.const 0
        i32.const 0
        i32.const 0
        call $dispatch
        drop
        drop)
    )"#;
    let dir = fixture("padded", KERNEL_WGSL, TEXT, wat);
    let mut runtime = match Runtime::load(&dir, quiet()) {
        Ok(runtime) => runtime,
        Err(error) => panic!("load should follow the wasm import, got {error}"),
    };
    runtime
        .run("main", &[])
        .expect("the four-parameter import should instantiate and return");
}


#[test]
fn caller_imports_override_the_zero_stub() -> Result<(), Box<dyn std::error::Error>> {
    const WAT: &str = r#"(module
      (import "env" "answer" (func $answer (result i32)))
      (memory (export "mem") 1)
      (func (export "run") (result i32) (call $answer))
    )"#;
    let dir = fixture("caller-imports", KERNEL_WGSL, KERNEL_MANIFEST, WAT);

    let mut stubbed = Runtime::load(&dir, quiet())?;
    assert_eq!(
        i32_result(&stubbed.run("run", &[])?),
        Some(0),
        "the default path must stay a zero stub"
    );

    let mut hooked = Runtime::load_with(&dir, quiet(), |_engine, linker, _module| {
        linker
            .func_wrap("env", "answer", || 7i32)
            .map_err(|error| RuntimeError::Import {
                reason: error.to_string(),
            })?;
        Ok(())
    })?;
    assert_eq!(
        i32_result(&hooked.run("run", &[])?),
        Some(7),
        "the caller defined import must win over the stub"
    );
    Ok(())
}


fn i32_result(results: &[Val]) -> Option<i32> {
    match results.first() {
        Some(Val::I32(value)) => Some(*value),
        _ => None,
    }
}


#[test]
fn mask_add_limit_keeps_the_index_span() {
    let shader = Shader {
        source: "",
        workgroup_size: 64,
        dispatch: DispatchSpec::FromFieldMaskAdd(1, 0xFFFF_FFFC, 0xFFFF_FFC0),
        min_offset_bytes: 0,
        max_constant_bytes: 60,
        max_stride_bytes: 64,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let params = [100_000_u32, 100_200, 200_000];
    let (lo, hi) =
        crate::dispatch::touched_range(&shader, &params, 16 * 1024 * 1024).expect("touched range");
    assert!(
        lo <= 100_000,
        "the destination span must start at the index slot"
    );
    assert!(
        hi >= 100_192,
        "the destination span must reach the last 64-byte block"
    );
    assert!(hi >= 200_192, "the source span must be kept as well");
}


#[test]
fn missing_artifact_directory_reports_the_read_error() {
    let dir = std::env::temp_dir().join("heterowasm-runtime-missing-artifacts");
    let _ = std::fs::remove_dir_all(&dir);
    let Err(error) = Runtime::load(&dir, quiet()) else {
        panic!("a missing artifact directory must not load");
    };
    assert!(
        matches!(error, RuntimeError::Read { .. }),
        "expected a read error, got {error}"
    );
    assert!(
        error.to_string().contains("kernels.json"),
        "the error must name the missing file, got {error}"
    );
}


#[test]
fn caller_import_failure_is_reported_as_an_import_error() -> Result<(), Box<dyn std::error::Error>>
{
    const WAT: &str = r#"(module
      (import "env" "answer" (func $answer (result i32)))
      (memory (export "mem") 1)
      (func (export "run") (result i32) (call $answer))
    )"#;
    let dir = fixture("caller-imports-failure", KERNEL_WGSL, KERNEL_MANIFEST, WAT);
    let Err(error) = Runtime::load_with(&dir, quiet(), |_engine, _linker, _module| {
        Err(RuntimeError::Import {
            reason: String::from("boom"),
        })
    }) else {
        panic!("a failing caller import must not load");
    };
    assert!(matches!(error, RuntimeError::Import { .. }), "got {error}");
    assert!(error.to_string().contains("boom"), "got {error}");
    Ok(())
}


#[test]
fn mask_add_dispatch_is_trip_count_not_absolute_address() {

    fn shader(comparison: &str) -> Shader<'static> {
        let source = format!(
            "fn main() {{\n  if (params.p0 + 64u * gid.x {comparison} ((params.p1 & 4294967292u) + 4294967232u)) {{\n    return;\n  }}\n}}\n"
        );
        let source: &'static str = Box::leak(source.into_boxed_str());
        Shader {
            source,
            workgroup_size: 64,
            dispatch: DispatchSpec::FromFieldMaskAdd(1, 0xFFFF_FFFC, 0xFFFF_FFC0),
            min_offset_bytes: 0,
            max_constant_bytes: 60,
            max_stride_bytes: 64,
            index_field: Some(0),
            launch_count: None,
            read_slots: Vec::new(),
            write_slots: Vec::new(),
        }
    }

    let params = [100_000_u32, 100_192, 999_999];
    assert_eq!(
        crate::dispatch::invocation_count(&shader(">"), &params, 100_136),
        3,
        "inclusive exit needs the boundary iteration"
    );
    assert_eq!(
        crate::dispatch::invocation_count(&shader(">="), &params, 100_136),
        2,
        "strict exit stops before the bound"
    );
}


#[test]
fn sum_shift_limit_keeps_the_index_span_and_skips_value_slots() {
    let source = "fn main() {\n  if (params.p0 + 32u * gid.x >= (params.p1 + (params.p2 << 2u))) {\n    return;\n  }\n  mem[params.p0 / 4u + 8u * gid.x] = params.p3;\n}\n";
    let shader = Shader {
        source,
        workgroup_size: 64,
        dispatch: DispatchSpec::FromSumShift(1, 2, 2),
        min_offset_bytes: 0,
        max_constant_bytes: 28,
        max_stride_bytes: 32,
        index_field: Some(0),
        launch_count: None,
        read_slots: Vec::new(),
        write_slots: Vec::new(),
    };
    let params = [735_588_u32, 735_584, 208_353, 0xFFFF_FFFF];
    let (lo, hi) =
        crate::dispatch::touched_range(&shader, &params, 16 * 1024 * 1024).expect("touched range");
    assert!(
        lo <= 735_588,
        "the span must start at the index slot, got {lo}"
    );
    assert!(hi >= 1_569_000, "the span must reach the bound, got {hi}");
    assert!(
        hi < 16 * 1024 * 1024,
        "the value slot must not be treated as an address, got {hi}"
    );
    assert_eq!(
        crate::dispatch::invocation_count(&shader, &params, 1_568_996),
        26_044,
        "dispatch must be the trip count, not the bound expression"
    );
}
