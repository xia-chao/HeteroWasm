use std::path::PathBuf;

use heterowasm_trace::{Level, Trace};

use crate::{DispatchSpec, Runtime, RuntimeError};


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
    };
    let error = crate::dispatch::check(&shader, &[0, 8], 4096)
        .expect_err("partially overlapping ranges must be rejected");
    let text = error.to_string();
    assert!(
        text.contains("overlap"),
        "reject reason must mention overlap: {text}"
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
    };


    let error = crate::dispatch::check(&shader, &[0], 65536)
        .expect_err("out-of-bounds access with stride 8 must be rejected");
    assert!(
        error.to_string().contains("65540"),
        "reason must include the computed byte count: {error}"
    );


    let unit_stride = crate::types::Shader {
        max_stride_bytes: 4,
        ..shader
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
    };
    let params = [83240_u32, 83268, 83280, 0, 0];
    crate::dispatch::check(&shader, &params, 16_777_216)
        .expect("destination before source must not be refused");
    let pieces = crate::dispatch::Span::pieces(&shader, &params, 16_777_216).expect("pieces");
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
    assert_eq!(super::host::import_arity(&[spec.clone()]), 1);
    assert_eq!(super::host::import_results(&[spec]), 1);
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
