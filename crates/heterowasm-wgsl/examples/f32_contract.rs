use heterowasm_address as _;
use heterowasm_bounds as _;
use heterowasm_cfg as _;
use heterowasm_corpus as _;
use heterowasm_frontend as _;
use heterowasm_legality as _;
use heterowasm_scev as _;
use heterowasm_trace as _;
use heterowasm_wgsl as _;
use naga as _;
use waffle as _;
use wasm_encoder as _;
use wasmparser as _;
use wasmprinter as _;
use wasmtime as _;
use wat as _;

use std::borrow::Cow;


const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const SLOTS: u32 = 8u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&a)) {
        return;
    }
    let x = a[i];
    let y = b[i];
    let z = c[i];
    out[i * SLOTS + 0u] = x + y;
    out[i * SLOTS + 1u] = x - y;
    out[i * SLOTS + 2u] = x * y;
    out[i * SLOTS + 3u] = x / y;
    out[i * SLOTS + 4u] = x * y + z;
    out[i * SLOTS + 5u] = fma(x, y, z);
    out[i * SLOTS + 6u] = sqrt(abs(x));
    out[i * SLOTS + 7u] = max(x, y);
}
"#;

const SLOTS: usize = 8;


const OP_NAMES: [&str; SLOTS] = [
    "a + b",
    "a - b",
    "a * b",
    "a / b",
    "a * b + c",
    "fma(a,b,c)",
    "sqrt(abs(a))",
    "max(a,b)",
];

const CATEGORIES: [&str; 6] = ["uniform01", "bits", "finite", "neartie", "denormal", "ml"];


const N: usize = 1 << 20;

fn main() -> Result<(), String> {
    let gpu = Gpu::new()?;
    println!("adapter       {}", gpu.adapter);
    println!("elements/cat  {N}");
    println!();

    smoke(&gpu)?;
    println!("positive control  PASS (harness reads real data)");
    println!();

    let rng_seed = 0x2545_F491_4F6C_DD1D_u64;
    for category in CATEGORIES {
        let (a, b, c) = generate(category, N, rng_seed);
        verify_inputs(category, &a, &b, &c)?;
        let (gpu_bits, cpu_bits) = evaluate(&gpu, &a, &b, &c)?;
        report(category, &a, &b, &c, &gpu_bits, &cpu_bits);
        probe_contraction(category, &a, &b, &c, &gpu_bits);
        probe_flush(category, a.len(), &cpu_bits, &gpu_bits);
    }

    Ok(())
}


fn verify_inputs(category: &str, a: &[f32], b: &[f32], c: &[f32]) -> Result<(), String> {
    let all = || a.iter().chain(b.iter()).chain(c.iter());

    let violation = match category {
        "uniform01" => all().find(|value| !(0.0..1.0).contains(*value)),
        "finite" => all().find(|value| !value.is_finite() || value.is_subnormal()),
        "denormal" => all().find(|value| !(value.is_subnormal() || **value == 0.0)),
        "neartie" => a.iter().find(|value| {
            let bits = value.to_bits() & 0x007f_ffff;
            bits != 0 || **value == 0.0
        }),
        _ => None,
    };

    match violation {
        Some(value) => Err(format!(
            "category `{category}` violated its own invariant: {value:e} (bits {:#010x})",
            value.to_bits()
        )),
        None => Ok(()),
    }
}


fn probe_contraction(category: &str, a: &[f32], b: &[f32], c: &[f32], gpu: &[u32]) {
    let mut informative = 0usize;
    let mut follows_nonfused = 0usize;
    let mut follows_fused = 0usize;
    let mut follows_neither = 0usize;

    for i in 0..a.len() {
        let nonfused = cpu_op(4, a[i], b[i], c[i]).to_bits();
        let fused = cpu_op(5, a[i], b[i], c[i]).to_bits();
        if nonfused == fused {
            continue;
        }
        informative += 1;
        let got = gpu[i * SLOTS + 4];
        if got == nonfused {
            follows_nonfused += 1;
        } else if got == fused {
            follows_fused += 1;
        } else {
            follows_neither += 1;
        }
    }

    let verdict = if informative == 0 {
        "no informative samples"
    } else if follows_fused == informative {
        "CONTRACTS (gpu fuses a*b+c)"
    } else if follows_nonfused == informative {
        "no contraction"
    } else {
        "MIXED"
    };
    println!(
        "[contraction] {category}: informative={informative} nonfused={follows_nonfused} \
         fused={follows_fused} neither={follows_neither} -> {verdict}"
    );
}


fn probe_flush(category: &str, n: usize, cpu: &[u32], gpu: &[u32]) {
    let mut flushed_to_zero = 0usize;
    let mut became_nan = 0usize;
    let mut cpu_subnormal_total = 0usize;

    for i in 0..n {
        for op in 0..SLOTS {
            let r = cpu[i * SLOTS + op];
            let g = gpu[i * SLOTS + op];
            let cpu_value = f32::from_bits(r);
            let gpu_value = f32::from_bits(g);
            if cpu_value.is_subnormal() {
                cpu_subnormal_total += 1;
                if gpu_value == 0.0 {
                    flushed_to_zero += 1;
                }
            }
            if cpu_value.is_finite() && gpu_value.is_nan() {
                became_nan += 1;
            }
        }
    }

    if cpu_subnormal_total > 0 || became_nan > 0 {
        println!(
            "[flush]       {category}: cpu_subnormal={cpu_subnormal_total} \
             gpu_turned_into_zero={flushed_to_zero} cpu_finite_but_gpu_nan={became_nan}"
        );
    }
}


struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }


    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}


fn finite_bits(rng: &mut Rng) -> u32 {
    let bits = rng.next_u32();
    let exponent = (bits >> 23) & 0xff;
    if exponent == 0 || exponent == 255 {
        (bits & 0x807f_ffff) | (127 << 23)
    } else {
        bits
    }
}

fn generate(category: &str, n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng(seed | 1);
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);

    for _ in 0..n {
        let (x, y, z) = match category {
            "uniform01" => (rng.unit(), rng.unit(), rng.unit()),
            "bits" => (
                f32::from_bits(rng.next_u32()),
                f32::from_bits(rng.next_u32()),
                f32::from_bits(rng.next_u32()),
            ),
            "finite" => (
                f32::from_bits(finite_bits(&mut rng)),
                f32::from_bits(finite_bits(&mut rng)),
                f32::from_bits(finite_bits(&mut rng)),
            ),

            "neartie" => {
                let exponent = (rng.next_u32() % 60) as i32 - 30;
                let scale = 2f32.powi(exponent);
                let k = (rng.next_u32() % 8) as f32;
                let m = rng.next_u32() % (1 << 23);
                (
                    scale,
                    k * scale * 2f32.powi(-24),
                    f32::from_bits(0x3f80_0000 | m) * scale,
                )
            }
            "denormal" => (
                f32::from_bits(rng.next_u32() & 0x007f_ffff),
                f32::from_bits(rng.next_u32() & 0x007f_ffff),
                f32::from_bits(rng.next_u32() & 0x007f_ffff),
            ),
            _ => {
                let normal = |rng: &mut Rng| {
                    let sum: f32 = (0..12).map(|_| rng.unit()).sum();
                    sum - 6.0
                };
                (normal(&mut rng), normal(&mut rng), normal(&mut rng))
            }
        };
        a.push(x);
        b.push(y);
        c.push(z);
    }

    (a, b, c)
}


fn cpu_op(op: usize, x: f32, y: f32, z: f32) -> f32 {
    match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        3 => x / y,
        4 => {
            let product = std::hint::black_box(x * y);
            product + z
        }
        5 => x.mul_add(y, z),
        6 => x.abs().sqrt(),
        _ => x.max(y),
    }
}


#[derive(Clone, Copy)]
struct Example {
    a: u32,
    b: u32,
    c: u32,
    cpu: u32,
    gpu: u32,
}

#[derive(Default)]
struct Stat {
    exact: usize,
    nan_agree: usize,
    mismatch: usize,
    examples: Vec<Example>,
}

impl Stat {

    fn is_bit_exact(&self) -> bool {
        self.mismatch == 0 && self.nan_agree == 0
    }
}

fn report(category: &str, a: &[f32], b: &[f32], c: &[f32], gpu: &[u32], cpu: &[u32]) {
    let n = a.len();
    let mut stats: [Stat; SLOTS] = std::array::from_fn(|_| Stat::default());

    for op in 0..SLOTS {
        let stat = &mut stats[op];
        for i in 0..n {
            let g = gpu[i * SLOTS + op];
            let r = cpu[i * SLOTS + op];
            if g == r {
                stat.exact += 1;
            } else if f32::from_bits(g).is_nan() && f32::from_bits(r).is_nan() {

                stat.nan_agree += 1;
            } else {
                stat.mismatch += 1;
                if stat.examples.len() < 3 {
                    stat.examples.push(Example {
                        a: a[i].to_bits(),
                        b: b[i].to_bits(),
                        c: c[i].to_bits(),
                        cpu: r,
                        gpu: g,
                    });
                }
            }
        }
    }

    println!("── {category} (n = {n}) ──");
    println!(
        "{:<14} {:>10} {:>10} {:>10} {:>10}   verdict",
        "op", "exact", "nan-only", "MISMATCH", "distinct"
    );
    for (op, stat) in stats.iter().enumerate() {

        let distinct: std::collections::HashSet<u32> =
            (0..n).map(|i| cpu[i * SLOTS + op]).collect();
        let verdict = if stat.is_bit_exact() {
            "bit-exact"
        } else if stat.mismatch == 0 {
            "exact except NaN payload"
        } else {
            "DIFFERS"
        };
        println!(
            "{:<14} {:>10} {:>10} {:>10} {:>10}   {}",
            OP_NAMES[op],
            stat.exact,
            stat.nan_agree,
            stat.mismatch,
            distinct.len(),
            verdict
        );
    }
    for (op, stat) in stats.iter().enumerate() {
        for example in &stat.examples {
            println!(
                "   [{}] a={:#010x} b={:#010x} c={:#010x} | cpu={:#010x} gpu={:#010x}",
                OP_NAMES[op], example.a, example.b, example.c, example.cpu, example.gpu
            );
        }
    }
    println!();
}


struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    adapter: String,
}

impl Gpu {
    fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|error| format!("no adapter: {error}"))?;

        let info = adapter.get_info();

        let adapter_name = if info.driver.is_empty() {
            format!("{:?} / {}", info.backend, info.name)
        } else {
            format!("{:?} / {} / {}", info.backend, info.name, info.driver)
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("f32-contract"),
            ..Default::default()
        }))
        .map_err(|error| format!("no device: {error}"))?;

        let storage = |read_only: bool| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = Vec::with_capacity(4);
        for binding in 0..4u32 {
            let mut entry = storage(binding != 3);
            entry.binding = binding;
            entries.push(entry);
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f32-contract"),
            entries: &entries,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("f32-contract"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Gpu {
            device,
            queue,
            pipeline,
            layout,
            adapter: adapter_name,
        })
    }


    fn run(&self, a: &[f32], b: &[f32], c: &[f32]) -> Result<Vec<u32>, String> {
        use wgpu::util::DeviceExt;

        let n = a.len();
        let bytes =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };

        let input_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buffer_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("a"),
                contents: &bytes(a),
                usage: input_usage,
            });
        let buffer_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("b"),
                contents: &bytes(b),
                usage: input_usage,
            });
        let buffer_c = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c"),
                contents: &bytes(c),
                usage: input_usage,
            });

        let out_size = (n * SLOTS * 4) as u64;
        let buffer_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_out.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n as u32).div_ceil(64);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buffer_out, 0, &staging, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|error| format!("poll failed: {error}"))?;
        receiver
            .recv()
            .map_err(|error| format!("map callback lost: {error}"))?
            .map_err(|error| format!("map failed: {error}"))?;

        let bits: Vec<u32> = {
            let mapped = slice
                .get_mapped_range()
                .map_err(|error| format!("mapped range failed: {error}"))?;
            mapped
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect()
        };
        staging.unmap();

        Ok(bits)
    }
}


fn smoke(gpu: &Gpu) -> Result<(), String> {
    let a = [1.0f32, 2.0, 3.0, 4.0];
    let b = [5.0f32, 6.0, 7.0, 8.0];
    let c = [0.0f32, 0.0, 0.0, 0.0];
    let bits = gpu.run(&a, &b, &c)?;

    let expected = [6.0f32, 8.0, 10.0, 12.0];
    let mismatch = expected.iter().enumerate().find_map(|(i, want)| {
        let got = f32::from_bits(bits[i * SLOTS]);
        (got != *want).then_some((i, *want, got))
    });
    match mismatch {
        Some((i, want, got)) => Err(format!(
            "positive control FAILED at i={i}: expected {want}, harness returned {got} \
             -- the harness is not reading real data, so the measurement is meaningless"
        )),
        None => Ok(()),
    }
}


fn evaluate(gpu: &Gpu, a: &[f32], b: &[f32], c: &[f32]) -> Result<(Vec<u32>, Vec<u32>), String> {
    let gpu_bits = gpu.run(a, b, c)?;
    let mut cpu_bits = Vec::with_capacity(a.len() * SLOTS);
    for i in 0..a.len() {
        for op in 0..SLOTS {
            cpu_bits.push(cpu_op(op, a[i], b[i], c[i]).to_bits());
        }
    }
    Ok((gpu_bits, cpu_bits))
}
