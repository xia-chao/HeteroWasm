use std::collections::HashMap;

use waffle::{
    Block, ConstVal, Func, FunctionBody, InterpContext, Module, Terminator, Value, ValueDef,
};

use crate::{Dispatch, Kernel};


pub struct Subject<'a, 'm> {
    pub kernel: &'a Kernel,
    pub body: &'a FunctionBody,
    pub module: &'a Module<'m>,
    pub func: Func,
    pub memory: waffle::Memory,
    pub seed: &'a [u8],
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    pub inputs: usize,
    pub bytes_compared: usize,
    pub mismatches: usize,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    Compared(Outcome),
    Skipped(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Interpreter,
    UntraceableField,
    UnparsableKernel,
    DispatchMismatch,
}

impl std::fmt::Display for Error {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Error::Interpreter => "wasm interpreter execution failed",
            Error::UntraceableField => "uniform field does not trace back to a function parameter",
            Error::UnparsableKernel => "emitted text cannot be parsed",
            Error::DispatchMismatch => "dispatch count does not match iteration count",
        };
        out.write_str(text)
    }
}

impl std::error::Error for Error {}


pub fn compare(subject: &Subject<'_, '_>, inputs: &[Vec<i32>]) -> Result<Comparison, Error> {
    let Subject {
        kernel,
        body,
        module,
        func,
        memory,
        seed,
    } = *subject;

    let Some(mapping) = field_mapping(body, &kernel.fields) else {
        return Ok(Comparison::Skipped(
            "uniform field does not trace back to a function parameter",
        ));
    };

    let plan = parse_kernel(&kernel.source).map_err(|_| Error::UnparsableKernel)?;

    let mut outcome = Outcome {
        inputs: 0,
        bytes_compared: 0,
        mismatches: 0,
    };

    for arguments in inputs {
        let mut context = InterpContext::new(module).map_err(|_| Error::Interpreter)?;

        for (offset, byte) in seed.iter().enumerate() {
            if offset < context.memories[memory].data.len() {
                context.memories[memory].data[offset] = *byte;
            }
        }
        let initial = context.memories[memory].data.clone();

        let values: Vec<ConstVal> = arguments
            .iter()
            .map(|argument| ConstVal::I32(*argument as u32))
            .collect();
        context
            .call(module, func, &values)
            .ok()
            .map_err(|_| Error::Interpreter)?;
        let expected = context.memories[memory].data.clone();

        let params: Vec<u32> = mapping
            .iter()
            .map(|index| arguments.get(*index as usize).copied().unwrap_or(0) as u32)
            .collect();

        let actual = evaluate(&plan, &initial, &params);

        outcome.inputs += 1;
        outcome.bytes_compared += expected.len();
        if expected != actual {
            outcome.mismatches += 1;
        }
    }

    Ok(Comparison::Compared(outcome))
}


pub fn field_mapping(body: &FunctionBody, fields: &[Value]) -> Option<Vec<u32>> {

    let mut memo: HashMap<Value, Option<u32>> = HashMap::new();
    fields
        .iter()
        .map(|field| parameter_of(body, *field, 0, &mut memo))
        .collect()
}


fn parameter_of(
    body: &FunctionBody,
    value: Value,
    depth: usize,
    memo: &mut HashMap<Value, Option<u32>>,
) -> Option<u32> {
    if depth > 32 {
        return None;
    }
    let canonical = heterowasm_scev::canonical_value(body, value);
    if let Some(cached) = memo.get(&canonical) {
        return *cached;
    }

    memo.insert(canonical, None);

    let traced = match body.values.get(canonical) {
        Some(ValueDef::BlockParam(block, index, _)) => match body.blocks.get(*block) {
            Some(definition) if definition.preds.is_empty() => Some(*index),
            Some(definition) => {
                let mut traced = None;
                for predecessor in &definition.preds {
                    let Some(argument) =
                        argument_passed(body, *predecessor, *block, *index as usize)
                    else {
                        continue;
                    };
                    if let Some(found) = parameter_of(body, argument, depth + 1, memo) {
                        traced = Some(found);
                        break;
                    }
                }
                traced
            }
            None => None,
        },
        _ => None,
    };

    memo.insert(canonical, traced);
    traced
}

fn argument_passed(body: &FunctionBody, from: Block, to: Block, position: usize) -> Option<Value> {
    let definition = body.blocks.get(from)?;
    let target = match &definition.terminator {
        Terminator::Br { target } if target.block == to => target,
        Terminator::CondBr {
            if_true, if_false, ..
        } => {
            if if_true.block == to {
                if_true
            } else if if_false.block == to {
                if_false
            } else {
                return None;
            }
        }
        _ => return None,
    };
    target.args.get(position).copied()
}


struct Plan {
    guard: Option<Expr>,
    stores: Vec<(Expr, Expr)>,
}

#[derive(Debug, Clone)]
enum Expr {
    Index,
    Word(u32),
    Param(usize),
    Load(Box<Expr>),
    Binary(Box<Expr>, char, Box<Expr>),
}

fn parse_kernel(source: &str) -> Result<Plan, ()> {
    let mut guard = None;
    let mut stores = Vec::new();

    for line in source.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("if (i >= ") {
            let bound = rest.strip_suffix(") {").ok_or(())?;
            guard = Some(parse_expression(bound)?);
        }
        if let Some(rest) = line.strip_prefix("mem[") {
            let (index, tail) = split_bracket(rest)?;
            let value = tail
                .strip_prefix(" = ")
                .and_then(|t| t.strip_suffix(';'))
                .ok_or(())?;
            stores.push((parse_expression(index)?, parse_expression(value)?));
        }
    }

    Ok(Plan { guard, stores })
}

fn split_bracket(text: &str) -> Result<(&str, &str), ()> {
    let mut depth = 0_usize;
    for (position, character) in text.char_indices() {
        match character {
            '[' => depth += 1,
            ']' => {
                if depth == 0 {
                    return Ok((&text[..position], &text[position + 1..]));
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    Err(())
}


fn parse_expression(text: &str) -> Result<Expr, ()> {
    let text = text.trim();


    for operators in [['+', '-'], ['*', '/']] {
        if let Some(position) = find_operator(text, &operators) {
            let character = text[position..].chars().next().ok_or(())?;
            let left = parse_expression(&text[..position])?;
            let right = parse_expression(&text[position + character.len_utf8()..])?;
            return Ok(Expr::Binary(Box::new(left), character, Box::new(right)));
        }
    }

    if text == "i" {
        return Ok(Expr::Index);
    }
    if let Some(rest) = text.strip_prefix("mem[") {
        let (index, tail) = split_bracket(rest)?;
        if !tail.is_empty() {
            return Err(());
        }
        return Ok(Expr::Load(Box::new(parse_expression(index)?)));
    }
    if let Some(rest) = text.strip_prefix("params.p") {
        return Ok(Expr::Param(rest.parse().map_err(|_| ())?));
    }
    if let Some(digits) = text.strip_suffix('u') {
        return Ok(Expr::Word(digits.parse().map_err(|_| ())?));
    }
    Err(())
}


fn find_operator(text: &str, operators: &[char]) -> Option<usize> {
    let mut depth = 0_usize;
    let mut found = None;
    for (position, character) in text.char_indices() {
        match character {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            _ if depth == 0 && operators.contains(&character) && position > 0 => {
                found = Some(position);
            }
            _ => {}
        }
    }
    found
}

fn evaluate(plan: &Plan, wasm: &[u8], params: &[u32]) -> Vec<u8> {
    let mut memory = wasm.to_vec();
    let words = memory.len() / 4;

    for index in 0..words {
        let i = index as u32;
        if let Some(guard) = &plan.guard {
            if i >= eval(guard, i, &memory, params) {
                continue;
            }
        }
        for (target, value) in &plan.stores {
            let position = eval(target, i, &memory, params) as usize;
            let stored = eval(value, i, &memory, params);
            let offset = position * 4;
            if offset + 4 <= memory.len() {
                memory[offset..offset + 4].copy_from_slice(&stored.to_le_bytes());
            }
        }
    }

    memory
}

fn eval(expr: &Expr, i: u32, memory: &[u8], params: &[u32]) -> u32 {
    match expr {
        Expr::Index => i,
        Expr::Word(value) => *value,
        Expr::Param(position) => params.get(*position).copied().unwrap_or(0),
        Expr::Load(index) => {
            let offset = eval(index, i, memory, params) as usize * 4;
            if offset + 4 <= memory.len() {
                let mut bytes = [0_u8; 4];
                bytes.copy_from_slice(&memory[offset..offset + 4]);
                u32::from_le_bytes(bytes)
            } else {
                0
            }
        }
        Expr::Binary(left, operator, right) => {
            let left = eval(left, i, memory, params);
            let right = eval(right, i, memory, params);
            match operator {
                '+' => left.wrapping_add(right),
                '-' => left.wrapping_sub(right),
                '*' => left.wrapping_mul(right),

                _ => left.checked_div(right).unwrap_or(0),
            }
        }
    }
}


pub fn dispatch_fields(dispatch: Dispatch) -> Option<usize> {
    match dispatch {
        Dispatch::Fixed(_) => None,
        Dispatch::FromField(field) => Some(field),
        Dispatch::FromFieldMask(field, _) => Some(field),
        Dispatch::FromFieldMaskAdd(field, _, _) => Some(field),
        Dispatch::FromSum(left, _) => Some(left),
        Dispatch::FromSumShift(base, _, _) => Some(base),
        Dispatch::FromSubShift(base, _, _, _) => Some(base),
        Dispatch::FromLoadedShiftMask(base, _, _, _) => Some(base),
        Dispatch::FromLoaded(slot, _) => Some(slot),
    }
}

#[cfg(test)]
pub mod gpu {


    use std::borrow::Cow;

    use wgpu::util::DeviceExt;

    use crate::{Dispatch, Kernel};


    pub struct Execution {
        pub memory: Vec<u8>,
        pub adapter: String,
    }


    #[derive(Debug, Clone)]
    pub enum GpuError {
        NoAdapter(String),
        NoDevice(String),
        DispatchUnknown,
        Compile(String),
    }

    impl std::fmt::Display for GpuError {
        fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                GpuError::NoAdapter(reason) => write!(out, "cannot obtain GPU adapter: {reason}"),
                GpuError::NoDevice(reason) => write!(out, "cannot obtain GPU device: {reason}"),
                GpuError::DispatchUnknown => out.write_str(
                    "dispatch count is not a compile-time constant and field is out of range",
                ),
                GpuError::Compile(reason) => write!(out, "shader compile failed: {reason}"),
            }
        }
    }

    impl std::error::Error for GpuError {}


    pub struct GpuContext {
        device: wgpu::Device,
        queue: wgpu::Queue,
        layout: wgpu::BindGroupLayout,
        adapter: String,
    }


    pub struct Compiled {
        pipeline: wgpu::ComputePipeline,
    }

    impl GpuContext {
        pub fn new() -> Result<Self, GpuError> {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    ..Default::default()
                }))
                .map_err(|error| GpuError::NoAdapter(error.to_string()))?;

            let adapter_name = format!("{:?}", adapter.get_info().backend);
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("heterowasm-conformance"),
                    ..Default::default()
                }))
                .map_err(|error| GpuError::NoDevice(error.to_string()))?;

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },

                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            Ok(GpuContext {
                device,
                queue,
                layout,
                adapter: adapter_name,
            })
        }


        pub fn adapter(&self) -> &str {
            &self.adapter
        }


        pub fn compile(&self, kernel: &Kernel) -> Result<Compiled, GpuError> {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("kernel"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&kernel.source)),
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&self.layout)],
                        immediate_size: 0,
                    });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            Ok(Compiled { pipeline })
        }


        pub fn run(
            &self,
            compiled: &Compiled,
            kernel: &Kernel,
            initial: &[u8],
            params: &[u32],
        ) -> Result<Vec<u8>, GpuError> {
            let storage = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mem"),
                    contents: initial,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                });

            let mut uniform_bytes: Vec<u8> = params
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            while uniform_bytes.len() < 16 {
                uniform_bytes.push(0);
            }
            let uniform = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: &uniform_bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });


            let result_slots = kernel.results.len().max(1);
            let result_bytes = (result_slots * 4) as u64;
            let results = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("results"),
                size: result_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: results.as_entire_binding(),
                    },
                ],
            });

            let dispatched = match kernel.dispatch {
                Dispatch::Fixed(limit) => limit,
                Dispatch::FromField(field) => params
                    .get(field)
                    .copied()
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromFieldMask(field, mask) => params
                    .get(field)
                    .copied()
                    .map(|value| value & mask)
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromFieldMaskAdd(field, mask, addend) => params
                    .get(field)
                    .copied()
                    .map(|value| (value & mask).wrapping_add(addend))
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromSum(left, right) => {
                    let left = params.get(left).copied().ok_or(GpuError::DispatchUnknown)?;
                    let right = params
                        .get(right)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    left.wrapping_add(right)
                }
                Dispatch::FromSumShift(base, shifted, amount) => {
                    let base = params.get(base).copied().ok_or(GpuError::DispatchUnknown)?;
                    let shifted = params
                        .get(shifted)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    base.wrapping_add(shifted.wrapping_shl(amount))
                }
                Dispatch::FromSubShift(base, minuend, subtrahend, amount) => {
                    let base = params.get(base).copied().ok_or(GpuError::DispatchUnknown)?;
                    let minuend = params
                        .get(minuend)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    let subtrahend = params
                        .get(subtrahend)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    base.wrapping_add(minuend.wrapping_sub(subtrahend).wrapping_shl(amount))
                }
                Dispatch::FromLoadedShiftMask(..) | Dispatch::FromLoaded(..) => kernel
                    .dispatch
                    .loaded_count(params, initial)
                    .ok_or(GpuError::DispatchUnknown)?,
            };
            let workgroups = dispatched.div_ceil(64);

            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: initial.len() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&compiled.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, initial.len() as u64);
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
                .ok();
            let _ = receiver.recv();

            let mapped = slice
                .get_mapped_range()
                .map_err(|error| GpuError::NoDevice(error.to_string()))?;
            let memory = mapped.to_vec();
            drop(mapped);
            staging.unmap();

            Ok(memory)
        }
    }


    pub struct Workspace {
        storage: wgpu::Buffer,
        uniform: wgpu::Buffer,
        staging: wgpu::Buffer,
        results: wgpu::Buffer,
        results_staging: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        size: usize,
        uniform_size: usize,
        result_slots: usize,
    }

    impl GpuContext {

        pub fn workspace(&self, size: usize, uniform_size: usize) -> Result<Workspace, GpuError> {
            self.workspace_with_results(size, uniform_size, 1)
        }


        pub fn workspace_with_results(
            &self,
            size: usize,
            uniform_size: usize,
            result_slots: usize,
        ) -> Result<Workspace, GpuError> {
            let size = size.max(4);
            let uniform_size = uniform_size.max(16);
            let result_slots = result_slots.max(1);
            let result_bytes = (result_slots * 4) as u64;
            let storage = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mem"),
                size: size as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"),
                size: uniform_size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: size as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let results = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("results"),
                size: result_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let results_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("results readback"),
                size: result_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: storage.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: results.as_entire_binding(),
                    },
                ],
            });
            Ok(Workspace {
                storage,
                uniform,
                staging,
                results,
                results_staging,
                bind_group,
                size,
                uniform_size,
                result_slots,
            })
        }


        pub fn run_in(
            &self,
            compiled: &Compiled,
            kernel: &Kernel,
            workspace: &Workspace,
            initial: &[u8],
            params: &[u32],
        ) -> Result<Vec<u8>, GpuError> {
            let (memory, _) =
                self.run_in_with_exit(compiled, kernel, workspace, initial, params)?;
            Ok(memory)
        }


        pub fn run_in_with_exit(
            &self,
            compiled: &Compiled,
            kernel: &Kernel,
            workspace: &Workspace,
            initial: &[u8],
            params: &[u32],
        ) -> Result<(Vec<u8>, Vec<u32>), GpuError> {
            let payload = initial.len().min(workspace.size);
            self.queue
                .write_buffer(&workspace.storage, 0, &initial[..payload]);

            let mut uniform_bytes: Vec<u8> = Vec::with_capacity(workspace.uniform_size);
            for param in params {
                uniform_bytes.extend_from_slice(&param.to_le_bytes());
            }
            while uniform_bytes.len() < workspace.uniform_size {
                uniform_bytes.push(0);
            }
            self.queue
                .write_buffer(&workspace.uniform, 0, &uniform_bytes);

            let dispatched = match kernel.dispatch {
                Dispatch::Fixed(limit) => limit,
                Dispatch::FromField(field) => params
                    .get(field)
                    .copied()
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromFieldMask(field, mask) => params
                    .get(field)
                    .copied()
                    .map(|value| value & mask)
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromFieldMaskAdd(field, mask, addend) => params
                    .get(field)
                    .copied()
                    .map(|value| (value & mask).wrapping_add(addend))
                    .ok_or(GpuError::DispatchUnknown)?,
                Dispatch::FromSum(left, right) => {
                    let left = params.get(left).copied().ok_or(GpuError::DispatchUnknown)?;
                    let right = params
                        .get(right)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    left.wrapping_add(right)
                }
                Dispatch::FromSumShift(base, shifted, amount) => {
                    let base = params.get(base).copied().ok_or(GpuError::DispatchUnknown)?;
                    let shifted = params
                        .get(shifted)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    base.wrapping_add(shifted.wrapping_shl(amount))
                }
                Dispatch::FromSubShift(base, minuend, subtrahend, amount) => {
                    let base = params.get(base).copied().ok_or(GpuError::DispatchUnknown)?;
                    let minuend = params
                        .get(minuend)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    let subtrahend = params
                        .get(subtrahend)
                        .copied()
                        .ok_or(GpuError::DispatchUnknown)?;
                    base.wrapping_add(minuend.wrapping_sub(subtrahend).wrapping_shl(amount))
                }
                Dispatch::FromLoadedShiftMask(..) | Dispatch::FromLoaded(..) => kernel
                    .dispatch
                    .loaded_count(params, initial)
                    .ok_or(GpuError::DispatchUnknown)?,
            };

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&compiled.pipeline);
                pass.set_bind_group(0, &workspace.bind_group, &[]);
                pass.dispatch_workgroups(dispatched.div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(
                &workspace.storage,
                0,
                &workspace.staging,
                0,
                workspace.size as u64,
            );
            let result_bytes = (workspace.result_slots * 4) as u64;
            encoder.copy_buffer_to_buffer(
                &workspace.results,
                0,
                &workspace.results_staging,
                0,
                result_bytes,
            );
            self.queue.submit(Some(encoder.finish()));

            let slice = workspace.staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            let result_slice = workspace.results_staging.slice(..);
            let (result_sender, result_receiver) = std::sync::mpsc::channel();
            result_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = result_sender.send(result);
            });
            self.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok();
            let _ = receiver.recv();
            let _ = result_receiver.recv();

            let mapped = slice
                .get_mapped_range()
                .map_err(|error| GpuError::NoDevice(error.to_string()))?;
            let memory = mapped.to_vec();
            drop(mapped);
            workspace.staging.unmap();

            let result_mapped = result_slice
                .get_mapped_range()
                .map_err(|error| GpuError::NoDevice(error.to_string()))?;
            let wanted = kernel.results.len().min(result_mapped.len() / 4);
            let exit_values: Vec<u32> = result_mapped
                .as_chunks::<4>()
                .0
                .iter()
                .take(wanted)
                .map(|bytes| u32::from_le_bytes(*bytes))
                .collect();
            drop(result_mapped);
            workspace.results_staging.unmap();

            Ok((memory, exit_values))
        }
    }


    pub fn execute(kernel: &Kernel, initial: &[u8], params: &[u32]) -> Result<Execution, GpuError> {
        let context = GpuContext::new()?;
        let compiled = context.compile(kernel)?;
        let memory = context.run(&compiled, kernel, initial, params)?;
        Ok(Execution {
            memory,
            adapter: context.adapter().to_string(),
        })
    }
}
