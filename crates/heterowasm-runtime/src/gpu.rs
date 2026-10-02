use std::borrow::Cow;

use crate::error::RuntimeError;
use crate::types::{DispatchSpec, Shader};
use heterowasm_trace::{Stage, Trace};


pub struct Execution {
    pub memory: Vec<u8>,
}


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

    pub fn new() -> Result<Self, RuntimeError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .map_err(|error| RuntimeError::GpuUnavailable {
            reason: error.to_string(),
        })?;

        let adapter_name = format!("{:?}", adapter.get_info().backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("heterowasm-runtime"),
            ..Default::default()
        }))
        .map_err(|error| RuntimeError::GpuUnavailable {
            reason: error.to_string(),
        })?;

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


    pub fn compile(&self, shader: &Shader<'_>) -> Result<Compiled, RuntimeError> {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernel"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader.source)),
            });
        let pipeline_layout = self
            .device
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
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Compiled { pipeline })
    }


    pub fn workspace(&self, size: usize, uniform_size: usize, result_slots: usize) -> Workspace {
        let size = size.max(4);
        let uniform_size = uniform_size.max(16);

        let result_slots = result_slots.max(1);
        let result_bytes = (result_slots * 4) as u64;
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
        Workspace {
            storage,
            uniform,
            staging,
            results,
            results_staging,
            bind_group,
            size,
            uniform_size,
            result_slots,
        }
    }


    pub(crate) fn copy_ranges(
        shader: &Shader<'_>,
        params: &[u32],
        limit: usize,
    ) -> Vec<(usize, usize, usize, usize)> {
        let size = u64::try_from(limit).unwrap_or(0);
        let mut raw =
            crate::dispatch::Span::pieces(shader, params, size).unwrap_or_else(|| vec![(0, size)]);

        if let Some((start, _)) = shader.dispatch.loaded_word(params) {
            let end = start.saturating_add(4);
            if end <= limit {
                let start = u64::try_from(start).unwrap_or(0);
                let end = u64::try_from(end).unwrap_or(start);
                raw.push((start, end));
            }
        }
        const COPY_ALIGN: usize = 8;
        let mut copies: Vec<(usize, usize, usize, usize)> = raw
            .iter()
            .filter_map(|(lo, hi)| {
                let lo = usize::try_from(*lo).ok()?;
                let hi = usize::try_from(*hi).ok()?;
                if hi <= lo {
                    return None;
                }
                let copy_lo = lo & !(COPY_ALIGN - 1);
                let mut copy_hi =
                    (hi.saturating_add(COPY_ALIGN - 1) & !(COPY_ALIGN - 1)).min(limit);
                if copy_hi < copy_lo {
                    copy_hi = copy_lo;
                }
                let mut copy_bytes = copy_hi.saturating_sub(copy_lo);
                if copy_lo + copy_bytes > limit {
                    copy_bytes = limit.saturating_sub(copy_lo) & !(COPY_ALIGN - 1);
                    copy_hi = copy_lo + copy_bytes;
                }
                if copy_bytes == 0 {
                    return None;
                }
                Some((lo, hi, copy_lo, copy_hi))
            })
            .collect();
        copies.sort_by_key(|piece| piece.2);
        let mut merged: Vec<(usize, usize, usize, usize)> = Vec::new();
        for piece in copies {
            if let Some(last) = merged.last_mut() {
                if piece.2 <= last.3 {
                    last.0 = last.0.min(piece.0);
                    last.1 = last.1.max(piece.1);
                    last.3 = last.3.max(piece.3);
                    continue;
                }
            }
            merged.push(piece);
        }
        merged
    }


    fn part_bytes<'a>(
        parts: &'a [(usize, Vec<u8>)],
        copy_lo: usize,
        copy_hi: usize,
    ) -> Result<&'a [u8], RuntimeError> {
        parts
            .iter()
            .find_map(|(lo, bytes)| {
                (*lo == copy_lo && lo.saturating_add(bytes.len()) == copy_hi)
                    .then_some(bytes.as_slice())
            })
            .ok_or_else(|| RuntimeError::DispatchRefused {
                reason: "copy range was not read from linear memory".to_string(),
            })
    }


    fn word_at(parts: &[(usize, Vec<u8>)], start: usize) -> Option<u32> {
        let end = start.checked_add(4)?;
        for (lo, bytes) in parts {
            let hi = lo.saturating_add(bytes.len());
            if *lo <= start && end <= hi {
                let offset = start - *lo;
                let mut buf = [0_u8; 4];
                buf.copy_from_slice(bytes.get(offset..offset + 4)?);
                return Some(u32::from_le_bytes(buf));
            }
        }
        None
    }


    pub fn run_in(
        &self,
        compiled: &Compiled,
        shader: &Shader<'_>,
        workspace: &Workspace,
        memory: &mut [u8],
        params: &[u32],
        trace: &Trace,
    ) -> Result<Vec<u32>, RuntimeError> {
        let limit = memory.len().min(workspace.size);
        let merged = Self::copy_ranges(shader, params, limit);
        let mut parts = Vec::with_capacity(merged.len());
        for (_, _, copy_lo, copy_hi) in &merged {
            let Some(bytes) = memory.get(*copy_lo..*copy_hi) else {
                return Err(RuntimeError::DispatchRefused {
                    reason: "copy range is outside linear memory".to_string(),
                });
            };
            parts.push((*copy_lo, bytes.to_vec()));
        }
        let (regions, exits) =
            self.run_parts(compiled, shader, workspace, limit, &parts, params, trace)?;
        for (lo, bytes) in regions {
            let Some(end) = lo.checked_add(bytes.len()) else {
                continue;
            };
            if let Some(slot) = memory.get_mut(lo..end) {
                slot.copy_from_slice(&bytes);
            }
        }
        Ok(exits)
    }


    pub(crate) fn run_parts(
        &self,
        compiled: &Compiled,
        shader: &Shader<'_>,
        workspace: &Workspace,
        limit: usize,
        parts: &[(usize, Vec<u8>)],
        params: &[u32],
        trace: &Trace,
    ) -> Result<(Vec<(usize, Vec<u8>)>, Vec<u32>), RuntimeError> {
        let merged = Self::copy_ranges(shader, params, limit);
        for (_, _, copy_lo, copy_hi) in &merged {
            let copy_lo_bytes = u64::try_from(*copy_lo).unwrap_or(0);
            let bytes = Self::part_bytes(parts, *copy_lo, *copy_hi)?;
            self.queue
                .write_buffer(&workspace.storage, copy_lo_bytes, bytes);
        }

        let mut uniform_bytes: Vec<u8> = Vec::with_capacity(workspace.uniform_size);
        for param in params {
            uniform_bytes.extend_from_slice(&param.to_le_bytes());
        }
        while uniform_bytes.len() < workspace.uniform_size {
            uniform_bytes.push(0);
        }

        if uniform_bytes.len() > workspace.uniform_size {
            uniform_bytes.truncate(workspace.uniform_size);
        }
        self.queue
            .write_buffer(&workspace.uniform, 0, &uniform_bytes);

        let dispatched = match shader.dispatch {
            DispatchSpec::Fixed(limit) => limit,
            DispatchSpec::FromField(field) => {
                *params
                    .get(field)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!(
                            "dispatch taken from field p{field}, but only {} arguments received",
                            params.len()
                        ),
                    })?
            }
            DispatchSpec::FromFieldMask(field, mask) => {
                params
                    .get(field)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!(
                            "dispatch taken from field p{field}, but only {} arguments received",
                            params.len()
                        ),
                    })?
                    & mask
            }
            DispatchSpec::FromFieldMaskAdd(field, mask, addend) => params
                .get(field)
                .ok_or_else(|| RuntimeError::DispatchRefused {
                    reason: format!(
                        "dispatch taken from field p{field}, but only {} arguments received",
                        params.len()
                    ),
                })
                .map(|value| (value & mask).wrapping_add(addend))?,
            DispatchSpec::FromSum(left, right) => {
                let left = *params
                    .get(left)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!("dispatch sum field p{left} is missing"),
                    })?;
                let right = *params
                    .get(right)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!("dispatch sum field p{right} is missing"),
                    })?;
                left.wrapping_add(right)
            }
            DispatchSpec::FromSumShift(base, shifted, amount) => {
                let base = *params
                    .get(base)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!("dispatch sum field p{base} is missing"),
                    })?;
                let shifted =
                    *params
                        .get(shifted)
                        .ok_or_else(|| RuntimeError::DispatchRefused {
                            reason: format!("dispatch sum field p{shifted} is missing"),
                        })?;
                base.wrapping_add(shifted.wrapping_shl(amount))
            }
            DispatchSpec::FromSubShift(base, minuend, subtrahend, amount) => {
                let base = *params
                    .get(base)
                    .ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: format!("dispatch sum field p{base} is missing"),
                    })?;
                let minuend =
                    *params
                        .get(minuend)
                        .ok_or_else(|| RuntimeError::DispatchRefused {
                            reason: format!("dispatch sum field p{minuend} is missing"),
                        })?;
                let subtrahend =
                    *params
                        .get(subtrahend)
                        .ok_or_else(|| RuntimeError::DispatchRefused {
                            reason: format!("dispatch sum field p{subtrahend} is missing"),
                        })?;
                base.wrapping_add(minuend.wrapping_sub(subtrahend).wrapping_shl(amount))
            }
            DispatchSpec::FromLoadedShiftMask(..) | DispatchSpec::FromLoaded(..) => {
                let (start, mask) = shader.dispatch.loaded_word(params).ok_or_else(|| {
                    RuntimeError::DispatchRefused {
                        reason: "loaded dispatch bound is out of range".to_string(),
                    }
                })?;
                let word =
                    Self::word_at(parts, start).ok_or_else(|| RuntimeError::DispatchRefused {
                        reason: "loaded dispatch bound is out of range".to_string(),
                    })?;
                word & mask
            }
        };

        let dispatched = crate::dispatch::scheduled_invocations(shader, params, dispatched);
        let workgroups = dispatched.div_ceil(shader.workgroup_size.max(1));

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
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        for (_, _, copy_lo, copy_hi) in &merged {
            let copy_lo_bytes = u64::try_from(*copy_lo).unwrap_or(0);
            let copy_bytes = u64::try_from(copy_hi.saturating_sub(*copy_lo)).unwrap_or(0);
            if copy_bytes == 0 {
                continue;
            }
            encoder.copy_buffer_to_buffer(
                &workspace.storage,
                copy_lo_bytes,
                &workspace.staging,
                copy_lo_bytes,
                copy_bytes,
            );
        }

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


        let exit_values: Vec<u32> = {
            let mapped =
                result_slice
                    .get_mapped_range()
                    .map_err(|error| RuntimeError::GpuUnavailable {
                        reason: error.to_string(),
                    })?;
            let available = mapped.len() / 4;

            let values = mapped
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| u32::from_le_bytes(*bytes))
                .collect::<Vec<u32>>();
            trace
                .debug(Stage::Runtime, "read back loop exit values")
                .field("available", available as i64)
                .emit();
            drop(mapped);
            values
        };
        workspace.results_staging.unmap();


        let mut regions = Vec::new();
        for (lo, hi, copy_lo, copy_hi) in &merged {
            if hi <= lo {
                continue;
            }
            let copy_lo_bytes = u64::try_from(*copy_lo).unwrap_or(0);
            let copy_hi_bytes = u64::try_from(*copy_hi).unwrap_or(0);
            let mapped = slice
                .slice(copy_lo_bytes..copy_hi_bytes)
                .get_mapped_range()
                .map_err(|error| RuntimeError::GpuUnavailable {
                    reason: error.to_string(),
                })?;
            let start = lo.saturating_sub(*copy_lo);
            let end = hi.saturating_sub(*copy_lo);
            if end <= mapped.len() && *hi <= limit {
                regions.push((*lo, mapped[start..end].to_vec()));
            }
            drop(mapped);
        }
        workspace.staging.unmap();


        trace
            .debug(Stage::Runtime, "dispatch executed")
            .field("adapter", self.adapter.clone())
            .field("workgroups", i64::from(workgroups))
            .field("dispatched", i64::from(dispatched))
            .field("buffer_bytes", workspace.size)
            .field(
                "touched_bytes",
                i64::try_from(merged.iter().fold(0_usize, |sum, piece| {
                    sum.saturating_add(piece.1.saturating_sub(piece.0))
                }))
                .unwrap_or(-1),
            )
            .emit();

        Ok((regions, exit_values))
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

impl Workspace {

    pub fn size(&self) -> usize {
        self.size
    }
}


pub fn execute(
    shader: &Shader<'_>,
    memory: &[u8],
    params: &[u32],
    uniform_size: usize,
    trace: &Trace,
) -> Result<Execution, RuntimeError> {
    let context = GpuContext::new()?;
    let compiled = context.compile(shader)?;
    let workspace = context.workspace(memory.len(), uniform_size, 0);

    let mut owned = memory.to_vec();
    context.run_in(&compiled, shader, &workspace, &mut owned, params, trace)?;
    Ok(Execution { memory: owned })
}
