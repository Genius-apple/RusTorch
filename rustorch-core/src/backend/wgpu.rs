use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use wgpu;
use wgpu::util::DeviceExt;

pub struct MemoryPool {
    pool: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl MemoryPool {
    fn new() -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_buffer(&self, device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        let mut pool = self.pool.lock().unwrap();
        if let Some(buffers) = pool.get_mut(&size) {
            if let Some(buf) = buffers.pop() {
                return buf;
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        })
    }

    pub fn return_buffer(&self, buffer: wgpu::Buffer, size: u64) {
        let mut pool = self.pool.lock().unwrap();
        pool.entry(size).or_insert_with(Vec::new).push(buffer);
    }

    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();
    }
}

static MEMORY_POOL: OnceLock<MemoryPool> = OnceLock::new();

pub fn get_memory_pool() -> &'static MemoryPool {
    MEMORY_POOL.get_or_init(MemoryPool::new)
}

pub fn clear_memory_pool() {
    get_memory_pool().clear();
}

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub matmul_bind_group_layout: wgpu::BindGroupLayout,
    pub elementwise_pipeline: wgpu::ComputePipeline,
    pub elementwise_bind_group_layout: wgpu::BindGroupLayout,
    pub elementwise_vec4_pipeline: wgpu::ComputePipeline,
    pub adam_pipeline: wgpu::ComputePipeline,
    pub adam_bind_group_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub reduce_general_pipeline: wgpu::ComputePipeline,
    pub reduce_all_pipeline: wgpu::ComputePipeline,
    pub reduce_bind_group_layout: wgpu::BindGroupLayout,
    pub contiguous_pipeline: wgpu::ComputePipeline,
    pub contiguous_bind_group_layout: wgpu::BindGroupLayout,
    pub current_encoder: Mutex<Option<wgpu::CommandEncoder>>,
}

static CONTEXT: OnceLock<WgpuContext> = OnceLock::new();

pub fn get_context() -> Option<&'static WgpuContext> {
    Some(CONTEXT.get_or_init(|| {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No suitable WGPU adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("RusTorch Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("Failed to create device");

        // --- MatMul Pipeline ---
        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
        });

        let matmul_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("MatMul Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MatMul Pipeline Layout"),
                    bind_group_layouts: &[&matmul_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &matmul_shader,
            entry_point: "main",
        });

        // --- Elementwise Pipeline ---
        let elem_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("elementwise.wgsl").into()),
        });

        let elem_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Elementwise Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let elementwise_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Elementwise Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Elementwise Pipeline Layout"),
                        bind_group_layouts: &[&elem_bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &elem_shader,
                entry_point: "main",
            });

        // --- Elementwise Vec4 Pipeline ---
        let elem_vec4_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elementwise Vec4 Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("elementwise_vec4.wgsl").into()),
        });

        let elementwise_vec4_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Elementwise Vec4 Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Elementwise Vec4 Pipeline Layout"),
                        bind_group_layouts: &[&elem_bind_group_layout], // Reusing same layout
                        push_constant_ranges: &[],
                    }),
                ),
                module: &elem_vec4_shader,
                entry_point: "main",
            });

        // --- Adam Pipeline ---
        let adam_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adam Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("adam.wgsl").into()),
        });

        let adam_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Adam Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Adam Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Adam Pipeline Layout"),
                    bind_group_layouts: &[&adam_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &adam_shader,
            entry_point: "main",
        });

        // --- Reduce Pipeline ---
        let reduce_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reduce Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("reduce.wgsl").into()),
        });

        let reduce_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Reduce Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Reduce Pipeline Layout"),
                    bind_group_layouts: &[&reduce_bind_group_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &reduce_shader,
            entry_point: "reduce_sum_dim0",
        });

        let reduce_general_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reduce General Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Reduce Pipeline Layout"),
                        bind_group_layouts: &[&reduce_bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &reduce_shader,
                entry_point: "reduce_sum_general",
            });

        let reduce_all_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reduce All Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Reduce Pipeline Layout"),
                        bind_group_layouts: &[&reduce_bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &reduce_shader,
                entry_point: "reduce_sum_all",
            });

        // --- Contiguous Pipeline ---
        let contiguous_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Contiguous Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("contiguous.wgsl").into()),
        });

        let contiguous_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Contiguous Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let contiguous_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Contiguous Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Contiguous Pipeline Layout"),
                        bind_group_layouts: &[&contiguous_bind_group_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &contiguous_shader,
                entry_point: "main",
            });

        WgpuContext {
            device,
            queue,
            matmul_pipeline,
            matmul_bind_group_layout,
            elementwise_pipeline,
            elementwise_bind_group_layout: elem_bind_group_layout,
            elementwise_vec4_pipeline,
            adam_pipeline,
            adam_bind_group_layout,
            reduce_pipeline,
            reduce_general_pipeline,
            reduce_all_pipeline,
            reduce_bind_group_layout,
            contiguous_pipeline,
            contiguous_bind_group_layout,
            current_encoder: Mutex::new(None),
        }
    }))
}

pub fn flush_queue() {
    let ctx = get_context().expect("WGPU not initialized");
    let mut lock = ctx.current_encoder.lock().unwrap();
    if let Some(encoder) = lock.take() {
        ctx.queue.submit(Some(encoder.finish()));
    }
}

pub fn record_commands<F>(f: F)
where
    F: FnOnce(&mut wgpu::CommandEncoder),
{
    let ctx = get_context().expect("WGPU not initialized");
    let mut lock = ctx.current_encoder.lock().unwrap();

    if lock.is_none() {
        *lock = Some(
            ctx.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Buffered Encoder"),
                }),
        );
    }

    let encoder = lock.as_mut().unwrap();
    f(encoder);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    ReLU = 4,
    Sigmoid = 5,
    Tanh = 6,
    ReLUBackward = 10,
    SigmoidBackward = 11,
    TanhBackward = 12,
    SGDStep = 20,
    ExpandRepeat = 30,
}

pub use crate::backend::Activation;

pub fn elementwise_wgpu_buffer(
    input1: &wgpu::Buffer,
    input1_shape: &[usize],
    input1_strides: &[usize],
    input2: Option<(&wgpu::Buffer, &[usize], &[usize])>,
    output_shape: &[usize],
    op: ElementwiseOp,
    alpha: Option<f32>,
) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;
    // queue used for buffer creation (internal)

    let output_size: usize = output_shape.iter().product();
    let output_byte_size = output_size * std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    fn is_contiguous_or_scalar(shape: &[usize], strides: &[usize]) -> bool {
        if shape.iter().product::<usize>() == 1 {
            return true;
        }
        let mut st = 1;
        for i in (0..shape.len()).rev() {
            if strides[i] != st {
                return false;
            }
            st *= shape[i];
        }
        return true;
    }

    let can_use_vec4 = output_size % 4 == 0
        && is_contiguous_or_scalar(input1_shape, input1_strides)
        && input2.map_or(true, |(_, s2, st2)| is_contiguous_or_scalar(s2, st2));

    if can_use_vec4 {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ParamsVec4 {
            numel_vec4: u32,
            op: u32,
            alpha: f32,
            stride_mode_1: u32,
            stride_mode_2: u32,
        }

        let stride_mode_1 = if input1_shape.iter().product::<usize>() == 1 {
            1
        } else {
            0
        };
        let stride_mode_2 = if let Some((_, s2, _)) = input2 {
            if s2.iter().product::<usize>() == 1 {
                1
            } else {
                0
            }
        } else {
            0
        };

        let params = ParamsVec4 {
            numel_vec4: (output_size / 4) as u32,
            op: op as u32,
            alpha: alpha.unwrap_or(0.0),
            stride_mode_1,
            stride_mode_2,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elementwise Vec4 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Input 2 (optional)
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let input2_buf = input2.map(|(b, _, _)| b).unwrap_or(&dummy);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Elementwise Vec4 Bind Group"),
            layout: &ctx.elementwise_bind_group_layout, // Compatible layout
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        record_commands(|encoder| {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Elementwise Vec4 Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.elementwise_vec4_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (params.numel_vec4 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        });
    } else {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            numel: u32,
            op: u32,
            alpha: f32,
            ndim: u32,
            shape: [u32; 4],
            strides_out: [u32; 4],
            strides_1: [u32; 4],
            strides_2: [u32; 4],
        }

        let mut shape_arr = [1u32; 4];
        let stride_out_arr = [0u32; 4]; // Unused in shader logic but present in struct
        let mut stride_1_arr = [0u32; 4];
        let mut stride_2_arr = [0u32; 4];

        // Populate arrays (broadcasting logic)
        // Note: Shader iterates over output_shape (target shape)
        // We need to map target_index -> target_coord -> input_coord -> input_index
        // Using strides directly allows: index -> dot(coord, stride)
        // But shader does: index -> coord -> dot(coord, stride)

        let ndim = output_shape.len();
        for i in 0..ndim {
            shape_arr[4 - ndim + i] = output_shape[i] as u32;
        }

        fn fill_strides(
            shape: &[usize],
            strides: &[usize],
            target_shape: &[usize],
            out_arr: &mut [u32; 4],
        ) {
            let ndim = target_shape.len();
            let s_ndim = shape.len();
            let offset = ndim - s_ndim;

            for i in 0..ndim {
                if i >= offset {
                    let s_idx = i - offset;
                    if shape[s_idx] == 1 && target_shape[i] > 1 {
                        out_arr[4 - ndim + i] = 0; // Broadcast
                    } else {
                        out_arr[4 - ndim + i] = strides[s_idx] as u32;
                    }
                } else {
                    out_arr[4 - ndim + i] = 0; // Broadcast new dim
                }
            }
        }

        fill_strides(
            input1_shape,
            input1_strides,
            output_shape,
            &mut stride_1_arr,
        );
        if let Some((_, s2, st2)) = input2 {
            fill_strides(s2, st2, output_shape, &mut stride_2_arr);
        }

        let p = Params {
            numel: output_size as u32,
            op: op as u32,
            alpha: alpha.unwrap_or(0.0),
            ndim: ndim as u32,
            shape: shape_arr,
            strides_out: stride_out_arr,
            strides_1: stride_1_arr,
            strides_2: stride_2_arr,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elementwise Params"),
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let input2_buf = input2.map(|(b, _, _)| b).unwrap_or(&dummy);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Elementwise Bind Group"),
            layout: &ctx.elementwise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input2_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        record_commands(|encoder| {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Elementwise Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.elementwise_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (output_size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        });
    }

    output_buffer
}

pub fn matmul_wgpu_buffer(
    lhs: &wgpu::Buffer,
    lhs_shape: &[usize],
    rhs: &wgpu::Buffer,
    rhs_shape: &[usize],
    activation: Activation,
) -> wgpu::Buffer {
    matmul_fused_wgpu_buffer(lhs, lhs_shape, rhs, rhs_shape, None, activation)
}

pub fn matmul_fused_wgpu_buffer(
    lhs: &wgpu::Buffer,
    lhs_shape: &[usize],
    rhs: &wgpu::Buffer,
    rhs_shape: &[usize],
    bias: Option<(&wgpu::Buffer, &[usize])>,
    activation: Activation,
) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let m = lhs_shape[0] as u32;
    let k = lhs_shape[1] as u32;
    let n = rhs_shape[1] as u32;

    let output_size = (m * n) as usize;
    let output_byte_size = output_size * std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct MatmulParams {
        m: u32,
        k: u32,
        n: u32,
        activation_packed: u32, // activation | (has_bias << 8)
    }

    let act_val = activation as u32;
    let has_bias_val = if bias.is_some() { 1u32 } else { 0u32 };

    let params = MatmulParams {
        m,
        k,
        n,
        activation_packed: act_val | (has_bias_val << 8),
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MatMul Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Handle Bias (Optional)
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Dummy Bias"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bias_buf = bias.map(|(b, _)| b).unwrap_or(&dummy);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MatMul Bind Group"),
        layout: &ctx.matmul_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: bias_buf.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.matmul_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // Workgroup size (16, 16)
        let wg_x = (n + 15) / 16;
        let wg_y = (m + 15) / 16;
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    });

    output_buffer
}

pub fn adam_step_wgpu(
    param: &wgpu::Buffer,
    grad: &wgpu::Buffer,
    m: &wgpu::Buffer,
    v: &wgpu::Buffer,
    numel: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    step: u32,
) {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct AdamParams {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step: u32,
        numel: u32,
        _pad1: u32,
        _pad2: u32,
    }

    let params = AdamParams {
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        numel: numel as u32,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Adam Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Adam Bind Group"),
        layout: &ctx.adam_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: grad.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: m.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Adam Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.adam_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (numel as u32 + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    });
}

pub fn matmul_wgpu(lhs: &[f32], lhs_shape: &[usize], rhs: &[f32], rhs_shape: &[usize]) -> Vec<f32> {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LHS"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RHS"),
        contents: bytemuck::cast_slice(rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buf = matmul_wgpu_buffer(&lhs_buf, lhs_shape, &rhs_buf, rhs_shape, Activation::None);

    // Create Staging Buffer
    let size = (lhs_shape[0] * rhs_shape[1] * 4) as u64;
    let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    record_commands(|encoder| {
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, size);
    });

    flush_queue();

    let slice = staging_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buf.unmap();

    get_memory_pool().return_buffer(out_buf, size);

    result
}

pub fn reduce_sum_dim0_wgpu(input: &wgpu::Buffer, input_shape: &[usize]) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let batch_size = input_shape[0] as u32;
    let input_size = input_shape[1] as u32;
    let stride_batch = input_size;
    let total_input_size: usize = input_shape.iter().product();

    let output_size = input_size as usize;
    let output_byte_size = output_size * std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceParams {
        input_size: u32,
        reduce_dim_size: u32,
        reduce_dim_stride: u32,
        outer_size: u32,
    }

    let params = ReduceParams {
        input_size: total_input_size as u32,
        reduce_dim_size: batch_size,
        reduce_dim_stride: stride_batch,
        outer_size: input_size,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Reduce Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce Bind Group"),
        layout: &ctx.reduce_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reduce Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.reduce_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (input_size + 255) / 256;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    });

    output_buffer
}

pub fn reduce_sum_all_wgpu(input: &wgpu::Buffer, input_size: usize) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let output_byte_size = std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceParams {
        input_size: u32,
        reduce_dim_size: u32,
        reduce_dim_stride: u32,
        outer_size: u32,
    }

    let params = ReduceParams {
        input_size: input_size as u32,
        reduce_dim_size: 0,
        reduce_dim_stride: 0,
        outer_size: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Reduce Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce Bind Group"),
        layout: &ctx.reduce_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reduce All Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.reduce_all_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, 1, 1);
    });

    output_buffer
}

pub fn reduce_sum_dim_wgpu(
    input: &wgpu::Buffer,
    input_shape: &[usize],
    dim: usize,
) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let ndim = input_shape.len();
    if dim >= ndim {
        panic!("Invalid reduction dimension");
    }

    if ndim == 1 || dim == 0 && ndim == 2 {
        return reduce_sum_dim0_wgpu(input, input_shape);
    }

    let total_input_size: usize = input_shape.iter().product();
    let reduce_dim_size = input_shape[dim];

    let outer_size: usize = if dim < ndim - 1 {
        input_shape[dim + 1..].iter().product()
    } else {
        1
    };

    let output_size = total_input_size / reduce_dim_size;
    let output_byte_size = output_size * std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ReduceParams {
        input_size: u32,
        reduce_dim_size: u32,
        reduce_dim_stride: u32,
        outer_size: u32,
    }

    let params = ReduceParams {
        input_size: total_input_size as u32,
        reduce_dim_size: reduce_dim_size as u32,
        reduce_dim_stride: outer_size as u32,
        outer_size: outer_size as u32,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Reduce Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Reduce Bind Group"),
        layout: &ctx.reduce_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Reduce General Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.reduce_general_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((output_size as u32 + 255) / 256).max(1);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    });

    output_buffer
}

pub fn contiguous_wgpu(input: &wgpu::Buffer, shape: &[usize], strides: &[usize]) -> wgpu::Buffer {
    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;

    let size: usize = shape.iter().product();
    let output_byte_size = size * std::mem::size_of::<f32>();
    let output_buffer = get_memory_pool().get_buffer(device, output_byte_size as u64);

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ContiguousParams {
        size: u32,
        ndim: u32,
        pad0: u32,
        pad1: u32,
    }

    let params = ContiguousParams {
        size: size as u32,
        ndim: shape.len() as u32,
        pad0: 0,
        pad1: 0,
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Contiguous Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ShapeStride {
        shape0: u32,
        shape1: u32,
        shape2: u32,
        shape3: u32,
        shape4: u32,
        shape5: u32,
        shape6: u32,
        shape7: u32,
        stride0: u32,
        stride1: u32,
        stride2: u32,
        stride3: u32,
        stride4: u32,
        stride5: u32,
        stride6: u32,
        stride7: u32,
    }

    let shape_stride = ShapeStride {
        shape0: shape.get(0).copied().unwrap_or(1) as u32,
        shape1: shape.get(1).copied().unwrap_or(1) as u32,
        shape2: shape.get(2).copied().unwrap_or(1) as u32,
        shape3: shape.get(3).copied().unwrap_or(1) as u32,
        shape4: shape.get(4).copied().unwrap_or(1) as u32,
        shape5: shape.get(5).copied().unwrap_or(1) as u32,
        shape6: shape.get(6).copied().unwrap_or(1) as u32,
        shape7: shape.get(7).copied().unwrap_or(1) as u32,
        stride0: strides.get(0).copied().unwrap_or(0) as u32,
        stride1: strides.get(1).copied().unwrap_or(0) as u32,
        stride2: strides.get(2).copied().unwrap_or(0) as u32,
        stride3: strides.get(3).copied().unwrap_or(0) as u32,
        stride4: strides.get(4).copied().unwrap_or(0) as u32,
        stride5: strides.get(5).copied().unwrap_or(0) as u32,
        stride6: strides.get(6).copied().unwrap_or(0) as u32,
        stride7: strides.get(7).copied().unwrap_or(0) as u32,
    };

    let shape_strides_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Shape Strides"),
        contents: bytemuck::bytes_of(&shape_stride),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Contiguous Bind Group"),
        layout: &ctx.contiguous_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shape_strides_buffer.as_entire_binding(),
            },
        ],
    });

    record_commands(|encoder| {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Contiguous Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.contiguous_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((size as u32 + 255) / 256).max(1);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    });

    output_buffer
}
