use rustorch_core::backend::wgpu::get_context;
use std::time::Instant;
use wgpu::util::DeviceExt;

fn main() {
    let size = 2048; // Increase size to show GPU scaling
    println!("Benchmarking WGPU MatMul [{}, {}] ...", size, size);

    // Prepare data
    let lhs = vec![1.0; size * size];
    let rhs = vec![1.0; size * size];

    let ctx = get_context().expect("WGPU not initialized");
    let device = &ctx.device;
    let queue = &ctx.queue;

    // --- Upload ---
    let start_upload = Instant::now();
    let lhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("LHS Buffer"),
        contents: bytemuck::cast_slice(&lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("RHS Buffer"),
        contents: bytemuck::cast_slice(&rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let _upload_time = start_upload.elapsed();

    let output_size = (size * size) * std::mem::size_of::<f32>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = [size as u32, size as u32, size as u32, 0];
    let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Param Buffer"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MatMul Bind Group"),
        layout: &ctx.matmul_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    // --- Compute Kernel Loop ---
    println!("Running Compute Loop...");
    let start_compute = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.matmul_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch (m/64, n/64, 1) because workgroup is 16x16 threads each computing 4x4 elements
            cpass.dispatch_workgroups((size as u32).div_ceil(64), (size as u32).div_ceil(64), 1);
        }
        queue.submit(Some(encoder.finish()));
    }
    device.poll(wgpu::Maintain::Wait);
    let compute_time = start_compute.elapsed();

    println!(
        "Compute Time (avg): {:.6} s",
        compute_time.as_secs_f32() / iters as f32
    );

    // GFLOPS calculation: 2 * M * N * K
    let gflops = (2.0 * (size as f64).powi(3)) / (compute_time.as_secs_f64() / iters as f64) / 1e9;
    println!("Throughput: {:.2} GFLOPS", gflops);
}
