use rustorch_core::Tensor;
use std::time::Instant;

fn main() {
    let size = 2048;
    println!("Benchmarking Fused MatMul+ReLU [{}, {}] ...", size, size);

    // Create tensors on CPU
    let lhs_cpu = Tensor::full(&[size, size], 1.0);
    let rhs_cpu = Tensor::full(&[size, size], 1.0);

    // Move to WGPU
    println!("Uploading to GPU...");
    let lhs = lhs_cpu.to_wgpu();
    let rhs = rhs_cpu.to_wgpu();

    // Warmup
    let _ = lhs.matmul_relu(&rhs);

    // Compute loop
    println!("Running Compute Loop...");
    let start_compute = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        let _out = lhs.matmul_relu(&rhs);
    }

    // Sync
    let out = lhs.matmul_relu(&rhs);
    let _cpu_res = out.to_cpu();

    let compute_time = start_compute.elapsed();
    let avg_time = compute_time.as_secs_f32() / (iters as f32 + 1.0);

    println!("Total Loop Time: {:.6} s", compute_time.as_secs_f32());
    println!("Avg Fused Op Time: {:.6} s", avg_time);

    let gflops = (2.0 * (size as f64).powi(3)) / avg_time as f64 / 1e9;
    println!("Throughput: {:.2} GFLOPS", gflops);
}
