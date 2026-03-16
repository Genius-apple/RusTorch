use rustorch_core::Tensor;
use std::time::Instant;

fn main() {
    let size = 2048;
    println!("Benchmarking WGPU Resident MatMul [{}, {}] ...", size, size);

    // Create tensors on CPU
    let lhs_cpu = Tensor::full(&[size, size], 1.0);
    let rhs_cpu = Tensor::full(&[size, size], 1.0);

    // Move to WGPU (Upload happens here)
    println!("Uploading to GPU...");
    let start_upload = Instant::now();
    let lhs = lhs_cpu.to_wgpu();
    let rhs = rhs_cpu.to_wgpu();
    let upload_time = start_upload.elapsed();
    println!("Upload Time: {:.6} s", upload_time.as_secs_f32());

    // Warmup
    let _ = lhs.matmul(&rhs);

    // Compute loop (Resident on GPU)
    println!("Running Compute Loop...");
    let start_compute = Instant::now();
    let iters = 50;
    for _ in 0..iters {
        let _out = lhs.matmul(&rhs);
        // Note: matmul is async submission.
        // But we are creating new output buffers every time.
        // We should synchronize to measure time correctly?
        // Actually, wgpu queue submission is fast.
        // To measure kernel time, we need to wait for device idle.
        // But `Tensor` operations don't expose device poll.
        // However, `matmul` returns a Tensor which holds a buffer.
        // The command is submitted.
    }

    // Force synchronization by downloading the last result
    let out = lhs.matmul(&rhs);
    let _cpu_res = out.to_cpu(); // This will block until done

    let compute_time = start_compute.elapsed();
    // This time includes the last download.
    // Let's approximate kernel time by subtracting download time from previous benchmark?
    // Or just divide by iters+1 roughly.

    println!(
        "Total Loop Time (incl. 1 download): {:.6} s",
        compute_time.as_secs_f32()
    );
    println!(
        "Avg Op Time: {:.6} s",
        compute_time.as_secs_f32() / (iters as f32 + 1.0)
    );

    let avg_time = compute_time.as_secs_f32() / (iters as f32 + 1.0);
    let gflops = (2.0 * (size as f64).powi(3)) / avg_time as f64 / 1e9;
    println!("Throughput: {:.2} GFLOPS", gflops);
}
