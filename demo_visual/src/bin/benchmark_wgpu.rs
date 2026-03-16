use rustorch_core::backend::wgpu::matmul_wgpu;
use std::time::Instant;

fn main() {
    let size = 1024;
    println!("Benchmarking WGPU MatMul [{}, {}] ...", size, size);

    let lhs = vec![1.0; size * size];
    let rhs = vec![1.0; size * size];

    // Warmup
    let _ = matmul_wgpu(&lhs, &[size, size], &rhs, &[size, size]);

    let start = Instant::now();
    for _ in 0..10 {
        let _ = matmul_wgpu(&lhs, &[size, size], &rhs, &[size, size]);
    }
    let duration = start.elapsed();

    println!("WGPU Average Time: {:.4} s", duration.as_secs_f32() / 10.0);
}
