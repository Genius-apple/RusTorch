use rustorch_core::Tensor;
use std::time::Instant;

fn main() {
    let size = 1024;
    println!("Benchmarking MatMul [{}, {}] ...", size, size);

    // Use ones instead of randn for simplicity if randn is missing
    let a = Tensor::ones(&[size, size]);
    let b = Tensor::ones(&[size, size]);

    // Warmup
    let _ = a.matmul(&b);

    let start = Instant::now();
    for _ in 0..10 {
        let _ = a.matmul(&b);
    }
    let duration = start.elapsed();

    println!("Rust Average Time: {:.4} s", duration.as_secs_f32() / 10.0);
}
