use rustorch_core::Tensor;
use std::time::Instant;

fn main() {
    let m = 64;
    let k = 784;
    let n = 128;
    println!(
        "Benchmarking Small MatMul [{}, {}] x [{}, {}] ...",
        m, k, k, n
    );

    let a = Tensor::ones(&[m, k]);
    let b = Tensor::ones(&[k, n]);

    // Warmup
    let _ = a.matmul(&b);

    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        let _ = a.matmul(&b);
    }
    let duration = start.elapsed();

    println!(
        "Rust Average Time: {:.6} s",
        duration.as_secs_f32() / iters as f32
    );
}
