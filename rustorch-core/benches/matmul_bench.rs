use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustorch_core::Tensor;

fn matmul_benchmark(c: &mut Criterion) {
    let size = 128;
    let a = Tensor::new(&vec![1.0; size * size], &[size, size]);
    let b = Tensor::new(&vec![1.0; size * size], &[size, size]);
    
    c.bench_function("matmul 128x128", |bencher| {
        bencher.iter(|| {
            black_box(a.matmul(&b))
        })
    });
}

criterion_group!(benches, matmul_benchmark);
criterion_main!(benches);
