use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rustorch_core::{ops, Tensor};

fn matmul_benchmark(c: &mut Criterion) {
    let size = 256usize;
    let a = Tensor::new(&vec![1.0f32; size * size], &[size, size]);
    let b = Tensor::new(&vec![1.0f32; size * size], &[size, size]);
    let mut group = c.benchmark_group("matmul");
    group.throughput(Throughput::Elements((size * size * size) as u64));
    group.bench_function("forward_256x256", |bencher| {
        bencher.iter(|| black_box(a.matmul(&b)))
    });
    group.bench_function("backward_256x256", |bencher| {
        bencher.iter_batched(
            || {
                let a =
                    Tensor::new(&vec![1.0f32; size * size], &[size, size]).set_requires_grad(true);
                let b =
                    Tensor::new(&vec![1.0f32; size * size], &[size, size]).set_requires_grad(true);
                (a, b)
            },
            |(a, b)| {
                let out = a.matmul(&b);
                let loss = ops::sum(&out);
                loss.backward();
                black_box((a, b, loss));
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn conv_benchmark(c: &mut Criterion) {
    let n = 8usize;
    let c_in = 16usize;
    let c_out = 32usize;
    let h = 32usize;
    let w = 32usize;
    let k = 3usize;
    let input = Tensor::new(&vec![0.1f32; n * c_in * h * w], &[n, c_in, h, w]);
    let weight = Tensor::new(&vec![0.1f32; c_out * c_in * k * k], &[c_out, c_in, k, k]);
    let mut group = c.benchmark_group("conv2d");
    group.throughput(Throughput::Elements((n * c_out * h * w) as u64));
    group.bench_function("forward_n8_c16_32_hw32_k3", |bencher| {
        bencher.iter(|| black_box(ops::conv2d(&input, &weight, (1, 1), (1, 1))))
    });
    group.finish();
}

criterion_group!(benches, matmul_benchmark, conv_benchmark);
criterion_main!(benches);
