# RusTorch 🦀🔥

[![Build Status](https://img.shields.io/github/actions/workflow/status/trae-ai/rustorch/ci.yml?branch=main)](https://github.com/trae-ai/rustorch/actions)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/rustorch-core.svg)](https://crates.io/crates/rustorch-core)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

> **PyTorch's API. Rust's Safety. The Next-Gen AI Infrastructure.**

**RusTorch** is a production-grade deep learning framework re-imagined in Rust. It combines the **usability** you love from PyTorch with the **performance, safety, and concurrency** guarantees of Rust. Say goodbye to GIL locks, GC pauses, and runtime errors. Say hello to **RusTorch**.

---

## 🚀 Why RusTorch?

*   **⚡ Blazing Fast**: Powered by `Rayon` for parallel CPU execution and optimized CUDA kernels (coming soon) for GPU. Zero-cost abstractions mean you pay for what you use.
*   **🛡️ Memory Safe**: Leveraging Rust's ownership model, RusTorch ensures memory safety without the overhead of a Garbage Collector. No more segfaults in production.
*   **🧠 PyTorch-like API**: If you know PyTorch, you already know RusTorch. We've meticulously mirrored the API design so you can switch instantly.
*   **🔮 JIT Graph Optimization**: Built-in XLA-style compiler that traces your code, fuses operators (e.g., Conv2d + ReLU), and eliminates dead code for maximum efficiency.
*   **🌐 Distributed Ready**: Native `DistributedDataParallel` support designed for modern multi-gpu, multi-node training clusters.

---

## 📦 Ecosystem

RusTorch is a modular workspace designed for scalability:

*   **`rustorch-core`**: The heart. N-dimensional Tensors, Autograd engine, and JIT compiler.
*   **`rustorch-nn`**: Neural network building blocks (Conv2d, LSTM, Transformer), Loss functions, and Optimizers.
*   **`rustorch-vision`**: Computer vision datasets (MNIST, CIFAR) and transforms.
*   **`rustorch-text`**: NLP primitives, Tokenizers, and Vocab.
*   **`rustorch-cuda`**: High-performance CUDA kernels.
*   **`rustorch-wasm`**: Run your models directly in the browser.

---

## 🛠️ Quick Start

Add RusTorch to your `Cargo.toml`:

```toml
[dependencies]
rustorch-core = { git = "https://github.com/your-username/rustorch" }
rustorch-nn = { git = "https://github.com/your-username/rustorch" }
```

### 🔥 Train a Model in 30 Seconds

```rust
use rustorch_core::Tensor;
use rustorch_nn::{Linear, Module, CrossEntropyLoss, SGD};

fn main() {
    // 1. Define a simple model
    let fc = Linear::new(10, 2); // Input: 10, Output: 2 classes
    
    // 2. Setup Loss & Optimizer
    let criterion = CrossEntropyLoss::new();
    let mut optimizer = SGD::new(fc.parameters(), 0.01);

    // 3. Dummy Data (Batch Size: 1, Features: 10)
    let input = Tensor::new(&[0.5; 10], &[1, 10]).set_requires_grad(true);
    let target = Tensor::new(&[1.0], &[1]); // Target Class 1

    // 4. Training Step
    optimizer.zero_grad();
    let output = fc.forward(&input);
    let loss = criterion.forward(&output, &target);
    loss.backward();
    optimizer.step();

    println!("🎉 Training step complete! Loss: {:?}", loss);
}
```

---

## 🎓 Documentation & Tutorials

*   **[Zero to Hero Tutorial](TUTORIAL.md)**: The best place to start for beginners.
*   **[Architecture Guide](ARCHITECTURE.md)**: Deep dive into RusTorch's internals.
*   **[Examples](examples/)**: Real-world examples including CNNs, RNNs, and JIT usage.

---

## 🤝 Contributing

We are building the future of AI in Rust, and we need YOU! Whether it's adding new operators, fixing bugs, or improving docs, all contributions are welcome.

Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📜 License

RusTorch is open-source software licensed under the [MIT](LICENSE) or [Apache-2.0](LICENSE) license.

<div align="center">
  <sub>Built with ❤️ by the Rust AI Community</sub>
</div>
