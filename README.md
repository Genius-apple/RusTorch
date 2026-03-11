# RusTorch 🦀🔥

[![Build Status](https://img.shields.io/github/actions/workflow/status/Genius-apple/RusTorch/ci.yml?branch=main)](https://github.com/Genius-apple/RusTorch/actions)
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

## 📦 Ecosystem & Architecture

RusTorch is a modular workspace designed for scalability. We adopt a "Core + Plugins" architecture to ensure lightweight runtime and maximum extensibility.

### 🧩 Project Structure

```mermaid
mindmap
  root((RusTorch))
    Core(rustorch-core)
      Tensor Engine
      Autograd
      JIT Compiler
    NN(rustorch-nn)
      Layers
      Optimizers
      Loss Functions
    Backends
      CUDA(rustorch-cuda)
      WGPU(rustorch-wgpu)
      Vulkan(rustorch-vulkan)
      Metal(rustorch-metal)
    Ecosystem
      Vision(rustorch-vision)
      Text(rustorch-text)
      Audio(rustorch-audio)
    Interop
      PyTorch(rustorch-pytorch)
      ONNX(rustorch-onnx)
      WASM(rustorch-wasm)
```

*   **`rustorch-core`**: The heart. N-dimensional Tensors, Autograd engine, and JIT compiler.
*   **`rustorch-nn`**: Neural network building blocks (Conv2d, LSTM, Transformer), Loss functions, and Optimizers.
*   **`rustorch-vision`**: Computer vision datasets (MNIST, CIFAR) and transforms.
*   **`rustorch-text`**: NLP primitives, Tokenizers, and Vocab.
*   **`rustorch-cuda`**: High-performance CUDA kernels.
*   **`rustorch-wasm`**: Run your models directly in the browser.
*   **`rustorch-pytorch`**: 🌉 **NEW!** Bridge to PyTorch ecosystem. Load `.pth` files and interop with LibTorch.
*   **`rustorch-wgpu`**: 🌐 **NEW!** WebGPU backend for browser and cross-platform GPU acceleration.
*   **`rustorch-vulkan`**: 🎮 **NEW!** Vulkan compute backend for high-performance graphics hardware.

### ✅ Feature Matrix

| Feature | RusTorch | PyTorch | TensorFlow |
| :--- | :---: | :---: | :---: |
| **Memory Safety** | 🛡️ **Guaranteed** | ❌ (C++) | ❌ (C++) |
| **GIL-Free** | 🚀 **Yes** | ❌ No | ❌ No |
| **WebGPU Support** | 🌐 **Native** | 🚧 Experimental | 🚧 Experimental |
| **Browser Inference** | ✅ **WASM + WebGPU** | ❌ Heavy | ✅ TFLite |
| **API Style** | 🔥 **Pythonic** | 🔥 Pythonic | 📉 Verbose |
| **Deployment** | 📦 **Single Binary** | 🐍 Python Env | 🐍 Python Env |

---

## 🌐 Universal Architecture

RusTorch isn't just a library; it's a universal tensor compiler.

```mermaid
graph TD
    %% Styling
    classDef core fill:#e85d04,stroke:#333,stroke-width:2px,color:white;
    classDef backend fill:#8338ec,stroke:#333,stroke-width:2px,color:white;
    classDef interop fill:#3a86ff,stroke:#333,stroke-width:2px,color:white;
    classDef user fill:#fb5607,stroke:#333,stroke-width:2px,color:white;

    User["👤 User Application"]:::user --> API["🔥 RusTorch API"]:::core
    API --> Core["🧠 rustorch-core"]:::core
    
    subgraph Compute_Backends ["⚙️ Compute Backends"]
        direction TB
        Core -.-> CPU["🖥️ Rayon CPU"]:::backend
        Core -.-> CUDA["🚀 CUDA (NVidia)"]:::backend
        Core -.-> WGPU["🌐 WebGPU (Browser)"]:::backend
        Core -.-> Vulkan["🎮 Vulkan (Cross-Platform)"]:::backend
    end
    
    subgraph Interoperability ["🔌 Interoperability"]
        direction TB
        PyTorch["🔥 PyTorch Ecosystem"]:::interop <-->|rustorch-pytorch| Core
        Model["💾 .pth Models"]:::interop <-->|Load/Save| Core
    end
```

### 🌉 PyTorch Bridge (`rustorch-pytorch`)

Seamlessly switch between RusTorch and PyTorch. No more rewriting models from scratch.

*   **🔄 Zero-Copy Conversion**: Convert `rustorch::Tensor` <-> `torch::Tensor` instantly.
*   **💾 Model Loading**: Load pre-trained `.pth` weights directly into RusTorch models.
*   **🛡️ Operator Fallback**: Use PyTorch's battle-tested operators when RusTorch implementation is missing.

```rust
use rustorch_pytorch::PyTorchAdapter;

// Load a PyTorch model checkpoint
let weights = PyTorchAdapter::load_state_dict("resnet18.pth")?;

// Run inference in RusTorch
let input = Tensor::randn(&[1, 3, 224, 224]);
// let output = model.forward(&input);
```

### 🎮 Graphics-Ready Compute (`rustorch-wgpu` & `rustorch-vulkan`)

Unlock the power of your GPU, anywhere.

*   **WebGPU Backend**: Run large language models directly in the browser with near-native performance.
*   **Vulkan Backend**: Cross-vendor GPU support (AMD, Intel, NVIDIA, Mobile) with low-level control.

---

## 🛠️ Quick Start

Add RusTorch to your `Cargo.toml`:

```toml
[dependencies]
rus-torch = "0.1.1"
```

### 🔥 Train a Model in 30 Seconds

```mermaid
sequenceDiagram
    autonumber
    participant Data as 💿 Dataset
    participant Model as 🧠 Model
    participant Loss as 📉 Loss Fn
    participant Optim as ⚙️ Optimizer

    loop Training Epochs
        Data->>Model: Forward(Batch)
        Model->>Loss: Compute Loss(Pred, Target)
        Loss-->>Model: Backward() (Compute Gradients)
        Optim->>Model: Step() (Update Weights)
        Optim->>Model: ZeroGrad()
    end
```

```rust
use rus_torch::core::Tensor;
use rus_torch::nn::{Linear, Module, CrossEntropyLoss, SGD};

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
