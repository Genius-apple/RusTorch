# Zero to Hero with RusTorch: A Beginner's Guide 🚀

Welcome to RusTorch! If you are coming from Python/PyTorch, you will feel right at home. If you are new to Deep Learning or Rust, this guide will walk you through the basics step-by-step.

## Table of Contents
1.  [Installation](#1-installation)
2.  [Tensor Basics](#2-tensor-basics)
3.  [Building Neural Networks](#3-building-neural-networks)
4.  [Training Loop](#4-training-loop)
5.  [Advanced: JIT Compilation](#5-advanced-jit-compilation)

---

## 1. Installation

First, ensure you have Rust installed. If not, get it at [rust-lang.org](https://www.rust-lang.org/tools/install).

Create a new project:
```bash
cargo new rustorch-demo
cd rustorch-demo
```

Add RusTorch to your `Cargo.toml`. Since we are using the local version or git version for now:

```toml
[dependencies]
rustorch-core = { path = "../rustorch/rustorch-core" } # Adjust path if needed
rustorch-nn = { path = "../rustorch/rustorch-nn" }
```

---

## 2. Tensor Basics

Tensors are the fundamental building blocks. They are multi-dimensional arrays with super-powers (Autograd!).

### Creating Tensors

```rust
use rustorch_core::Tensor;

fn main() {
    // Create a 2x3 tensor filled with zeros
    let a = Tensor::zeros(&[2, 3]);
    println!("Zeros:\n{}", a);

    // Create from data
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    println!("From data:\n{}", b);
    
    // Random tensor (Normal distribution)
    let c = Tensor::zeros(&[2, 2]);
    c.normal_(0.0, 1.0); // Mean 0, Std 1
    println!("Random:\n{}", c);
}
```

### Math Operations

RusTorch supports standard arithmetic. Broadcasting works just like NumPy!

```rust
let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
let y = Tensor::new(&[3.0, 4.0], &[1, 2]);

let z = &x + &y; // Add
let w = &x * &y; // Element-wise Mul
let m = x.matmul(&y.t()); // Matrix Multiplication
```

### Autograd (Automatic Differentiation)

The magic of deep learning! RusTorch tracks operations to compute gradients automatically.

```rust
// 1. Create tensors that require gradients
let x = Tensor::new(&[2.0], &[1]).set_requires_grad(true);
let w = Tensor::new(&[3.0], &[1]).set_requires_grad(true);
let b = Tensor::new(&[1.0], &[1]).set_requires_grad(true);

// 2. Compute: y = w * x + b
let y = &w * &x + &b;

// 3. Backward pass
y.backward();

// 4. Check gradients
// dy/dw = x = 2.0
// dy/dx = w = 3.0
// dy/db = 1.0
println!("dL/dw: {:?}", w.grad()); 
println!("dL/dx: {:?}", x.grad());
```

---

## 3. Building Neural Networks

`rustorch-nn` provides layers like `Linear`, `Conv2d`, `RNN`, etc.

### Defining a Custom Module

In Rust, we define a struct and implement the `Module` trait.

```rust
use rustorch_core::Tensor;
use rustorch_nn::{Module, Linear};

struct MyModel {
    fc1: Linear,
    fc2: Linear,
}

impl MyModel {
    fn new() -> Self {
        Self {
            fc1: Linear::new(10, 32), // 10 inputs -> 32 hidden
            fc2: Linear::new(32, 2),  // 32 hidden -> 2 outputs
        }
    }
}

impl Module for MyModel {
    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.fc1.forward(input).relu(); // Activation
        self.fc2.forward(&x)
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        // Collect parameters from sub-modules
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }
}
```

---

## 4. Training Loop

Let's put it all together to train a model.

```rust
use rustorch_nn::{SGD, CrossEntropyLoss};

fn train() {
    let model = MyModel::new();
    let criterion = CrossEntropyLoss::new();
    // Optimizer takes ownership of parameters or references? 
    // In RusTorch v0.1, we pass a list of parameters.
    let mut optimizer = SGD::new(model.parameters(), 0.01); // lr = 0.01
    
    // Mock Data
    let input = Tensor::new(&[0.0; 10], &[1, 10]);
    let target = Tensor::new(&[1.0], &[1]); // Class 1
    
    for epoch in 0..10 {
        optimizer.zero_grad(); // Reset gradients
        
        let output = model.forward(&input);
        let loss = criterion.forward(&output, &target);
        
        loss.backward(); // Backprop
        optimizer.step(); // Update weights
        
        println!("Epoch {}: Loss = {:?}", epoch, loss);
    }
}
```

---

## 5. Advanced: JIT Compilation

RusTorch isn't just an interpreter. It can compile your graph for speed!

```rust
use rustorch_core::jit::{Graph, NodeType, Optimizer, Executor};

fn jit_demo() {
    let mut graph = Graph::new();
    // ... define nodes ...
    
    // Magic happens here:
    Optimizer::optimize(&mut graph); 
    // Detects patterns like Conv2d -> ReLU and fuses them into a single kernel!
    
    let result = Executor::run(&graph, inputs);
}
```

---

## Next Steps

*   Check out the `examples/` directory for full code.
*   Try implementing a ResNet!
*   Contribute your own layers to `rustorch-nn`.

Happy Coding! 🦀
