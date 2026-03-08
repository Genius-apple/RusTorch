# RusTorch Examples 🧪

This directory contains examples demonstrating the capabilities of RusTorch.

## Runnable Examples

To run these examples, you should use `cargo run -p <package> --example <example_name>`.

### 🧠 Neural Networks (`rustorch-nn`)

Located in [`rustorch-nn/examples`](../rustorch-nn/examples/):

*   **`full_training_demo`**: A complete CNN training loop (Conv2d -> ReLU -> MaxPool -> Linear) on dummy data.
    ```bash
    cargo run -p rustorch-nn --example full_training_demo
    ```
*   **`serialization`**: Demonstrates saving and loading models using Serde.
    ```bash
    cargo run -p rustorch-nn --example serialization
    ```
*   **`tracing`**: Shows how to use the JIT tracer to optimize a model.
    ```bash
    cargo run -p rustorch-nn --example tracing
    ```

### ⚡ Core & JIT (`rustorch-core`)

Located in [`rustorch-core/examples`](../rustorch-core/examples/):

*   **`jit_demo`**: Demonstrates graph construction, operator fusion (Conv+ReLU), and execution.
    ```bash
    cargo run -p rustorch-core --example jit_demo
    ```

---

## Featured Code: CNN Training

Here is a snippet of what the training code looks like (from `full_training_demo.rs`):

```rust
// Define a simple CNN
struct CNN {
    conv1: Conv2d,
    fc: Linear,
}

impl Module for CNN {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv1.forward(x).relu().max_pool2d((2,2), (2,2), (0,0));
        let x = x.reshape(&[x.shape()[0], 16 * 14 * 14]);
        self.fc.forward(&x)
    }
}

// Training Loop
let model = CNN::new();
let optimizer = SGD::new(model.parameters(), 0.01);

for epoch in 0..10 {
    let output = model.forward(&input);
    let loss = criterion.forward(&output, &target);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
}
```
